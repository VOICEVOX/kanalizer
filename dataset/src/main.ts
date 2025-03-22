import * as fs from "node:fs/promises";
import { Semaphore } from "@core/asyncutil/semaphore";
import { load as loadYaml } from "js-yaml";
import { type Config, configSchema } from "./config.ts";
import type { InferenceProvider } from "./inference/index.ts";
import { GeminiInferenceProvider } from "./inference/gemini.ts";
import { OpenAIInferenceProvider } from "./inference/openai.ts";
import { DummyInferenceProvider } from "./inference/dummy.ts";
import { Random } from "./random.ts";
import { CmuDictSourceProvider } from "./source/cmudict.ts";
import type { SourceProvider } from "./source/index.ts";
import {
  ExhaustiveError,
  bisectMax,
  filterPronunciations,
  sleep,
} from "./utils.ts";

async function main() {
  const config = await loadConfig();

  let sourceProvider: SourceProvider;
  switch (config.source.provider) {
    case "cmudict":
      sourceProvider = new CmuDictSourceProvider();
      break;
    default:
      throw new ExhaustiveError(config.source.provider);
  }
  console.log(`Source provider: ${config.source.provider}`);

  let inferenceProvider: InferenceProvider;
  switch (config.inference.provider) {
    case "gemini":
      inferenceProvider = new GeminiInferenceProvider(config);
      break;
    case "openai":
      inferenceProvider = new OpenAIInferenceProvider(config);
      break;
    case "dummy":
      inferenceProvider = new DummyInferenceProvider(config);
      break;
    default:
      throw new ExhaustiveError(config.inference.provider);
  }
  console.log(`Inference provider: ${config.inference.provider}`);

  const random = new Random(config.randomSeed);

  console.log("1: Loading words...");
  const words = await loadWords({
    sourceProvider,
    maxNumWords: config.source.maxNumWords,
    random,
  });
  if (words.length <= 10) {
    console.error(`Too few words: ${words.length}`);
    return;
  }

  console.log("2: Determining batch size...");
  const batchSize = await determineBatchSize({
    batchConfig: config.inference.batch,
    inferenceProvider,
    words,
    random,
  });

  console.log("3: Inferring pronunciations...");
  const allResults = await inferPronunciations({
    inferenceProvider,
    concurrency: config.inference.concurrency,
    words,
    batchSize,
    random,
    rateLimit: config.inference.rateLimit,
  });

  console.log("4: Writing results...");
  const path = `${import.meta.dirname}/../../train/vendor/data.jsonl`;
  await writeResults({ path, results: allResults });

  console.log(
    `${allResults.size} pronunciations inferred and written to ${path}`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

async function determineBatchSize(params: {
  batchConfig: Config["inference"]["batch"];
  inferenceProvider: InferenceProvider;
  words: string[];
  random: Random;
}) {
  let batchSize: number;
  switch (params.batchConfig.type) {
    case "fixed": {
      console.log("Using fixed batch size");
      batchSize = params.batchConfig.batchSize;
      break;
    }
    case "bisect": {
      console.log("Finding optimal batch size by bisection...");
      // ちょっと余裕を持たせる
      const maxBatchSize = await findMaxBatchSize({
        inferenceProvider: params.inferenceProvider,
        words: params.words,
        random: params.random,
        maxBatchSize: params.batchConfig.maxBatchSize,
      });
      batchSize = Math.floor(maxBatchSize * params.batchConfig.ratio);
      break;
    }
    default:
      throw new ExhaustiveError(params.batchConfig);
  }
  console.log(`Batch size: ${batchSize}`);
  return batchSize;
}

async function loadConfig() {
  return configSchema.parse(
    loadYaml(
      await fs.readFile(`${import.meta.dirname}/../config.yml`, "utf-8"),
    ),
  );
}

async function loadWords(params: {
  sourceProvider: SourceProvider;
  maxNumWords: number | "all";
  random: Random;
}) {
  let words = await params.sourceProvider.getWords();
  console.log(`Loaded ${words.length} words`);
  if (params.maxNumWords !== "all") {
    console.log(`Shuffling and limiting to ${params.maxNumWords} words...`);
    words = params.random.shuffle(words).slice(0, params.maxNumWords);
  }

  return words;
}

async function findMaxBatchSize(params: {
  inferenceProvider: InferenceProvider;
  words: string[];
  random: Random;
  maxBatchSize: number;
}) {
  const maxPossibleBatchSize = await bisectMax(
    1,
    Math.min(params.words.length, params.maxBatchSize),
    async (batchSize) => {
      console.log(`Trying batch size ${batchSize}...`);
      const currentWords = params.random
        .shuffle(params.words)
        .slice(0, batchSize);
      const results = await params.inferenceProvider
        .infer(currentWords)
        .catch((err) => {
          console.error(err);
          return {};
        });
      return Object.keys(results).length === batchSize;
    },
  );
  console.log(`Found maximum batch size: ${maxPossibleBatchSize}`);

  if (maxPossibleBatchSize < 10) {
    throw new Error(`Batch size too small: ${maxPossibleBatchSize}`);
  }
  return maxPossibleBatchSize;
}

async function inferPronunciations(params: {
  inferenceProvider: InferenceProvider;
  concurrency: number;
  words: string[];
  batchSize: number;
  random: Random;
  rateLimit: Config["inference"]["rateLimit"];
}) {
  const semaphore = new Semaphore(params.concurrency);
  console.log(`Using ${params.concurrency} concurrency`);

  const allResults = new Map<string, string>();

  const shuffledWords = params.random.shuffle(params.words);

  const inferBatch = (words: string[]) =>
    semaphore.lock(async () => {
      await sleep(params.rateLimit.throttleMs);

      const results = await params.inferenceProvider.infer(words);

      const validResults = filterPronunciations(results);

      for (const [word, pronunciation] of Object.entries(validResults)) {
        allResults.set(word, pronunciation);
      }

      console.log(
        `Inferred ${Object.keys(results).length} pronunciations, ${
          Object.keys(validResults).length
        } valid, ${words.length - Object.keys(validResults).length} invalid, ${
          shuffledWords.length - allResults.size
        } remaining`,
      );
    });

  let numTries = 0;
  while (allResults.size < shuffledWords.length) {
    const remainingWords = shuffledWords.filter(
      (word) => !allResults.has(word),
    );
    const promises: Promise<void>[] = [];

    while (remainingWords.length > 0) {
      const currentWords = remainingWords.splice(0, params.batchSize);

      promises.push(inferBatch(currentWords));
    }
    console.log(`Waiting for ${promises.length} batches...`);

    const results = await Promise.allSettled(promises);

    const isAllFulfilled = results.every(
      (result) => result.status === "fulfilled",
    );
    if (!isAllFulfilled) {
      const errors = results.flatMap((result) =>
        result.status === "rejected" ? [result.reason] : [],
      );
      const error = new AggregateError(errors);
      if (errors.some((err) => !String(err).includes("429"))) {
        throw error;
      }

      console.error(`Rate limited, waiting ${params.rateLimit.waitMs}ms...`);
      console.error(error);
      await sleep(params.rateLimit.waitMs);
    }

    numTries++;
    if (numTries > params.rateLimit.maxRetries) {
      throw new Error("Too many retries");
    }
  }

  return allResults;
}

async function writeResults(params: {
  path: string;
  results: Map<string, string>;
}) {
  await fs.writeFile(
    params.path,
    [...params.results]
      .map(([word, pronunciation]) =>
        JSON.stringify({
          word,
          kata: [pronunciation],
        }),
      )
      .join("\n"),
  );
}
