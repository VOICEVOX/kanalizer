import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";
import { z } from "zod";
import { filterPronunciations, getSuspiciousWordReasons } from "../utils.ts";

const datasetLineSchema = z.object({
  word: z.string(),
  kata: z.array(z.string()).length(1),
});

async function main() {
  const datasetPath = process.argv[2];
  if (datasetPath == null) {
    throw new Error("Usage: pnpm run tools:validateDataset <dataset.jsonl>");
  }

  let lineCount = 0;
  let suspiciousWordCount = 0;
  let invalidPronunciationCount = 0;

  const rl = createInterface({
    input: createReadStream(datasetPath, "utf-8"),
    crlfDelay: Number.POSITIVE_INFINITY,
  });

  for await (const line of rl) {
    if (line.trim().length === 0) continue;

    const { word, kata } = datasetLineSchema.parse(JSON.parse(line));
    lineCount++;

    const suspiciousReasons = getSuspiciousWordReasons(word);
    if (suspiciousReasons.length > 0) {
      suspiciousWordCount++;
      console.warn(
        `Warning: suspicious dataset word: ${word} (${suspiciousReasons.join(",")})`,
      );
    }

    const filtered = filterPronunciations({ [word]: kata[0] });
    if (!(word in filtered)) {
      invalidPronunciationCount++;
      console.warn(`Warning: invalid pronunciation in dataset: ${word}`);
    }
  }

  console.log(`Validated ${lineCount} lines in ${datasetPath}`);
  console.log(
    `Warnings: suspiciousWord=${suspiciousWordCount}, invalidPronunciation=${invalidPronunciationCount}`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
