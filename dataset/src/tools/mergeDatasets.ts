// データセットをマージするスクリプト。
//
// 使い方:
// pnpm run tools:mergeDatasets dataset1.jsonl dataset2.jsonl ... output.jsonl
//
import fs from "node:fs/promises";
import { z } from "zod";

const datasetLineSchema = z.object({
  word: z.string(),
  kata: z.array(z.string()).length(1),
});
type DatasetLine = z.infer<typeof datasetLineSchema>;

async function main() {
  const datasetPaths = process.argv.slice(2, -1);
  const outputPath = process.argv[process.argv.length - 1];

  if (datasetPaths.length === 0) {
    throw new Error("No dataset files provided.");
  }

  const dataset = await mergeDatasets(datasetPaths);

  await writeDataset(dataset, outputPath, datasetPaths);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

async function mergeDatasets(datasets: string[]) {
  const dataset = new Map<string, string>();
  for (const datasetPath of datasets) {
    const content = await fs.readFile(datasetPath, "utf-8");
    const parsedContent = content
      .split("\n")
      .map((line) => datasetLineSchema.parse(JSON.parse(line)));

    for (const { word, kata } of parsedContent) {
      dataset.set(word, kata[0]);
    }
  }

  return dataset;
}

async function writeDataset(
  dataset: Map<string, string>,
  outputPath: string,
  datasets: string[],
) {
  const outputContent = Array.from(dataset.entries())
    .map(([word, kata]) =>
      JSON.stringify({ word, kata: [kata] } satisfies DatasetLine),
    )
    .join("\n");
  await fs.writeFile(outputPath, outputContent, "utf-8");
  console.log(`Merged ${datasets.length} datasets into ${outputPath}`);
}
