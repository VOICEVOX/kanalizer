import fs from "node:fs/promises";

type DatasetLine = {
  word: string;
  kata: string[];
};

async function main() {
  const datasets = process.argv.slice(2, -1);
  const outputPath = process.argv[process.argv.length - 1];

  if (datasets.length === 0) {
    throw new Error("No dataset files provided.");
  }

  const dataset = new Map<string, string>();
  for (const datasetPath of datasets) {
    const content = await fs.readFile(datasetPath, "utf-8");
    const parsedContent = content
      .split("\n")
      .map((line) => JSON.parse(line)) as DatasetLine[];
    for (const { word, kata } of parsedContent) {
      if (kata.length !== 1) {
        throw new Error(
          `Invalid dataset: ${datasetPath}, word: ${word}, kata: ${kata}`,
        );
      }
      dataset.set(word, kata[0]);
    }
  }

  const outputContent = Array.from(dataset.entries())
    .map(([word, kata]) => JSON.stringify({ word, kata: [kata] }))
    .join("\n");
  await fs.writeFile(outputPath, outputContent, "utf-8");
  console.log(`Merged ${datasets.length} datasets into ${outputPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
