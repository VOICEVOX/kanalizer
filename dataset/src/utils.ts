import { Mutex } from "@core/asyncutil";

export const bisectMax = async (
  min: number,
  max: number,
  predicate: (mid: number) => boolean | Promise<boolean>,
) => {
  let currentMin = min;
  let currentMax = max;
  while (currentMin < currentMax) {
    const mid = Math.floor((currentMin + currentMax) / 2);

    if (await predicate(mid)) {
      currentMin = mid + 1;
    } else {
      currentMax = mid;
    }
  }

  return currentMin;
};

// 半角カタカナ、ひらがなを全角カタカナに変換し、長音っぽい文字を長音に変換する
const normalizeKana = (text: string) => {
  return text
    .replace(/[\uFF65-\uFF9F]/g, (s) =>
      String.fromCharCode(s.charCodeAt(0) - 0x60),
    )
    .replace(/[\u3041-\u3096]/g, (s) =>
      String.fromCharCode(s.charCodeAt(0) + 0x60),
    )
    .replace(/[ｰ―－ー]/g, "ー");
};

const normalizeOrNull = (pronunciation: string) => {
  const normalized = normalizeKana(pronunciation.trim());
  if (!normalized.match(/^[\p{Script=Katakana}ー]+$/u)) {
    return null;
  }
  return normalized;
};

export const filterPronunciations = (
  pronunciations: Record<string, string>,
) => {
  const filtered: Record<string, string> = {};
  for (const [word, pronunciation] of Object.entries(pronunciations)) {
    // 読みを正規化し、それでも無効な読み方だったらスキップ
    const normalized = normalizeOrNull(pronunciation);
    if (normalized == null) {
      console.error(`Invalid pronunciation: ${word} -> ${pronunciation}`);
      continue;
    }

    // アルファベット読みならスキップ
    if (isAlphabetPronunciation(word, normalized)) {
      console.error(`Pronunciation is too simple: ${word} -> ${pronunciation}`);
      continue;
    }

    filtered[word] = normalized;
  }
  return filtered;
};

export type SuspiciousWordReason =
  | "tooLong"
  | "containsNonAlphabet"
  | "hasImmediateRepeatedChunk"
  | "hasLongRepeatedChar";

export const getSuspiciousWordReasons = (
  word: string,
  options: { maxWordLength?: number } = {},
): SuspiciousWordReason[] => {
  const reasons: SuspiciousWordReason[] = [];
  const maxWordLength = options.maxWordLength ?? 40;

  if (word.length > maxWordLength) {
    reasons.push("tooLong");
    return reasons;
  }

  if (!/^[a-z]+$/i.test(word)) {
    reasons.push("containsNonAlphabet");
  }

  if (/(.{3,})\1{2,}/i.test(word)) {
    reasons.push("hasImmediateRepeatedChunk");
  }

  if (/(.)\1{4,}/i.test(word)) {
    reasons.push("hasLongRepeatedChar");
  }

  return reasons;
};

export const isSuspiciousWord = (
  word: string,
  options: { maxWordLength?: number } = {},
) => getSuspiciousWordReasons(word, options).length > 0;

export const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

export class Throttle {
  private mutex: Mutex;
  private throttleMs: number;

  constructor(throttleMs: number) {
    this.mutex = new Mutex();
    this.throttleMs = throttleMs;
  }

  async throttle() {
    // TODO: Node.jsがusing文に対応したらusingに置き換える
    const lock = await this.mutex.acquire();
    await sleep(this.throttleMs);
    lock[Symbol.dispose]();
  }
}

export class ExhaustiveError extends Error {
  constructor(value: never, message?: string) {
    super(message ?? `Unexpected value: ${JSON.stringify(value, null, 2)}`);
  }
}

/** 単語をアルファベット読み（エービーシー）に変換する */
export const wordToAlphabetPronunciation = (word: string): string => {
  return word
    .split("")
    .map((c) => alphabetMap[c.toLowerCase() as keyof typeof alphabetMap][0])
    .join("");
};

/** 単語がアルファベット読み（エービーシー）かどうかを判定する */
export const isAlphabetPronunciation = (
  word: string,
  pronunciation: string,
) => {
  const pattern = `^(?:${word
    .split("")
    .map((c) =>
      alphabetMap[c.toLowerCase() as keyof typeof alphabetMap].join("|"),
    )
    .join(")(?:")})$`;
  const regex = new RegExp(pattern);
  return regex.test(pronunciation);
};

const alphabetMap = {
  a: ["エー", "エィ", "エイ", "エエ"],
  b: ["ビー", "ビィ", "ビイ"],
  c: ["シー", "スィー", "シイ"],
  d: ["ディー", "デー", "デイ"],
  e: ["イー", "イィ", "イイ"],
  f: ["エフ"],
  g: ["ジー", "ジィ", "ジイ"],
  h: ["エイチ", "エッチ"],
  i: ["アイ"],
  j: ["ジェー", "ジェイ"],
  k: ["ケー", "ケイ"],
  l: ["エル"],
  m: ["エム"],
  n: ["エヌ", "エン"],
  o: ["オー", "オウ", "オオ"],
  p: ["ピー", "ピイ"],
  q: ["キュー", "キウ", "キュウ"],
  r: ["アール"],
  s: ["エス"],
  t: ["ティー", "ティイ"],
  u: ["ユー", "ユウ"],
  v: ["ブイ", "ヴイ", "ヴィ", "ヴィー", "ヴィイ"],
  w: ["ダブリュー", "ダブル", "ワブリュー", "ダブリュ", "ダブリュウ"],
  x: ["エックス", "エクス"],
  y: ["ワイ"],
  z: ["ゼット", "ズィー", "ゼッド", "ジー", "ズィイ", "ジイ"],
};
