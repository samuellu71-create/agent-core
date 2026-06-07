/**
 * Zero-dependency feature-hashing embedder.
 *
 * Uses the hashing trick (feature hashing) on character n-grams and word
 * unigrams/bigrams to produce a fixed-dimension vector for any text.
 *
 * Advantages over mem0's OpenAIEmbedder:
 *   - Zero external dependencies or API calls
 *   - Deterministic — same input always produces same vector
 *   - Sub-millisecond latency (vs 100-500ms for API calls)
 *   - Works offline / in tests without credentials
 *
 * Trade-offs:
 *   - No true semantic understanding (synonym/paraphrase detection)
 *   - Quality bounded by lexical overlap
 *   - For production semantic search, use OpenAIEmbedder or similar
 *
 * Reference: scikit-learn FeatureHasher / Vowpal Wabbit hashing trick.
 */

import type { EmbeddingProvider } from "./types.js";

const DEFAULT_DIMENSIONS = 256;

/**
 * FNV-1a 32-bit hash — fast, well-distributed, deterministic.
 */
function fnv1a(str: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = (hash * 0x01000193) >>> 0;
  }
  return hash;
}

/**
 * Extract features from text: word unigrams, bigrams, and character 3-grams.
 */
function extractFeatures(text: string): string[] {
  const lower = text.toLowerCase().replace(/[^a-z0-9\s]/g, " ");
  const words = lower.split(/\s+/).filter((w) => w.length > 0);
  const features: string[] = [];

  // Word unigrams
  for (const w of words) {
    features.push(`w:${w}`);
  }

  // Word bigrams
  for (let i = 0; i < words.length - 1; i++) {
    features.push(`b:${words[i]}_${words[i + 1]}`);
  }

  // Character 3-grams (captures subword structure)
  for (const w of words) {
    const padded = `^${w}$`;
    for (let i = 0; i < padded.length - 2; i++) {
      features.push(`c:${padded.slice(i, i + 3)}`);
    }
  }

  return features;
}

/**
 * L2-normalize a vector in-place.
 */
function l2Normalize(vec: number[]): number[] {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) {
    norm += vec[i] * vec[i];
  }
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < vec.length; i++) {
      vec[i] /= norm;
    }
  }
  return vec;
}

export class HashEmbedder implements EmbeddingProvider {
  readonly name = "hash-embedder";
  readonly dimensions: number;

  constructor(dimensions: number = DEFAULT_DIMENSIONS) {
    this.dimensions = dimensions;
  }

  async embed(text: string): Promise<number[]> {
    return this.hashEmbed(text);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return texts.map((t) => this.hashEmbed(t));
  }

  private hashEmbed(text: string): number[] {
    const vec = new Array<number>(this.dimensions).fill(0);
    const features = extractFeatures(text);

    for (const feature of features) {
      const hash = fnv1a(feature);
      const index = hash % this.dimensions;
      // Sign bit determines +1 or -1 contribution (reduces collision bias)
      const sign = hash & 0x80000000 ? -1 : 1;
      vec[index] += sign;
    }

    return l2Normalize(vec);
  }
}
