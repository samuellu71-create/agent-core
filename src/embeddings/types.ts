/**
 * Embedding provider interface.
 *
 * Implementations:
 *   - HashEmbedder  (built-in, zero-dep, feature-hashing n-grams)
 *   - OpenAIEmbedder (optional, calls OpenAI embedding API)
 *
 * Reference: mem0 Embedder interface (embeddings/base.ts:1-4)
 *   mem0 exposes embed(text) → number[] and embedBatch(texts) → number[][].
 *   We mirror the same contract, adding dimensionality metadata.
 */
export interface EmbeddingProvider {
  readonly name: string;
  readonly dimensions: number;
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
}
