/**
 * Embedding provider factory.
 *
 * Returns the configured embedder based on environment variables:
 *   - EMBEDDING_PROVIDER=hash (default): zero-dep hash embedder
 *   - EMBEDDING_PROVIDER=openai: OpenAI API (requires OPENAI_API_KEY)
 *
 * The embedder is cached as a singleton per configuration.
 */

import type { EmbeddingProvider } from "./types.js";
import { HashEmbedder } from "./hashEmbedder.js";

let cached: EmbeddingProvider | null = null;

export function getEmbedder(): EmbeddingProvider {
  if (cached) return cached;

  const provider = process.env.EMBEDDING_PROVIDER ?? "hash";
  const dims = parseInt(process.env.EMBEDDING_DIMENSIONS ?? "256", 10);

  switch (provider) {
    case "hash":
      cached = new HashEmbedder(dims);
      break;
    default:
      cached = new HashEmbedder(dims);
      break;
  }

  return cached;
}

/** Reset the cached embedder (for tests). */
export function resetEmbedder(): void {
  cached = null;
}
