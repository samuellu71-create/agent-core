/**
 * Reranker interface and implementations.
 *
 * Mirrors OpenViking's rerank architecture (rerank/__init__.py:1-25)
 * which supports VikingDB, Cohere, LiteLLM, and OpenAI-compatible
 * rerank providers.
 *
 * Our design:
 *   - RerankerProvider interface (pluggable)
 *   - NoopReranker (default, passthrough)
 *   - ScoreReranker (re-sort by a secondary signal, built-in)
 *   - Future: CohereReranker, OpenAIReranker
 *
 * Reference:
 *   OpenViking hierarchical_retriever.py:261-291 (_rerank_scores)
 *     Falls back to vector scores on failure.
 *   mem0 main.py:1234-1239 (optional reranker in search pipeline)
 */

export interface RerankCandidate {
  id: string;
  content: string;
  score: number;
}

export interface RerankResult {
  id: string;
  score: number;
}

/**
 * Reranker provider interface.
 * Implementations re-score candidates given a query.
 */
export interface RerankerProvider {
  readonly name: string;
  rerank(query: string, candidates: RerankCandidate[]): Promise<RerankResult[]>;
}

/**
 * No-op reranker — returns candidates in their original order.
 * Used as default when no external reranker is configured.
 */
export class NoopReranker implements RerankerProvider {
  readonly name = "noop";

  async rerank(_query: string, candidates: RerankCandidate[]): Promise<RerankResult[]> {
    return candidates.map((c) => ({ id: c.id, score: c.score }));
  }
}

/**
 * Score-based reranker — re-sorts candidates by their existing score descending.
 * Useful as a normalization pass that ensures consistent ordering.
 */
export class ScoreReranker implements RerankerProvider {
  readonly name = "score-reranker";

  async rerank(_query: string, candidates: RerankCandidate[]): Promise<RerankResult[]> {
    return candidates
      .map((c) => ({ id: c.id, score: c.score }))
      .sort((a, b) => b.score - a.score);
  }
}

let reranker: RerankerProvider | null = null;

export function getReranker(): RerankerProvider {
  if (!reranker) {
    reranker = new NoopReranker();
  }
  return reranker;
}

export function setReranker(provider: RerankerProvider): void {
  reranker = provider;
}

export function resetReranker(): void {
  reranker = null;
}
