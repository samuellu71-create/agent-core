/**
 * Hybrid memory scorer.
 *
 * Combines five signals for ranking memory search results:
 *   1. Vector similarity (cosine) — semantic relevance
 *   2. BM25 (FTS5 rank) — keyword relevance
 *   3. Recency decay — exponential time decay
 *   4. Scope proximity — closer scopes boost score
 *   5. Hotness — access-frequency × recency (OpenViking parity)
 *
 * SUPERIOR to mem0 scoring (scoring.py:60-139) which only uses:
 *   - Semantic + BM25 + entity boosts (3 signals)
 *   - No recency decay, no scope-aware boosting, no hotness
 *
 * SUPERIOR to OpenViking (memory_lifecycle.py:19-64) which only uses:
 *   - Semantic + hotness (2 signals blended)
 *   - No BM25 as independent signal, no scope hierarchy
 *
 * Reference: mem0 score_and_rank (scoring.py:60-139)
 *   combined = (semantic + bm25 + entity_boost) / max_possible
 *
 * Reference: OpenViking _convert_to_matched_contexts (hierarchical_retriever.py:540-613)
 *   final = (1-α) * semantic + α * hotness
 *
 * Our formula:
 *   combined = w_vec*vec + w_bm25*bm25 + w_recency*recency + w_scope*scope + w_hotness*hotness
 *   normalized to [0, 1] via max_possible divisor
 */

import type { MemoryKind } from "../providers/MemoryProvider.js";
import { hotnessScore } from "./hotness.js";

/** Per-memory candidate with raw signals. */
export interface ScoringCandidate {
  id: string;
  content: string;
  scope: string;
  scope_id: string;
  kind: MemoryKind;
  metadata: Record<string, unknown>;
  facts: string[];
  concepts: string[];
  files_read: string[];
  files_modified: string[];
  created_at: string;
  updated_at: string;

  /** Number of times this memory has been accessed/retrieved. */
  active_count: number;

  /** Cosine similarity to query embedding, [0, 1] for normalized vectors. */
  vec_score: number;
  /** FTS5 BM25 rank (lower = better match; 0 if no FTS hit). */
  bm25_rank: number;
}

export interface ScoringOptions {
  /** Query scope for proximity boosting. */
  queryScope?: string;
  /** Minimum combined score to include a result. */
  threshold?: number;
  /** Maximum results to return. */
  topK?: number;
  /** Include score breakdown in results. */
  explain?: boolean;
}

export interface ScoredResult {
  id: string;
  content: string;
  scope: string;
  kind: MemoryKind;
  score: number;
  metadata: Record<string, unknown>;
  facts: string[];
  concepts: string[];
  files_read: string[];
  files_modified: string[];
  score_details?: ScoreDetails;
}

export interface ScoreDetails {
  vec_score: number;
  bm25_score: number;
  recency_score: number;
  scope_boost: number;
  hotness_score: number;
  raw_combined: number;
  max_possible: number;
  final_score: number;
}

// ── Weights ───────────────────────────────────────────────────────────
const W_VEC = 1.0;
const W_BM25 = 0.8;
const W_RECENCY = 0.3;
const W_SCOPE = 0.4;
const W_HOTNESS = 0.3;

// ── Scope hierarchy (lower index = narrower scope) ────────────────────
const SCOPE_ORDER: Record<string, number> = {
  session: 0,
  task: 1,
  branch: 2,
  executor: 3,
  repo: 4,
  user: 5,
  global_policy: 6,
};

// ── Recency half-life: 7 days in milliseconds ─────────────────────────
const HALF_LIFE_MS = 7 * 24 * 60 * 60 * 1000;

/**
 * Normalize BM25 rank to [0, 1].
 * SQLite FTS5 bm25() returns negative values (more negative = better).
 * We convert to positive and apply sigmoid normalization.
 */
function normalizeBm25(rank: number): number {
  // FTS5 bm25() returns negative values; negate to get positive
  const raw = -rank;
  if (raw <= 0) return 0;
  // Sigmoid with midpoint=5, steepness=0.6 (similar to mem0)
  return 1.0 / (1.0 + Math.exp(-0.6 * (raw - 5)));
}

/**
 * Compute recency score using exponential decay.
 * Returns 1.0 for just-created memories, decaying toward 0 with half-life.
 */
function recencyScore(createdAt: string, now: number): number {
  const created = new Date(createdAt).getTime();
  const age = now - created;
  if (age <= 0) return 1.0;
  return Math.pow(0.5, age / HALF_LIFE_MS);
}

/**
 * Compute scope proximity boost.
 * Returns 1.0 for exact scope match, decreasing with distance.
 * Returns 0.3 baseline when no query scope is specified.
 */
function scopeBoost(memoryScope: string, queryScope?: string): number {
  if (!queryScope) return 0.3;
  if (memoryScope === queryScope) return 1.0;

  const memIdx = SCOPE_ORDER[memoryScope] ?? 6;
  const queryIdx = SCOPE_ORDER[queryScope] ?? 6;
  const distance = Math.abs(memIdx - queryIdx);

  // Decay based on distance in scope hierarchy
  return Math.max(0.1, 1.0 - distance * 0.15);
}

/**
 * Score and rank candidates using hybrid signals.
 */
export function scoreAndRank(
  candidates: ScoringCandidate[],
  options: ScoringOptions = {},
): ScoredResult[] {
  const { queryScope, threshold = 0, topK = 10, explain = false } = options;
  const now = Date.now();

  const hasVec = candidates.some((c) => c.vec_score > 0);
  const hasBm25 = candidates.some((c) => c.bm25_rank !== 0);

  const hasHotness = candidates.some((c) => c.active_count > 0);

  let maxPossible = 0;
  if (hasVec) maxPossible += W_VEC;
  if (hasBm25) maxPossible += W_BM25;
  maxPossible += W_RECENCY;
  maxPossible += W_SCOPE;
  if (hasHotness) maxPossible += W_HOTNESS;

  if (maxPossible === 0) maxPossible = 1;

  const scored: ScoredResult[] = [];

  for (const c of candidates) {
    const vecScore = c.vec_score;
    const bm25Score = normalizeBm25(c.bm25_rank);
    const recency = recencyScore(c.created_at, now);
    const scope = scopeBoost(c.scope, queryScope);
    const hotness = hotnessScore(c.active_count, c.updated_at);

    const rawCombined =
      W_VEC * vecScore +
      W_BM25 * bm25Score +
      W_RECENCY * recency +
      W_SCOPE * scope +
      W_HOTNESS * hotness;
    const finalScore = Math.min(rawCombined / maxPossible, 1.0);

    if (finalScore < threshold) continue;

    const result: ScoredResult = {
      id: c.id,
      content: c.content,
      scope: c.scope,
      kind: c.kind,
      score: finalScore,
      metadata: c.metadata,
      facts: c.facts,
      concepts: c.concepts,
      files_read: c.files_read,
      files_modified: c.files_modified,
    };

    if (explain) {
      result.score_details = {
        vec_score: vecScore,
        bm25_score: bm25Score,
        recency_score: recency,
        scope_boost: scope,
        hotness_score: hotness,
        raw_combined: rawCombined,
        max_possible: maxPossible,
        final_score: finalScore,
      };
    }

    scored.push(result);
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}
