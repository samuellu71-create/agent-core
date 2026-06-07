/**
 * Retrieval statistics collector.
 *
 * Thread-safe singleton that accumulates per-query metrics from the
 * memory search pipeline. Mirrors OpenViking's RetrievalStatsCollector
 * (retrieval_stats.py:78-151) — tracks query counts, result counts,
 * score distributions, latencies, and rerank usage.
 *
 * Usage:
 *   const collector = getStatsCollector();
 *   collector.recordQuery({ resultCount: 3, scores: [0.82, 0.71], latencyMs: 42 });
 *   const stats = collector.snapshot();
 */

export interface QueryRecord {
  resultCount: number;
  scores: number[];
  latencyMs?: number;
  rerankUsed?: boolean;
}

export interface RetrievalStats {
  totalQueries: number;
  totalResults: number;
  zeroResultQueries: number;
  avgResultsPerQuery: number;
  zeroResultRate: number;
  avgScore: number;
  maxScore: number;
  minScore: number;
  rerankUsed: number;
  avgLatencyMs: number;
  maxLatencyMs: number;
}

class RetrievalStatsCollector {
  private totalQueries = 0;
  private totalResults = 0;
  private zeroResultQueries = 0;
  private totalScoreSum = 0;
  private maxScore = 0;
  private minScore = Infinity;
  private rerankUsed = 0;
  private totalLatencyMs = 0;
  private maxLatencyMs = 0;

  recordQuery(record: QueryRecord): void {
    this.totalQueries++;
    this.totalResults += record.resultCount;

    if (record.resultCount === 0) {
      this.zeroResultQueries++;
    }

    for (const s of record.scores) {
      this.totalScoreSum += s;
      if (s > this.maxScore) this.maxScore = s;
      if (s < this.minScore) this.minScore = s;
    }

    if (record.rerankUsed) this.rerankUsed++;

    const latency = record.latencyMs ?? 0;
    this.totalLatencyMs += latency;
    if (latency > this.maxLatencyMs) this.maxLatencyMs = latency;
  }

  snapshot(): RetrievalStats {
    return {
      totalQueries: this.totalQueries,
      totalResults: this.totalResults,
      zeroResultQueries: this.zeroResultQueries,
      avgResultsPerQuery:
        this.totalQueries > 0 ? this.totalResults / this.totalQueries : 0,
      zeroResultRate:
        this.totalQueries > 0 ? this.zeroResultQueries / this.totalQueries : 0,
      avgScore:
        this.totalResults > 0 ? this.totalScoreSum / this.totalResults : 0,
      maxScore: this.totalResults > 0 ? this.maxScore : 0,
      minScore: this.totalResults > 0 ? this.minScore : 0,
      rerankUsed: this.rerankUsed,
      avgLatencyMs:
        this.totalQueries > 0 ? this.totalLatencyMs / this.totalQueries : 0,
      maxLatencyMs: this.maxLatencyMs,
    };
  }

  reset(): void {
    this.totalQueries = 0;
    this.totalResults = 0;
    this.zeroResultQueries = 0;
    this.totalScoreSum = 0;
    this.maxScore = 0;
    this.minScore = Infinity;
    this.rerankUsed = 0;
    this.totalLatencyMs = 0;
    this.maxLatencyMs = 0;
  }
}

let instance: RetrievalStatsCollector | null = null;

export function getStatsCollector(): RetrievalStatsCollector {
  if (!instance) {
    instance = new RetrievalStatsCollector();
  }
  return instance;
}

export function resetStatsCollector(): void {
  instance = null;
}
