import { describe, it, expect, beforeAll, beforeEach, afterAll } from "vitest";
import { closeDb, getDb } from "../src/db.js";
import { MockMemoryProvider } from "../src/providers/mocks/MockMemoryProvider.js";
import { getStatsCollector, resetStatsCollector } from "../src/memory/retrievalStats.js";
import { scoreAndRank, type ScoringCandidate } from "../src/memory/hybridScorer.js";
import { hotnessScore } from "../src/memory/hotness.js";
import { HashEmbedder } from "../src/embeddings/hashEmbedder.js";
import { cosineSimilarity } from "../src/embeddings/similarity.js";
import { checkDuplicate } from "../src/memory/deduplicator.js";
import { NoopReranker, ScoreReranker } from "../src/memory/reranker.js";

beforeAll(() => {
  process.env.AGENT_CORE_DB = ":memory:";
});

afterAll(() => {
  closeDb();
});

describe("Hybrid Search — 5-signal scoring", () => {
  let mem: MockMemoryProvider;

  beforeEach(() => {
    closeDb();
    mem = new MockMemoryProvider();
    resetStatsCollector();
  });

  it("ranks semantically similar content higher than keyword-only", async () => {
    // Write memories with different content
    await mem.write({
      scope: "repo",
      scope_id: "test-repo",
      content: "TypeScript project uses ESLint for code quality enforcement",
    });
    await mem.write({
      scope: "repo",
      scope_id: "test-repo",
      content: "Python backend uses Flask for the API layer",
    });
    await mem.write({
      scope: "repo",
      scope_id: "test-repo",
      content: "JavaScript linting configuration and code style rules",
    });

    // Search for something semantically similar to the first and third
    const results = await mem.search({
      query: "linting and code quality tools",
      scope: "repo",
      scope_id: "test-repo",
    });

    expect(results.length).toBeGreaterThan(0);
    // The ESLint and JavaScript linting memories should score higher than Flask
    const flaskResult = results.find((r) => r.content.includes("Flask"));
    const eslintResult = results.find((r) => r.content.includes("ESLint"));
    if (flaskResult && eslintResult) {
      expect(eslintResult.score).toBeGreaterThan(flaskResult.score);
    }
  });

  it("returns results with scores in [0, 1]", async () => {
    await mem.write({
      scope: "repo",
      scope_id: "test-repo",
      content: "Always run tests before committing changes",
    });

    const results = await mem.search({
      query: "running tests before commit",
      scope: "repo",
      scope_id: "test-repo",
    });

    expect(results.length).toBeGreaterThan(0);
    for (const r of results) {
      expect(r.score).toBeGreaterThanOrEqual(0);
      expect(r.score).toBeLessThanOrEqual(1);
    }
  });

  it("boosts results in the same scope as the query via scoring", async () => {
    // Directly test the scorer since the DB search layer filters by scope
    const now = new Date().toISOString();
    const candidates: ScoringCandidate[] = [
      {
        id: "1", content: "migration seed", scope: "session", scope_id: "s1",
        kind: "observation", metadata: {}, facts: [], concepts: [],
        files_read: [], files_modified: [], created_at: now, updated_at: now,
        active_count: 0, vec_score: 0.8, bm25_rank: -5,
      },
      {
        id: "2", content: "migration seed", scope: "repo", scope_id: "r1",
        kind: "observation", metadata: {}, facts: [], concepts: [],
        files_read: [], files_modified: [], created_at: now, updated_at: now,
        active_count: 0, vec_score: 0.8, bm25_rank: -5,
      },
    ];

    const results = scoreAndRank(candidates, { queryScope: "session", explain: true });
    expect(results.length).toBe(2);
    // Session-scoped should rank higher due to exact scope match
    expect(results[0].scope).toBe("session");
    expect(results[0].score_details!.scope_boost).toBe(1.0);
    expect(results[1].score_details!.scope_boost).toBeLessThan(1.0);
  });

  it("applies recency decay — newer memories rank higher (all else equal)", async () => {
    const db = getDb();
    // Write two memories with distinct content in different scope_ids to avoid dedup
    await mem.write({
      scope: "repo",
      scope_id: "repo-a",
      content: "Use vitest for all unit tests in the project codebase",
    });

    // Backdating the first memory to 30 days ago
    db.prepare(`UPDATE memories SET created_at = datetime('now', '-30 days') WHERE scope_id = 'repo-a'`).run();

    await mem.write({
      scope: "repo",
      scope_id: "repo-b",
      content: "Vitest is the testing framework for all unit test suites",
    });

    const results = await mem.search({
      query: "vitest unit testing",
    });

    expect(results.length).toBe(2);
    // The newer one should rank higher due to recency
    // MemorySearchResult doesn't include scope_id, but we can check content
    expect(results[0].content).toContain("Vitest is the testing");
  });

  it("filters by metadata", async () => {
    await mem.write({
      scope: "repo",
      scope_id: "r1",
      content: "Use prettier for formatting",
      metadata: { language: "typescript" },
    });
    await mem.write({
      scope: "repo",
      scope_id: "r1",
      content: "Use black for formatting",
      metadata: { language: "python" },
    });

    const results = await mem.search({
      query: "formatting",
      scope: "repo",
      scope_id: "r1",
      filters: { language: "python" },
    });

    expect(results.length).toBe(1);
    expect(results[0].content).toContain("black");
  });

  it("respects top_k limit", async () => {
    for (let i = 0; i < 5; i++) {
      await mem.write({
        scope: "repo",
        scope_id: `r-${i}`,
        content: `Memory item number ${i} about testing`,
      });
    }

    const results = await mem.search({
      query: "testing",
      top_k: 2,
    });

    expect(results.length).toBe(2);
  });

  it("respects threshold filter", async () => {
    await mem.write({
      scope: "repo",
      scope_id: "r1",
      content: "TypeScript strict mode is mandatory",
    });

    const results = await mem.search({
      query: "completely unrelated ocean biology topic",
      threshold: 0.99, // Very high threshold — should filter most results
    });

    // Most results should be filtered out due to low relevance
    expect(results.length).toBeLessThanOrEqual(1);
  });
});

describe("Hybrid Scorer — scoreAndRank", () => {
  it("combines all 5 signals correctly", () => {
    const now = new Date().toISOString();
    const candidates: ScoringCandidate[] = [
      {
        id: "1",
        content: "test memory",
        scope: "repo",
        scope_id: "r1",
        kind: "observation",
        metadata: {},
        facts: [],
        concepts: [],
        files_read: [],
        files_modified: [],
        created_at: now,
        updated_at: now,
        active_count: 5,
        vec_score: 0.9,
        bm25_rank: -10,
      },
    ];

    const results = scoreAndRank(candidates, { queryScope: "repo", explain: true });
    expect(results.length).toBe(1);
    expect(results[0].score).toBeGreaterThan(0);
    expect(results[0].score).toBeLessThanOrEqual(1);
    expect(results[0].score_details).toBeDefined();
    expect(results[0].score_details!.vec_score).toBe(0.9);
    expect(results[0].score_details!.bm25_score).toBeGreaterThan(0);
    expect(results[0].score_details!.recency_score).toBeGreaterThan(0.9);
    expect(results[0].score_details!.scope_boost).toBe(1.0); // Exact scope match
    expect(results[0].score_details!.hotness_score).toBeGreaterThan(0);
  });

  it("adapts max_possible based on available signals", () => {
    const now = new Date().toISOString();
    // No vec or bm25 signals → only recency + scope contribute to max
    const candidates: ScoringCandidate[] = [
      {
        id: "1",
        content: "test",
        scope: "repo",
        scope_id: "r1",
        kind: "manual",
        metadata: {},
        facts: [],
        concepts: [],
        files_read: [],
        files_modified: [],
        created_at: now,
        updated_at: now,
        active_count: 0,
        vec_score: 0,
        bm25_rank: 0,
      },
    ];

    const results = scoreAndRank(candidates, { queryScope: "repo", explain: true });
    expect(results.length).toBe(1);
    // With only recency (0.3) + scope (0.4) contributing, max_possible = 0.7
    expect(results[0].score_details!.max_possible).toBe(0.7);
  });
});

describe("Hotness scoring — OpenViking parity", () => {
  it("returns 0 for null updatedAt", () => {
    expect(hotnessScore(10, null)).toBe(0);
  });

  it("returns higher score for higher active_count", () => {
    const now = new Date();
    const s1 = hotnessScore(1, now.toISOString(), now);
    const s10 = hotnessScore(10, now.toISOString(), now);
    const s100 = hotnessScore(100, now.toISOString(), now);

    expect(s10).toBeGreaterThan(s1);
    expect(s100).toBeGreaterThan(s10);
  });

  it("decays over time with 7-day half-life", () => {
    const now = new Date();
    const fresh = hotnessScore(5, now.toISOString(), now);
    const sevenDaysAgo = new Date(now.getTime() - 7 * 86_400_000);
    const old = hotnessScore(5, sevenDaysAgo.toISOString(), now);

    // After 7 days, recency should be ~50% → hotness should be roughly half
    expect(old).toBeLessThan(fresh);
    expect(old / fresh).toBeCloseTo(0.5, 1);
  });

  it("returns value in [0, 1]", () => {
    const now = new Date();
    for (const count of [0, 1, 10, 100, 10000]) {
      const s = hotnessScore(count, now.toISOString(), now);
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });
});

describe("HashEmbedder", () => {
  it("produces deterministic embeddings", async () => {
    const embedder = new HashEmbedder();
    const a = await embedder.embed("hello world");
    const b = await embedder.embed("hello world");
    expect(a).toEqual(b);
  });

  it("produces L2-normalized vectors (unit length)", async () => {
    const embedder = new HashEmbedder();
    const vec = await embedder.embed("test content for embedding");
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    expect(norm).toBeCloseTo(1.0, 4);
  });

  it("similar text has higher cosine similarity", async () => {
    const embedder = new HashEmbedder();
    const a = await embedder.embed("TypeScript linting with ESLint");
    const b = await embedder.embed("JavaScript linting with ESLint rules");
    const c = await embedder.embed("Ocean biology and marine ecosystems");

    const simAB = cosineSimilarity(a, b);
    const simAC = cosineSimilarity(a, c);

    expect(simAB).toBeGreaterThan(simAC);
  });

  it("embedBatch produces consistent results", async () => {
    const embedder = new HashEmbedder();
    const texts = ["hello", "world"];
    const batch = await embedder.embedBatch(texts);
    const individual = [await embedder.embed("hello"), await embedder.embed("world")];

    expect(batch[0]).toEqual(individual[0]);
    expect(batch[1]).toEqual(individual[1]);
  });
});

describe("Retrieval stats collector — OpenViking parity", () => {
  beforeEach(() => {
    resetStatsCollector();
  });

  it("tracks query metrics", () => {
    const c = getStatsCollector();
    c.recordQuery({ resultCount: 3, scores: [0.9, 0.7, 0.5], latencyMs: 42 });
    c.recordQuery({ resultCount: 0, scores: [], latencyMs: 10 });

    const stats = c.snapshot();
    expect(stats.totalQueries).toBe(2);
    expect(stats.totalResults).toBe(3);
    expect(stats.zeroResultQueries).toBe(1);
    expect(stats.zeroResultRate).toBe(0.5);
    expect(stats.avgResultsPerQuery).toBe(1.5);
    expect(stats.maxScore).toBe(0.9);
    expect(stats.minScore).toBe(0.5);
    expect(stats.avgLatencyMs).toBe(26);
    expect(stats.maxLatencyMs).toBe(42);
  });

  it("integrates with search pipeline", async () => {
    closeDb();
    const mem = new MockMemoryProvider();
    await mem.write({ scope: "repo", scope_id: "r1", content: "test memory content" });
    await mem.search({ query: "test memory" });

    const stats = getStatsCollector().snapshot();
    expect(stats.totalQueries).toBe(1);
    expect(stats.totalResults).toBeGreaterThan(0);
  });
});

describe("Memory deduplication — mem0 parity", () => {
  let mem: MockMemoryProvider;

  beforeEach(() => {
    closeDb();
    mem = new MockMemoryProvider();
  });

  it("detects and merges near-duplicate memories", async () => {
    const first = await mem.write({
      scope: "repo",
      scope_id: "r1",
      content: "Always run linter before committing code changes",
      facts: ["linter-required"],
    });

    // Write nearly identical content — should merge
    const second = await mem.write({
      scope: "repo",
      scope_id: "r1",
      content: "Always run linter before committing code changes",
      facts: ["pre-commit-hook"],
    });

    // Should reuse the existing record
    expect(second.id).toBe(first.id);

    // Facts should be merged
    const record = await mem.get(first.id);
    expect(record).not.toBeNull();
    expect(record!.facts).toContain("linter-required");
    expect(record!.facts).toContain("pre-commit-hook");
  });

  it("does NOT merge distinct memories", async () => {
    const first = await mem.write({
      scope: "repo",
      scope_id: "r1",
      content: "TypeScript strict mode is mandatory",
    });
    const second = await mem.write({
      scope: "repo",
      scope_id: "r1",
      content: "Python uses black for code formatting",
    });

    expect(second.id).not.toBe(first.id);
  });

  it("checkDuplicate returns similarity score for near-duplicates", async () => {
    await mem.write({
      scope: "repo",
      scope_id: "r1",
      content: "Database migrations must be reviewed before merging",
    });

    const result = await checkDuplicate(
      "Database migrations must be reviewed before merging",
      "repo",
      "r1",
    );

    expect(result.isDuplicate).toBe(true);
    expect(result.similarity).toBeGreaterThan(0.9);
  });
});

describe("Reranker interface — OpenViking parity", () => {
  it("NoopReranker passes through unchanged", async () => {
    const reranker = new NoopReranker();
    const candidates = [
      { id: "1", content: "first", score: 0.5 },
      { id: "2", content: "second", score: 0.9 },
    ];
    const results = await reranker.rerank("query", candidates);
    expect(results[0].id).toBe("1");
    expect(results[1].id).toBe("2");
  });

  it("ScoreReranker sorts by score descending", async () => {
    const reranker = new ScoreReranker();
    const candidates = [
      { id: "1", content: "first", score: 0.3 },
      { id: "2", content: "second", score: 0.9 },
      { id: "3", content: "third", score: 0.6 },
    ];
    const results = await reranker.rerank("query", candidates);
    expect(results[0].id).toBe("2");
    expect(results[1].id).toBe("3");
    expect(results[2].id).toBe("1");
  });
});

describe("Active count tracking", () => {
  let mem: MockMemoryProvider;

  beforeEach(() => {
    closeDb();
    mem = new MockMemoryProvider();
  });

  it("increments active_count on search hits", async () => {
    await mem.write({
      scope: "repo",
      scope_id: "r1",
      content: "Use vitest for testing TypeScript projects",
    });

    // Search twice — should bump active_count
    await mem.search({ query: "vitest testing", scope: "repo", scope_id: "r1" });
    await mem.search({ query: "vitest testing", scope: "repo", scope_id: "r1" });

    const db = getDb();
    const row = db.prepare("SELECT active_count FROM memories").get() as { active_count: number };
    expect(row.active_count).toBeGreaterThanOrEqual(2);
  });
});
