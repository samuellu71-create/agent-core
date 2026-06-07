import { FastifyInstance } from "fastify";
import {
  MemoryWriteInput,
  MemorySearchInput,
  MemoryExtractInput,
  MemoryPromoteInput,
  MemoryPatchInput,
  MemoryExtractionResult,
} from "../schemas/memory.js";
import {
  isDeepSeekConfigured,
  DeepSeekClient,
  isGeminiConfigured,
  GeminiClient,
  llmJson,
} from "../llm/index.js";
import type { LLMClient } from "../llm/index.js";
import { getStatsCollector } from "./retrievalStats.js";
import { getProvider } from "../providers/registry.js";

export async function memoryRoutes(app: FastifyInstance): Promise<void> {
  app.post("/memory/write", async (req, reply) => {
    const input = MemoryWriteInput.parse(req.body);
    const memory = await getProvider("memory").write(input);
    reply.code(201);
    return { id: memory.id, scope: memory.scope, scope_id: memory.scope_id };
  });

  app.post("/memory/search", async (req) => {
    const input = MemorySearchInput.parse(req.body);
    const results = await getProvider("memory").search({
      query: input.query,
      scope: input.scope,
      scope_id: input.scope_id,
      filters: input.filters,
      top_k: input.limit,
    });
    return {
      results,
    };
  });

  app.post("/memory/extract", async (req) => {
    const input = MemoryExtractInput.parse(req.body);

    // Try LLM extraction with multi-provider fallback (Gemini → DeepSeek)
    const llmResult = await extractWithLLM(input.text);
    if (llmResult) {
      return { session_id: input.session_id, candidates: llmResult, method: "llm" };
    }

    // Fallback to keyword heuristic extraction
    const candidates = extractCandidates(input.text);
    return {
      session_id: input.session_id,
      candidates,
      method: "keyword",
    };
  });

  app.get("/memory/stats", async () => {
    return getStatsCollector().snapshot();
  });

  app.post("/memory/promote", async (req) => {
    const input = MemoryPromoteInput.parse(req.body);
    const source = await getProvider("memory").get(input.memory_id);
    if (!source) {
      return { error: "memory not found" };
    }
    const promoted = await getProvider("memory").write({
      scope: input.target_scope,
      scope_id: input.target_scope_id,
      content: source.content,
      metadata: source.metadata,
      kind: source.kind,
      facts: source.facts,
      concepts: source.concepts,
      files_read: source.files_read,
      files_modified: source.files_modified,
    });
    return { id: promoted.id, promoted_from: input.memory_id, scope: promoted.scope };
  });

  app.get("/memory/:memory_id", async (req) => {
    const { memory_id } = req.params as { memory_id: string };
    const memory = await getProvider("memory").get(memory_id);
    if (!memory) {
      return { error: "not found" };
    }
    return memory;
  });

  app.patch("/memory/:memory_id", async (req) => {
    const { memory_id } = req.params as { memory_id: string };
    const input = MemoryPatchInput.parse(req.body);
    const current = await getProvider("memory").get(memory_id);
    if (!current) return { error: "not found" };
    const updated = await getProvider("memory").update(memory_id, {
      content: input.content ?? current.content,
      metadata: input.metadata ?? current.metadata,
      kind: input.kind ?? current.kind,
      facts: input.facts ?? current.facts,
      concepts: input.concepts ?? current.concepts,
      files_read: input.files_read ?? current.files_read,
      files_modified: input.files_modified ?? current.files_modified,
    });
    return updated ?? { error: "not found" };
  });

  app.delete("/memory/:memory_id", async (req, reply) => {
    const { memory_id } = req.params as { memory_id: string };
    await getProvider("memory").delete(memory_id);
    reply.code(204);
    return;
  });
}

const EXTRACTION_SYSTEM_PROMPT = `You are a memory extraction system for a software development agent platform.
Given text from a coding session, extract durable facts, conventions, patterns, and lessons worth remembering.

Return valid JSON matching this schema:
{
  "candidates": [
    {
      "content": "The extracted fact or convention",
      "confidence": 0.0-1.0 float indicating relevance/importance,
      "scope": optional, one of "user", "repo", "branch", "task", "session", "executor", "global_policy"
    }
  ]
}

Guidelines:
- Extract facts that would be useful in future sessions (conventions, patterns, gotchas, preferences)
- Confidence: 0.9+ for explicit rules ("always", "never", "must"), 0.6-0.8 for conventions, 0.3-0.5 for observations
- Scope: "repo" for repo-specific facts, "user" for user preferences, "global_policy" for universal rules
- Return at most 10 candidates, sorted by confidence descending
- Skip trivial or ephemeral facts`;

/**
 * Resolve the best available LLM client.
 *
 * Priority: Gemini → DeepSeek → null (fallback to keyword)
 *
 * Reference: mem0 uses GPT (OpenAI) for extraction.
 *            cognee uses Instructor (OpenAI/Gemini/Anthropic) for entity extraction.
 *            We support multi-provider with automatic fallback.
 */
function resolveExtractionClient(): LLMClient | null {
  if (isGeminiConfigured()) return new GeminiClient();
  if (isDeepSeekConfigured()) return new DeepSeekClient();
  return null;
}

async function extractWithLLM(
  text: string,
): Promise<Array<{ content: string; confidence: number; scope?: string }> | null> {
  const client = resolveExtractionClient();
  if (!client) return null;

  const result = await llmJson(client, MemoryExtractionResult, [
    { role: "system", content: EXTRACTION_SYSTEM_PROMPT },
    { role: "user", content: text },
  ]);

  if (result.success) return result.data.candidates;
  return null;
}

function extractCandidates(text: string): { content: string; confidence: number }[] {
  const sentences = text
    .split(/[.!?\n]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 10);
  const keywords = [
    "always",
    "never",
    "must",
    "should",
    "important",
    "remember",
    "note",
    "convention",
    "pattern",
    "prefer",
    "avoid",
    "requires",
    "depends",
  ];
  return sentences
    .map((s) => {
      const lower = s.toLowerCase();
      const matchCount = keywords.filter((k) => lower.includes(k)).length;
      const confidence = Math.min(0.95, 0.3 + matchCount * 0.15);
      return { content: s, confidence };
    })
    .filter((c) => c.confidence >= 0.4)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 10);
}
