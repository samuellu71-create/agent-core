/**
 * Memory deduplication via embedding similarity.
 *
 * Detects near-duplicate memories on write and merges them.
 *
 * Reference: mem0 main.py (dedup via LLM conflict resolution)
 *   mem0 uses GPT to detect conflicting memories and merge them.
 *   We use embedding cosine similarity (faster, cheaper, deterministic)
 *   with content-based merge when duplicates are detected.
 *
 * Reference: cognee (graph-based dedup via entity resolution)
 *   cognee deduplicates at the entity level — same entity = same node.
 *   We deduplicate at the memory level — similar content = merge.
 *
 * Algorithm:
 *   1. On write, compute embedding for new content
 *   2. Scan existing memories in same scope for high-similarity matches
 *   3. If cosine similarity > threshold (default 0.92), merge instead of insert
 *   4. Merge strategy: keep newer content, union metadata/facts/concepts
 */

import { getDb } from "../db.js";
import { getEmbedder, cosineSimilarity } from "../embeddings/index.js";

const DEFAULT_DEDUP_THRESHOLD = 0.92;

export interface DedupResult {
  isDuplicate: boolean;
  existingId?: string;
  similarity?: number;
}

/**
 * Check if a new memory is a near-duplicate of an existing one.
 *
 * @param content New memory content
 * @param scope   Scope to search within
 * @param scopeId Scope ID to search within
 * @param threshold Cosine similarity threshold (default 0.92)
 */
export async function checkDuplicate(
  content: string,
  scope: string,
  scopeId: string,
  threshold: number = DEFAULT_DEDUP_THRESHOLD,
): Promise<DedupResult> {
  const embedder = getEmbedder();
  const newEmbedding = await embedder.embed(content);

  // Fetch existing memories in the same scope
  const db = getDb();
  const rows = db
    .prepare(
      `SELECT id, embedding FROM memories WHERE scope = ? AND scope_id = ? AND embedding IS NOT NULL`,
    )
    .all(scope, scopeId) as Array<{ id: string; embedding: string }>;

  let bestSimilarity = 0;
  let bestId: string | undefined;

  for (const row of rows) {
    const existingEmbedding = JSON.parse(row.embedding) as number[];
    const sim = cosineSimilarity(newEmbedding, existingEmbedding);
    if (sim > bestSimilarity) {
      bestSimilarity = sim;
      bestId = row.id;
    }
  }

  if (bestSimilarity >= threshold && bestId) {
    return { isDuplicate: true, existingId: bestId, similarity: bestSimilarity };
  }

  return { isDuplicate: false };
}

/**
 * Merge arrays by union (no duplicates).
 */
function mergeArrays(a: string[], b: string[]): string[] {
  return [...new Set([...a, ...b])];
}

/**
 * Merge metadata objects (new values override old).
 */
function mergeMetadata(
  existing: Record<string, unknown>,
  incoming: Record<string, unknown>,
): Record<string, unknown> {
  return { ...existing, ...incoming };
}

/**
 * Merge a new memory into an existing duplicate.
 * Keeps the newer content, unions facts/concepts/files, merges metadata.
 */
export async function mergeIntoExisting(
  existingId: string,
  newContent: string,
  newMetadata: Record<string, unknown>,
  newFacts: string[],
  newConcepts: string[],
  newFilesRead: string[],
  newFilesModified: string[],
): Promise<void> {
  const db = getDb();
  const existing = db.prepare(`SELECT * FROM memories WHERE id = ?`).get(existingId) as {
    content: string;
    metadata: string;
    facts: string;
    concepts: string;
    files_read: string;
    files_modified: string;
  } | null;

  if (!existing) return;

  const existingMeta = JSON.parse(existing.metadata) as Record<string, unknown>;
  const existingFacts = JSON.parse(existing.facts) as string[];
  const existingConcepts = JSON.parse(existing.concepts) as string[];
  const existingFilesRead = JSON.parse(existing.files_read) as string[];
  const existingFilesModified = JSON.parse(existing.files_modified) as string[];

  const mergedMeta = mergeMetadata(existingMeta, newMetadata);
  const mergedFacts = mergeArrays(existingFacts, newFacts);
  const mergedConcepts = mergeArrays(existingConcepts, newConcepts);
  const mergedFilesRead = mergeArrays(existingFilesRead, newFilesRead);
  const mergedFilesModified = mergeArrays(existingFilesModified, newFilesModified);

  // Recompute embedding for the new content
  const embedder = getEmbedder();
  const newEmbedding = await embedder.embed(newContent);

  db.prepare(
    `UPDATE memories SET content = ?, metadata = ?, facts = ?, concepts = ?,
     files_read = ?, files_modified = ?, embedding = ?,
     updated_at = datetime('now') WHERE id = ?`,
  ).run(
    newContent,
    JSON.stringify(mergedMeta),
    JSON.stringify(mergedFacts),
    JSON.stringify(mergedConcepts),
    JSON.stringify(mergedFilesRead),
    JSON.stringify(mergedFilesModified),
    JSON.stringify(newEmbedding),
    existingId,
  );
}
