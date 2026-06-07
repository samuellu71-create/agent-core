/**
 * Hotness scoring for memory lifecycle management.
 *
 * Mirrors OpenViking's memory_lifecycle.py:19-64 — blends access
 * frequency (active_count) with recency (updated_at) into a 0-1 score.
 *
 * Formula:
 *   hotness = sigmoid(log1p(active_count)) * time_decay(updated_at)
 *   time_decay = exp(-ln2 / half_life * age_days)
 *
 * OpenViking (memory_lifecycle.py:48): freq = 1/(1+exp(-log1p(active_count)))
 * OpenViking (memory_lifecycle.py:61): decay = exp(-decay_rate * age_days)
 *
 * Our implementation is a direct port with identical math.
 */

const DEFAULT_HALF_LIFE_DAYS = 7.0;

/**
 * Compute a 0.0-1.0 hotness score.
 *
 * @param activeCount Number of times this memory was accessed/retrieved.
 * @param updatedAt   Last update/access timestamp (ISO string or Date).
 * @param now         Current time override (for deterministic tests).
 * @param halfLifeDays Half-life for recency decay.
 */
export function hotnessScore(
  activeCount: number,
  updatedAt: string | Date | null,
  now?: Date,
  halfLifeDays: number = DEFAULT_HALF_LIFE_DAYS,
): number {
  const currentTime = now ?? new Date();

  // Frequency component: sigmoid(log1p(active_count))
  const freq = 1.0 / (1.0 + Math.exp(-Math.log1p(activeCount)));

  // Recency component
  if (updatedAt === null) return 0.0;
  const updatedTime = typeof updatedAt === "string" ? new Date(updatedAt) : updatedAt;
  const ageDays = Math.max((currentTime.getTime() - updatedTime.getTime()) / 86_400_000, 0);
  const decayRate = Math.LN2 / halfLifeDays;
  const recency = Math.exp(-decayRate * ageDays);

  return freq * recency;
}
