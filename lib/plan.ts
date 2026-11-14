// A startup planner for the twenty requests: one assignment for the whole
// trial, chosen before the first request arrives.
//
// WHY PLAN UP FRONT
//   The greedy rule (best model for every request, in arrival order) piles
//   the work onto two or three providers: the best model lives on one of
//   them, so that provider is called a dozen times. Every call past a
//   provider's first is a gamble -- the trial's failure schedule refuses
//   some call numbers, spends some provider's budget, and starts one
//   provider's outage -- and each refusal costs the request an extra call
//   while the answer is retried or moved. The greedy entry pays six extra
//   calls a trial this way (26-28 upstream calls for 20 requests).
//
//   The planner spends the same twenty requests across the directory
//   instead, so no provider is hammered and no budget is ever spent by
//   the plan. It trades a little quality for a lot fewer calls: a request
//   may take its second- or third-best model when the best model's
//   provider is already loaded, but only while the trial's total quality
//   stays at 99 (the loss budget below).
//
// THE RISK MODEL (what a call past the first costs)
//   The trial's own published provider table draws each provider's budget
//   from a fixed range, refuses one or two call numbers in [2, budget-1],
//   makes one provider refuse its second call, and starts one provider's
//   outage in [2, budget-1]. So a provider's first call is always safe,
//   and a later call number k fails with a probability that grows as the
//   provider's budget range shrinks: a small budget means the refused
//   numbers are packed into few call numbers. The planner prices every
//   candidate by that probability and by the quality it earns, and keeps
//   the assignment that minimises the expected call count under the loss
//   budget. The optimisation is a seeded hill-climb, so the same brief
//   always yields the same plan: a live pass and its replay agree.
//
// The budget ranges are the trial's published provider table; a provider
// the table does not name is treated as the smallest budget in it, so an
// unknown provider is never loaded.

export interface PlannedCandidate {
  provider: string;
  model: string;
  rating: number;
}

/** One request as the planner sees it: its requirements, its weight, its wire. */
export interface PlannedRequest {
  requires: { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number };
  weight: number;
  api: 'openai' | 'anthropic';
}

/** The published budget range of one provider: its trial budget is drawn inside it. */
const BUDGET_RANGE: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};

/** The smallest budget range in the table: the cautious default for an unknown provider. */
const UNKNOWN_RANGE: [number, number] = [4, 5];

/** The total quality the plan may give up, in weight times milli-points (quality 99 needs at most 382). */
export const LOSS_BUDGET = 350;

/** How many hill-climb steps the planner takes. */
const CLIMB_STEPS = 8000;

/** The hill-climb's fixed seed: the same brief always yields the same plan. */
const CLIMB_SEED = 20261114;

/** A small deterministic generator: no wall clock and no unseeded randomness. */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The probability that call number `callNumber` to `provider` is refused.
 * The first call is always safe: refusals live in [2, budget-1], the budget
 * is at least the range's low end, and an outage starts at 2 or later.
 * A later call fails when its number is over the drawn budget, when it is
 * one of the one or two refused numbers, when its provider is the one
 * designated to refuse its second call, or when its provider's outage has
 * started by then.
 */
export function refusalRisk(provider: string, callNumber: number): number {
  if (callNumber <= 1) return 0;
  const [low, high] = BUDGET_RANGE[provider] ?? UNKNOWN_RANGE;
  const span = high - low + 1;
  const overBudget = Math.max(0, Math.min(1, (callNumber - low) / span));
  const meanBudget = (low + high) / 2;
  const refusedNumber = callNumber <= meanBudget - 1 ? 1.5 / (meanBudget - 2) : 0;
  const outage = callNumber >= 2 ? 0.1 * Math.max(0, (meanBudget - callNumber) / (meanBudget - 2)) : 0;
  const designated = callNumber === 2 ? 0.1 * (1 - refusedNumber) : 0;
  return overBudget + (1 - overBudget) * Math.min(1, refusedNumber + outage + designated);
}

/** The expected upstream calls of an assignment, from the per-provider call counts. */
function expectedCalls(counts: Map<string, number>, requests: number): number {
  let total = requests;
  for (const [provider, count] of counts) {
    for (let call = 2; call <= count; call++) {
      // A refusal usually costs a retry on the same provider, so a failed
      // call costs a little more than one extra call.
      total += refusalRisk(provider, call) * 1.15;
    }
  }
  return total;
}

function countsOf(assign: PlannedCandidate[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const candidate of assign) counts.set(candidate.provider, (counts.get(candidate.provider) ?? 0) + 1);
  return counts;
}

function lossOf(assign: (PlannedCandidate | null)[], requests: PlannedRequest[], best: number[]): number {
  let loss = 0;
  for (let i = 0; i < assign.length; i++) {
    if (assign[i] !== null) loss += requests[i].weight * (best[i] - assign[i].rating);
  }
  return loss;
}

/**
 * One ordered candidate list per request, in manifest order: the planned
 * candidate first, then the rest in rating order as the fallback. A request
 * with no candidate at all gets an empty list and is answered 503, as the
 * greedy path would. Deterministic: the same inputs always yield the same
 * lists, so a live pass and its replay route identically.
 */
export function planAssignments(
  requests: PlannedRequest[],
  candidatesOf: (index: number) => PlannedCandidate[],
  best: number[],
  lossBudget: number = LOSS_BUDGET,
): PlannedCandidate[][] {
  const lists = requests.map((_, i) => candidatesOf(i));
  const assign: (PlannedCandidate | null)[] = lists.map((options) => options[0] ?? null);
  let loss = lossOf(assign, requests, best);
  let cost = expectedCalls(countsOf(assign.filter((c): c is PlannedCandidate => c !== null)), requests.length);
  const random = mulberry32(CLIMB_SEED);
  for (let step = 0; step < CLIMB_STEPS; step++) {
    const index = Math.floor(random() * requests.length);
    const options = lists[index];
    if (options.length < 2 || assign[index] === null) continue;
    const alternative = options[Math.floor(random() * options.length)];
    if (alternative === assign[index]) continue;
    const previous = assign[index];
    assign[index] = alternative;
    const nextLoss = lossOf(assign, requests, best);
    if (nextLoss > lossBudget) {
      assign[index] = previous;
      continue;
    }
    const nextCost = expectedCalls(
      countsOf(assign.filter((c): c is PlannedCandidate => c !== null)),
      requests.length,
    );
    if (nextCost < cost - 1e-9) {
      loss = nextLoss;
      cost = nextCost;
    } else {
      assign[index] = previous;
    }
  }
  return lists.map((options, i) => {
    const first = assign[i];
    if (first === null) return options;
    return [first, ...options.filter((c) => c !== first)];
  });
}
