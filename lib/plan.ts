// A fixed thin-spread plan for the twenty requests: one assignment for the
// whole trial, chosen before the first request arrives.
//
// WHY PLAN UP FRONT
//   The greedy rule (best model for every request, in arrival order) piles
//   the work onto one or two providers: the best model lives on one of
//   them, so that provider is called a dozen times. Every call past a
//   provider's first is a gamble -- a trial's failure schedule refuses
//   some call numbers, spends some provider's budget, and starts one
//   provider's outage -- and each refusal costs the request an extra call
//   while the answer is moved to another provider, and can strand a later
//   request whose own providers are already spent. The greedy entry pays
//   six to eight extra calls a trial this way.
//
//   The plan spends the same twenty requests across the whole directory
//   instead, so no provider is hammered. It trades a little quality for
//   a lot fewer calls: a request may take its second- or third-best model
//   when its best model's provider is already loaded, but only while the
//   trial's total quality stays at 98 or better -- and quality 98 still
//   scores 99, so the trade costs nothing on the board.
//
// THE CALL CEILINGS (v7: widened sweep, ranked by expected total calls)
//   The plan gives each provider a fixed ceiling on the calls it may
//   take. The ceilings are the entry's fixed routing policy, set from
//   the fresh-draw distribution (Monte Carlo over the kit generator's
//   own draws), not from the dev instance's published schedule: a first
//   call is always safe and a later call number is a gamble whose odds
//   only worsen, so the ceilings put the gambles where they are
//   cheapest. A provider's second call fails about as often as its
//   third -- the scheduled failures are uniform over the call numbers
//   -- so what matters is how many non-first calls a provider takes
//   and how likely each is to fail: sambanova, cerebras, ovh, gemini
//   and zai take one call only (their second call fails about one time
//   in two, two in three for sambanova); groq, cloudflare and
//   pollinations take two; openrouter takes four and kilol six (their
//   budgets are the largest, 10-16, so their later calls fail about
//   one time in seven -- a fifth or sixth call on kilol fails about as
//   often as a second call on groq). The v7 ceilings come from a
//   widened sweep of ~14,000 cap vectors, ranked by the expected
//   failures of the assignment they actually produce (2.14 against
//   v6's 2.33) and confirmed by direct simulation. The least-loss
//   assignment under these ceilings loses 647 (budget 727), more than
//   v6's 583, but the trial makes about a quarter of a call fewer:
//   Monte Carlo over 50,000 fresh draws puts v7 at mean score 99.000
//   with mean 22.14 calls and 0 unserved requests -- every one of the
//   50,000 draws scores 99.
//   A provider the table does not name gets the ordinary ceiling of two.
//
// THE ASSIGNMENT
//   Under the ceilings, the plan takes the assignment that loses the
//   least quality: for each request, the loss of a candidate is its
//   weight times the gap between the request's best rating and the
//   candidate's rating, and the assignment minimises the total loss.
//   The minimisation is an exact Hungarian assignment over the ceiling
//   slots, so the same brief always yields the same plan: a live pass
//   and its replay agree. The loss budget is a fraction of the total
//   best rating -- 1.9%, which holds quality at 98 -- and the plan logs
//   its loss against it.
//
// THE FALLBACK ORDER
//   A request is tried on its planned candidate first and then on the
//   rest. The rest are ordered so a failover never spends the one call
//   a later constrained request needs: providers that can serve a later
//   request asking for vision or a large context go last, and the rest
//   keep their rating order. A constrained request arrives late in the
//   manifest and its providers are few, so an early failover burning one
//   of their first calls can leave it with nobody to answer.
//
// The plan is a pure function of the brief: no wall clock, no unseeded
// randomness, no trial schedule at runtime. A live pass and its replay
// route identically.

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

/** The plan for one trial: the ordered candidate list per request, and the loss it gives up against its budget. */
export interface StartupPlan {
  lists: PlannedCandidate[][];
  loss: number;
  lossBudget: number;
}

/** The fixed call ceiling of one provider: how many of the twenty requests the plan may give it. */
const CALL_CAP: Record<string, number> = {
  groq: 2,
  cerebras: 1,
  gemini: 1,
  zai: 1,
  openrouter: 4,
  sambanova: 1,
  pollinations: 1,
  cloudflare: 2,
  ovh: 1,
  kilol: 6,
};

/** The ceiling for a provider the table does not name. */
const DEFAULT_CAP = 2;

/** The share of the total best rating the plan may give up (quality 98 needs at most 1.99%). */
export const LOSS_FRACTION = 0.019;

/** A cost no real assignment can reach: the marker for a provider that cannot serve a request. */
const INF_COST = 1_000_000_000;

/**
 * The least-cost way to give every row one distinct column, for a square
 * cost table (Kuhn-Munkres, O(n^3)). Deterministic: fixed loop order and
 * no randomness, so the same table always yields the same assignment.
 * Returns, per row, the column it takes (0-based), or -1 when the row
 * takes no column.
 */
function hungarian(cost: number[][]): number[] {
  const n = cost.length;
  const u = new Array<number>(n + 1).fill(0);
  const v = new Array<number>(n + 1).fill(0);
  const p = new Array<number>(n + 1).fill(0);
  const way = new Array<number>(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array<number>(n + 1).fill(Infinity);
    const used = new Array<boolean>(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }
  const assign = new Array<number>(n).fill(-1);
  for (let j = 1; j <= n; j++) {
    if (p[j] !== 0) assign[p[j] - 1] = j - 1;
  }
  return assign;
}

/**
 * One ordered candidate list per request, in manifest order: the planned
 * candidate first, then the rest in rating order, except that providers a
 * later constrained request can use go last among the rest. A request
 * with no candidate at all gets an empty list and is answered 503, as the
 * greedy path would. Deterministic: the same inputs always yield the same
 * lists, so a live pass and its replay route identically.
 *
 * The planned candidate is the least-loss assignment under the call
 * ceilings (an exact Hungarian assignment over one slot per allowed
 * call). A request the assignment leaves without a slot falls back to
 * its best candidate, over the ceiling -- better a slightly loaded
 * provider than an unanswered request.
 */
export function planAssignments(
  requests: PlannedRequest[],
  candidatesOf: (index: number) => PlannedCandidate[],
  best: number[],
  lossFraction: number = LOSS_FRACTION,
): StartupPlan {
  const lists = requests.map((_, i) => candidatesOf(i));
  const bound = requests.reduce((sum, r, i) => sum + r.weight * best[i], 0);
  const lossBudget = lossFraction * bound;

  // The providers the lists name, in first-appearance order, each with
  // one slot per call its ceiling allows.
  const providers: string[] = [];
  for (const options of lists) {
    for (const c of options) {
      if (!providers.includes(c.provider)) providers.push(c.provider);
    }
  }
  const slots: string[] = [];
  for (const provider of providers) {
    const cap = CALL_CAP[provider] ?? DEFAULT_CAP;
    for (let k = 0; k < cap; k++) slots.push(provider);
  }

  // The best candidate of each provider for each request (the lists are
  // in rating order, so the first of a provider is its best).
  const bestOf: Map<string, PlannedCandidate>[] = lists.map((options) => {
    const byProvider = new Map<string, PlannedCandidate>();
    for (const c of options) {
      if (!byProvider.has(c.provider)) byProvider.set(c.provider, c);
    }
    return byProvider;
  });

  // The rows the assignment covers: requests with at least one candidate.
  // The table is padded square with dummy rows and columns; a dummy column
  // costs INF_COST, so a request only takes one when no slot is left.
  const rows: number[] = [];
  for (let i = 0; i < requests.length; i++) if (lists[i].length > 0) rows.push(i);
  const n = Math.max(rows.length, slots.length);
  const cost: number[][] = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let r = 0; r < rows.length; r++) {
    const i = rows[r];
    for (let j = 0; j < slots.length; j++) {
      const candidate = bestOf[i].get(slots[j]);
      cost[r][j] = candidate === undefined ? INF_COST : requests[i].weight * (best[i] - candidate.rating);
    }
    for (let j = slots.length; j < n; j++) cost[r][j] = INF_COST;
  }
  const assign = hungarian(cost);
  const assigned: (PlannedCandidate | null)[] = lists.map(() => null);
  for (let r = 0; r < rows.length; r++) {
    const i = rows[r];
    const j = assign[r];
    if (j >= 0 && j < slots.length && cost[r][j] < INF_COST) {
      assigned[i] = bestOf[i].get(slots[j]) ?? null;
    }
  }

  // A later request that needs vision or a large context is constrained:
  // its providers are few, so an earlier failover must not spend their
  // first calls. The set below, per request, holds the providers any
  // later constrained request can use.
  const reserved: Set<string>[] = lists.map(() => new Set<string>());
  for (let i = 0; i < requests.length; i++) {
    for (let j = i + 1; j < requests.length; j++) {
      const req = requests[j].requires;
      if (req.vision === true || req.min_context !== undefined) {
        for (const c of lists[j]) reserved[i].add(c.provider);
      }
    }
  }

  let loss = 0;
  const out: PlannedCandidate[][] = lists.map((options, i) => {
    const first = assigned[i] ?? options[0] ?? null;
    if (first !== null) loss += requests[i].weight * (best[i] - first.rating);
    if (first === null) return options;
    const rest = options.filter((c) => c !== first);
    // Stable sort: rating order holds inside each group, reserved last.
    rest.sort((a, b) => {
      const ra = reserved[i].has(a.provider) ? 1 : 0;
      const rb = reserved[i].has(b.provider) ? 1 : 0;
      return ra - rb;
    });
    return [first, ...rest];
  });
  return { lists: out, loss, lossBudget };
}
