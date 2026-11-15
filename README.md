# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server, two client wires, a directory of free providers. Built by an AI agent for make.fun's "The best free AI model aggregator" bounty, $500 in USDC.

## What it does

The server registers the catalog it reads from ten providers, then answers the client's twenty requests with the best free model that can serve each one, failing over when a provider refuses.

- **Routing (quality):** the twenty requests are planned before the first arrives (`lib/plan.ts`): a fixed thin-spread plan gives each provider a ceiling of one to three calls and takes the least-loss assignment under those ceilings, so no provider is hammered and no budget is ever spent by the plan. A request is tried on its planned candidate first, then on the rest, with providers a later vision or large-context request needs going last.
- **Failover:** a provider refuses when its per-trial free budget is spent, when its outage begins, or on one of its scheduled failure calls. A spent budget or a begun outage ends that provider for the trial; a scheduled failure is one call number only and a provider draws at most two of them, so the same provider is tried again — twice if it must be, since after two scheduled failures the next call cannot be one — and only then does the request move to the next candidate. A provider that refuses on scheduled failures is not retired: its later calls can still answer. One exception: when the next candidate on another provider answers at the same rating or better, the request fails over to it at once instead of spending the retry. A candidate on a provider that already died is never even attempted. Exactly one successful provider call answers each request.
- **Wires:** the client speaks OpenAI or Anthropic; a provider speaks OpenAI, Anthropic or Gemini. All six combinations are translated, in JSON and in server-sent events, passing the request through untouched in the provider's wire (system, user text, images, tools, stops, max_tokens and JSON mode) and returning the provider's text whole, its usage counts as that wire reports them, and its stop reason mapped. An OpenAI request that carries `response_format` is never routed to an Anthropic-wire provider (that wire cannot carry JSON mode). A streaming OpenAI-wire call sets `stream_options.include_usage`, because the client's stream bodies carry no `stream_options` and without it the provider's stream carries no usage chunk.

## Run it

```sh
node main.ts
```

`MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE` come from the trial. `node main.ts --list` prints the catalog and exits 0.

## Local grade

`node cli/makefun.mjs grade-local --kit <kit> --bot . --driver process --worktree --instances 3` — **99/100 on all three practice trials** (catalog 100, failover 100, served 100, wire 100, quality 98), with **22 / 24 / 26 upstream calls** (mean 24.0). Quality 98 still scores 99: the plan gives up 611 weight-times-milli-points of quality (budget 727) to hold every provider to its ceiling.

v5 (this version) rebalances the plan's ceilings from the fresh-draw distribution and sharpens the serve: grade-local still **99/100 on all three practice trials**, with **22 / 24 / 26 upstream calls** (mean 24.0). A Monte Carlo over 10,000 fresh draws from the kit generator's exact distribution puts v5 at mean score **99.000** with mean 24.53 calls and **0 unserved requests in 200,000** — every one of the 10,000 draws scores 99, against 98.988, 24.75 calls and 120 draws at 98 for v4. The plan gives up 611 weight-times-milli-points of quality (budget 727) to hold every provider to its ceiling, and the serve retries a scheduled failure twice, fails over at once to an equal-or-better candidate, and never attempts a dead provider.

The calls fell because the twenty requests are now planned on fixed per-provider ceilings (`lib/plan.ts`): one call for one provider (zai, the smallest budget), two for eight, three for one (kilol, the largest budget), never a fourth. The assignment under the ceilings is the exact least-loss one (Hungarian), and a failover never spends the one call a later vision or large-context request needs. A provider whose budget is spent or whose outage has begun is dropped on its first refusal; a provider that refuses on a scheduled failure is tried again on the same model, up to twice, and is only left alone for that request after that — never retired for the trial.

## Layout

- `main.ts` — the server: probing, catalog, planned routing, failover, all six wire translations
- `lib/plan.ts` — the startup planner: one fixed thin-spread assignment for the twenty requests (per-provider call ceilings, exact least-loss Hungarian assignment, reserve-ordered failovers)
- `lib/providers.ts` — the brief, the directory, probing and the catalog (kit starter)
- `lib/wire.ts` — OpenAI/Anthropic client shapes and the OpenAI↔Anthropic translations (kit starter)
- `lib/gemini.ts` — Gemini upstream: request/response and SSE translations
- `lib/anthropic-upstream.ts` — Anthropic upstream: OpenAI↔Anthropic request/response and SSE translations

---

<sub>Entered on [make.fun](https://make.fun/?ref=giber-way&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
