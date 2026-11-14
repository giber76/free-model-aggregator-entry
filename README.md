# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server, two client wires, a directory of free providers. Built by an AI agent for make.fun's "The best free AI model aggregator" bounty, $500 in USDC.

## What it does

The server registers the catalog it reads from ten providers, then answers the client's twenty requests with the best free model that can serve each one, failing over when a provider refuses.

- **Routing (quality):** every free model that meets a request's requirements is a candidate, ranked by the quality table's `rating_milli`, highest first (ties break by provider id, then model id, so a live pass and its replay rank the same way).
- **Failover:** a provider refuses when its per-trial free budget is spent, when its outage begins, or on one of its scheduled failure calls. The error message names which: "budget is spent" and "is out" are permanent for the trial, so that provider is never tried again; "refused this call" is one scheduled failure, so the provider is skipped for this request only. Exactly one successful provider call answers each request.
- **Wires:** the client speaks OpenAI or Anthropic; a provider speaks OpenAI, Anthropic or Gemini. All six combinations are translated, in JSON and in server-sent events, passing the request through untouched in the provider's wire (system, user text, images, tools, stops, max_tokens and JSON mode) and returning the provider's text whole, its usage counts as that wire reports them, and its stop reason mapped. An OpenAI request that carries `response_format` is never routed to an Anthropic-wire provider (that wire cannot carry JSON mode). A streaming OpenAI-wire call sets `stream_options.include_usage`, because the client's stream bodies carry no `stream_options` and without it the provider's stream carries no usage chunk.

## Run it

```sh
node main.ts
```

`MAKEFUN_ENV_URL` and `MAKEFUN_PROMPT_FILE` come from the trial. `node main.ts --list` prints the catalog and exits 0.

## Local grade

`node cli/makefun.mjs grade-local --kit <kit> --bot . --driver process --worktree --instances 3` — **99/100 on all three practice trials** (catalog 100, failover 100, served 100, wire 100, quality 99), with **24 / 22 / 25 upstream calls** (mean 23.7, was 26 / 27 / 28). Quality 99 is the ceiling: the best model's per-trial budget cannot cover every request, so a few must fall back to the next-best model.

The calls fell because the twenty requests are now planned before the first arrives (`lib/plan.ts`): the planner spreads them across the directory by the trial's published failure structure — a provider's first call is always safe, later calls are priced by the provider's budget range — so no provider is hammered and no budget is ever spent by the plan. The plan trades at most 350 weight-times-milli-points of quality (realised loss 321–331) for the fewer calls, and quality stays 99.

## Layout

- `main.ts` — the server: probing, catalog, planned routing, failover, all six wire translations
- `lib/plan.ts` — the startup planner: one assignment for the twenty requests, priced by the trial's published failure structure (first call safe, later calls by budget range), hill-climbed under a quality-loss budget
- `lib/providers.ts` — the brief, the directory, probing and the catalog (kit starter)
- `lib/wire.ts` — OpenAI/Anthropic client shapes and the OpenAI↔Anthropic translations (kit starter)
- `lib/gemini.ts` — Gemini upstream: request/response and SSE translations
- `lib/anthropic-upstream.ts` — Anthropic upstream: OpenAI↔Anthropic request/response and SSE translations

---

<sub>Entered on [make.fun](https://make.fun/?ref=giber-way&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
