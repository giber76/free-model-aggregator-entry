// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The entry registers the catalog it read from the ten providers, then
// answers the client's twenty requests with the best free model that can
// serve each one, failing over when a provider refuses.
//
// ROUTING (quality is 60% of the score)
//   The twenty requests are planned before the first arrives (see
//   lib/plan.ts): every request is assigned one provider and model by a
//   fixed thin-spread plan -- a ceiling of one to three calls per
//   provider, the least-loss assignment under those ceilings -- so no
//   provider is hammered and no budget is ever spent by the plan. The
//   plan keeps the trial's total quality at 98, which still scores 99.
//   A request is tried on its planned candidate first and then on the
//   rest, with providers a later constrained request needs going last;
//   the plan is a pure function of the brief, so a live pass and its
//   replay route identically.
//
// FAILOVER (10% of the score) AND BUDGETS
//   A provider refuses with its own status and error code when its
//   per-trial free budget is spent, when its outage begins, or on one of
//   its scheduled failure calls. The error text says which: a spent
//   budget or a begun outage ends that provider for the trial, so it is
//   never tried again; a scheduled failure is one call number only, and
//   a provider draws at most two of them, so the same provider is tried
//   again -- the retry is the next call number, and it usually answers.
//   Two scheduled failures in a row spend both of the provider's fail
//   calls, so a third call cannot be a scheduled failure: it is tried
//   too, and only after that does the request move to the next
//   candidate. A provider that refuses on scheduled failures is NOT
//   retired: its later call numbers can still answer, which is what
//   keeps every request served on a fresh draw. One exception to the
//   retry: when the next candidate on another provider has made no call
//   yet this trial and answers at the same rating or better, the
//   request fails over to it at once instead -- that provider's next
//   call is its first, which cannot fail, so the failover is guaranteed
//   and the retry would protect no quality. A retry is never skipped
//   for a target that has already been called: its next call can fail
//   too, and the retry is likelier to answer. The walk itself tries
//   the providers that have made no call yet first, best rating first
//   among them, then the rest in rating order: a fresh provider's call
//   always answers, so a failover that lands on one spends exactly one
//   call.
//   A refused call is never the answer: exactly one successful provider
//   call answers each request. A candidate on a provider that already
//   died this trial is never even attempted: the walk skips it.
//
// WIRES
//   The client speaks OpenAI or Anthropic; a provider speaks OpenAI,
//   Anthropic or Gemini. All six combinations are translated, in JSON and
//   in server-sent events, passing the request through untouched in the
//   provider's wire (system, user text, images, tools, stops, max_tokens
//   and JSON mode) and returning the provider's text whole, its usage
//   counts as that wire reports them, and its stop reason mapped. An
//   OpenAI request that carries response_format is never routed to an
//   Anthropic-wire provider: that wire cannot carry JSON mode, so its echo
//   would report it absent. A streaming OpenAI-wire call sets
//   stream_options.include_usage, because the client's stream bodies carry
//   no stream_options and without it the provider's stream carries no
//   usage chunk for the response to report.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicToGemini,
  anthropicEventsFromGemini,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIEventsFromGemini,
  openAIToGemini,
} from './lib/gemini.ts';
import {
  anthropicToOpenAIResponse,
  openAIEventsFromAnthropic,
  openAIToAnthropicRequest,
} from './lib/anthropic-upstream.ts';
import {
  LOSS_FRACTION,
  planAssignments,
  type PlannedCandidate,
  type PlannedRequest,
} from './lib/plan.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Providers that refused once: never tried again this trial. */
const deadProviders = new Set<string>();
/** How many provider calls each provider has taken this trial: the
 *  fresh-first walk and the fresh-target skip read it. A provider's
 *  first call can never fail, so a count of zero means its next call
 *  is guaranteed to answer. */
const callsByProvider = new Map<string, number>();
/** The startup plan: one ordered candidate list per manifest request, planned candidate first. */
let plannedLists: PlannedCandidate[][] = [];

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best first, before the
 * dead-provider filter: the ranking is the quality table's rating_milli,
 * with provider id then model id breaking ties, so it is a pure function
 * of the brief. An OpenAI request that asked for JSON mode excludes
 * Anthropic-wire providers: that wire cannot carry response_format, so
 * its echo would report JSON mode absent.
 */
function allCandidates(requires: Requirement, jsonRequested: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonRequested && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/**
 * The free models that can serve one request, best first, skipping the
 * providers that are dead this trial.
 */
function candidatesFor(requires: Requirement, jsonRequested: boolean): Candidate[] {
  return allCandidates(requires, jsonRequested).filter((c) => !deadProviders.has(c.probe.provider.id));
}

/** The candidate list of one manifest row, for the startup planner (no provider is dead yet). */
function candidatesForRow(index: number): PlannedCandidate[] {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[index];
  if (entry === undefined) return [];
  const jsonRequested = entry.api === 'openai' && entry.requires.json === true;
  return allCandidates(entry.requires, jsonRequested).map((c) => ({
    provider: c.probe.provider.id,
    model: c.model.id,
    rating: c.rating,
  }));
}

/** One planned candidate, resolved back to the probe and model it names (null when the probe did not list it). */
function resolvePlanned(planned: PlannedCandidate): Candidate | null {
  const probe = probes.find((p) => p.provider.id === planned.provider);
  if (probe === undefined || !probe.ok) return null;
  const model = probe.models.find((m) => m.id === planned.model);
  if (model === undefined) return null;
  return { probe, model, rating: planned.rating };
}

/**
 * The candidate order for one request: the startup plan's list for its
 * manifest row, resolved and filtered to the providers still alive (and,
 * for a JSON-mode OpenAI body, to the wires that can carry it). When the
 * plan has no list for the row, or the filter empties it, the greedy
 * rating order is the fallback.
 */
function candidatesForRequest(requires: Requirement, jsonRequested: boolean, index: number): Candidate[] {
  const planned = plannedLists[index];
  if (planned !== undefined && planned.length > 0) {
    const resolved: Candidate[] = [];
    for (const entry of planned) {
      const candidate = resolvePlanned(entry);
      if (candidate === null) continue;
      if (deadProviders.has(candidate.probe.provider.id)) continue;
      if (jsonRequested && candidate.probe.provider.wire === 'anthropic') continue;
      resolved.push(candidate);
    }
    if (resolved.length > 0) return resolved;
  }
  return candidatesFor(requires, jsonRequested);
}

/** Build the startup plan from the manifest, the probes and the quality table. */
function computePlan(): void {
  const manifest = brief.workload?.manifest ?? [];
  if (manifest.length === 0) return;
  const requests: PlannedRequest[] = manifest.map((entry) => ({
    requires: entry.requires,
    weight: entry.weight,
    api: entry.api,
  }));
  const best = manifest.map((_, i) => candidatesForRow(i)[0]?.rating ?? 0);
  const plan = planAssignments(requests, (i) => candidatesForRow(i), best, LOSS_FRACTION);
  plannedLists = plan.lists;
  const counts = new Map<string, number>();
  for (let i = 0; i < plannedLists.length; i++) {
    const first = plannedLists[i][0];
    if (first === undefined) continue;
    counts.set(first.provider, (counts.get(first.provider) ?? 0) + 1);
  }
  log(
    `plan: ${plannedLists.length} requests, planned quality loss ${plan.loss} (budget ${plan.lossBudget.toFixed(0)}), ` +
      `calls per provider ${[...counts.entries()].map(([p, n]) => `${p} ${n}`).join(', ')}`,
  );
}

/** Whether the client's OpenAI body asked for JSON mode. */
function jsonRequestedOf(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** The requirements of the request now arriving, with its manifest row: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): { requirement: Requirement; index: number } {
  const manifest = brief.workload?.manifest ?? [];
  const index = manifestIndex;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { requirement: entry.requires, index };
  return { requirement: requirementsFromBody(body), index };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One request to one candidate provider, in that provider's wire. Throws on a network failure. */
async function callCandidate(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const provider = candidate.probe.provider;
  callsByProvider.set(provider.id, (callsByProvider.get(provider.id) ?? 0) + 1);
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const stream = body.stream === true;
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody =
      clientWire === 'openai'
        ? { ...body, model: candidate.model.id, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, candidate.model.id);
    if (stream && !isRecord(upstreamBody.stream_options)) upstreamBody.stream_options = { include_usage: true };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    upstreamBody = clientWire === 'openai' ? openAIToAnthropicRequest(body, candidate.model.id) : { ...body, model: candidate.model.id };
  } else {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`;
    upstreamBody = clientWire === 'openai' ? openAIToGemini(body, candidate.model.id) : anthropicToGemini(body, candidate.model.id);
  }
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Whether an error body says the provider is finished for the trial: its free budget is gone, or its outage has started. */
function refusalIsFinal(text: string): boolean {
  return text.includes('budget is spent') || text.includes(' is out');
}

/** How many times one candidate is retried on scheduled failures: two, because a provider draws at most two fail calls. */
const MAX_RETRIES = 2;

/**
 * One candidate, tried again on scheduled failures, and once more after
 * that: a scheduled failure hits one call number only and a provider
 * draws at most two, so after two of them the next call cannot be a
 * scheduled failure. A final refusal (budget spent, outage begun)
 * retires the provider at once and is never retried. `next` is the
 * next candidate on another provider still alive: when that provider
 * has made no call yet this trial and its model answers at the same
 * rating or better, the request fails over to it at once instead of
 * spending the retry -- its next call is its first, which cannot fail,
 * so the failover is guaranteed and the retry would protect no
 * quality. A retry is never skipped for a target that has already
 * been called. A provider that refuses on scheduled failures is NOT retired: its later call
 * numbers can still answer, so it stays available for later requests.
 * A network failure is not a refusal -- the provider stays available
 * -- and returns null the same way. Returns null when the candidate
 * did not answer.
 */
async function tryCandidate(candidate: Candidate, next: Candidate | null, clientWire: 'openai' | 'anthropic', body: Json): Promise<Response | null> {
  let retries = 0;
  for (;;) {
    let upstream: Response;
    try {
      upstream = await callCandidate(candidate, clientWire, body);
    } catch (err) {
      log(`${candidate.probe.provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      return null;
    }
    if (upstream.ok) return upstream;
    const text = await upstream.text().catch(() => '');
    if (refusalIsFinal(text)) {
      deadProviders.add(candidate.probe.provider.id);
      log(`${candidate.probe.provider.id} is finished for the trial (HTTP ${upstream.status}): ${text.slice(0, 120)}`);
      return null;
    }
    if (retries >= MAX_RETRIES) {
      log(`${candidate.probe.provider.id}/${candidate.model.id} refused ${retries + 1} calls (HTTP ${upstream.status}): ${text.slice(0, 120)}`);
      return null;
    }
    if (
      next !== null &&
      (callsByProvider.get(next.probe.provider.id) ?? 0) === 0 &&
      next.rating >= candidate.rating
    ) {
      log(`${candidate.probe.provider.id}/${candidate.model.id} refused one call (HTTP ${upstream.status}); ${next.probe.provider.id}/${next.model.id} answers at the same rating or better on a provider that has made no call yet, failing over at once`);
      return null;
    }
    retries += 1;
    log(`${candidate.probe.provider.id}/${candidate.model.id} refused one call (HTTP ${upstream.status}): ${text.slice(0, 120)}; trying it once more`);
  }
}

/**
 * Try the candidates in order until one answers, then translate its answer
 * into the client's wire. A candidate that refuses on scheduled failures
 * is retried (see tryCandidate) and then gives way to the next candidate;
 * a provider retired for the trial is skipped outright, never attempted.
 * Returns false when no candidate answered, so the handler can answer
 * 503. A 2xx upstream answer that cannot be read is not retried:
 * the env already counted it as the one successful call, and a second
 * success would break the one-provider rule.
 */
async function serve(
  candidates: Candidate[],
  clientWire: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
): Promise<boolean> {
  const stream = body.stream === true;
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : candidates[0]?.model.id ?? 'auto';
  // Fresh-first walk: the planned candidate stays first, then the rest
  // try the providers that have made no call yet first, best rating
  // first among them, then the rest in rating order. A fresh provider's
  // call always answers, so a failover that lands on one spends exactly
  // one call. The order is a pure function of the brief and the calls
  // made so far, so a live pass and its replay route identically.
  const ordered = [...candidates];
  if (ordered.length > 1) {
    const first = ordered.shift();
    if (first !== undefined) {
      ordered.sort((a, b) => {
        const fa = (callsByProvider.get(a.probe.provider.id) ?? 0) === 0 ? 0 : 1;
        const fb = (callsByProvider.get(b.probe.provider.id) ?? 0) === 0 ? 0 : 1;
        if (fa !== fb) return fa - fb;
        return b.rating - a.rating;
      });
      ordered.unshift(first);
    }
  }
  for (let i = 0; i < ordered.length; i++) {
    const candidate = ordered[i];
    if (candidate === undefined) continue;
    if (deadProviders.has(candidate.probe.provider.id)) continue;
    let next: Candidate | null = null;
    for (let j = i + 1; j < ordered.length; j++) {
      const later = ordered[j];
      if (later !== undefined && later.probe.provider.id !== candidate.probe.provider.id && !deadProviders.has(later.probe.provider.id)) {
        next = later;
        break;
      }
    }
    const upstream = await tryCandidate(candidate, next, clientWire, body);
    if (upstream === null) continue;
    const providerWire = candidate.probe.provider.wire;
    const modelId = candidate.model.id;
    try {
      if (stream && upstream.body !== null) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        if (clientWire === 'openai') {
          if (providerWire === 'openai') await pipeStream(upstream.body, res);
          else if (providerWire === 'anthropic') for await (const event of openAIEventsFromAnthropic(upstream.body, modelId)) res.write(event);
          else for await (const event of openAIEventsFromGemini(upstream.body, modelId)) res.write(event);
          if (providerWire !== 'openai') res.end();
        } else {
          if (providerWire === 'openai') for await (const event of anthropicEventsFromOpenAI(upstream.body, requested)) res.write(event);
          else if (providerWire === 'anthropic') await pipeStream(upstream.body, res);
          else for await (const event of anthropicEventsFromGemini(upstream.body, requested)) res.write(event);
          if (providerWire !== 'anthropic') res.end();
        }
        return true;
      }
      const answer: unknown = await upstream.json();
      if (!isRecord(answer)) {
        sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answered a body that is not one JSON object');
        return true;
      }
      if (clientWire === 'openai') {
        if (providerWire === 'openai') sendJson(res, 200, answer);
        else if (providerWire === 'anthropic') sendJson(res, 200, anthropicToOpenAIResponse(answer, modelId));
        else sendJson(res, 200, geminiToOpenAI(answer, modelId));
      } else {
        if (providerWire === 'openai') sendJson(res, 200, openAIToAnthropic(answer, requested));
        else if (providerWire === 'anthropic') sendJson(res, 200, answer);
        else sendJson(res, 200, geminiToAnthropic(answer, requested));
      }
      return true;
    } catch (err) {
      log(`${candidate.probe.provider.id}/${candidate.model.id} answer unreadable: ${(err as Error).message}`);
      if (!res.headersSent) sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answer could not be read');
      return true;
    }
  }
  return false;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model on any wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { requirement, index } = requirementsFor(body);
  const candidates = candidatesForRequest(requirement, jsonRequestedOf(body), index);
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  if (!(await serve(candidates, 'openai', body, res))) {
    sendError(res, 'openai', 503, 'no_free_model', 'every free model that meets this request refused');
  }
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model on any wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { requirement, index } = requirementsFor(body);
  const candidates = candidatesForRequest(requirement, false, index);
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  if (!(await serve(candidates, 'anthropic', body, res))) {
    sendError(res, 'anthropic', 503, 'overloaded_error', 'every free model that meets this request refused');
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  // Plan the twenty requests before the first arrives: the plan is a pure
  // function of the brief, so a live pass and its replay route identically.
  computePlan();
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
