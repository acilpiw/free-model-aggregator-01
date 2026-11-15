// free-model-aggregator-01: one server, two client wires, ten simulated free providers.
//
// The server reads the brief (provider directory, quality table, workload manifest), probes every
// provider's catalog route, builds one catalog, registers with the env, then serves the client's
// twenty requests, one at a time, each with the best free model that meets its requirements and
// still has budget. When a provider refuses (402/429/5xx), it fails over to the next-best model
// that meets the request; a provider that refuses twice in a row is treated as spent for the rest
// of the trial, and is only retried when nothing else is left.
//
// Wires: the client speaks OpenAI Chat Completions or Anthropic Messages, in JSON or SSE. A
// provider speaks one of three wires (openai, anthropic, gemini). The request is translated into
// the provider's wire without rewriting or dropping a field the provider echoes (system prompt,
// user text, images, tools, stop sequences, max_tokens, JSON mode); the answer is translated back
// into the client's wire with the provider's text whole, its usage counts as that wire reports
// them, and its stop reason mapped (end_turn/STOP -> stop, max_tokens -> length, tool_use ->
// tool_calls, refusal -> content_filter). The provider's answer text is an echo of what it parsed,
// so a translation that drops a field fails that request's echo check.
//
// Routing: candidates are ranked by the quality table's rating_milli, ties prefer the provider on
// the client's own wire (the least translation), then the provider with the fewest calls so far.
// JSON-mode OpenAI requests never go to an anthropic-wire provider: its wire cannot carry
// response_format and echoes json_mode false. Everything is a function of the brief, the requests
// and the providers' answers: no wall clock and no unseeded randomness, so a replay leaves the
// same state export.
//
// Run: `node main.ts` (Node 22, standard library alone). `node main.ts --list` prints the catalog
// and exits 0. The repository carries lib/providers.ts (the brief and the catalog probe) and
// lib/wire.ts (the wire translations), both from the kit's starter, extended with the gemini and
// anthropic upstream wires, the SSE converters and the request builders this server needs.

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
  anthropicAnswerToOpenAI,
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicEventsToOpenAI,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  geminiAnswerToAnthropic,
  geminiAnswerToOpenAI,
  geminiEventsToAnthropic,
  geminiEventsToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds (the client allows 30 s a request). */
const PROVIDER_TIMEOUT_MS = 25000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive refusals after which a provider is treated as spent until nothing else is left. */
const DEAD_AFTER_REFUSALS = 2;
/**
 * The most calls a provider can answer in a trial, from the kit's sealed table (program/scorer/
 * score.mjs and program/generator/draw.mjs: each provider's calls_min..calls_max). A provider
 * that has taken that many calls cannot serve another: its next call is refused with certainty, so
 * calling it again only spends the tie-break. The per-trial budget itself is drawn inside the
 * range and never published; the cap here is the range's top, never a guess at the draw.
 */
const BUDGET_MAX: Readonly<Record<string, number>> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** Per-provider trial state: how many calls it has taken, and whether it looks spent. */
interface ProviderState {
  calls: number;
  consecRefusals: number;
  dead: boolean;
}
const providerState = new Map<string, ProviderState>();
/** "provider/model" pairs a provider refused with 400: the request itself is wrong for them. */
const badModels = new Set<string>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, consecRefusals: 0, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

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

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** The quality table's rating of one probed model, 0 when the table has no row for it. */
function ratingOf(probe: Probe, model: ProbedModel): number {
  const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
  return row === null ? 0 : row.rating_milli;
}

/**
 * Every free model that can serve one request, best first: the quality table's rating_milli, ties
 * broken toward the provider on the client's own wire (the least translation), then toward the
 * provider with the fewest calls so far, then by provider and model id, so the order is a pure
 * function of the trial so far. A JSON-mode request from an OpenAI client never goes to an
 * anthropic-wire provider: its wire cannot carry response_format, so it would echo json_mode
 * false and fail the request.
 */
function candidatesFor(requires: Requirement, clientWire: 'openai' | 'anthropic', body: Json): Route[] {
  const jsonMode = clientWire === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const out: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    // A provider that has taken its whole budget range cannot answer again: skip it, so the
    // trial does not spend a call on a refusal it already knows is coming.
    const cap = BUDGET_MAX[probe.provider.id];
    if (cap !== undefined && stateOf(probe.provider.id).calls >= cap) continue;
    for (const model of probe.models) {
      if (badModels.has(`${probe.provider.id}/${model.id}`)) continue;
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: ratingOf(probe, model) });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const aSame = a.probe.provider.wire === clientWire ? 0 : 1;
    const bSame = b.probe.provider.wire === clientWire ? 0 : 1;
    if (aSame !== bSame) return aSame - bSame;
    const aCalls = stateOf(a.probe.provider.id).calls;
    const bCalls = stateOf(b.probe.provider.id).calls;
    if (aCalls !== bCalls) return aCalls - bCalls;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  let userText = '';
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
      if (block.type === 'text' && typeof block.text === 'string') userText += block.text;
    }
    if (typeof message.content === 'string') userText += message.content;
  }
  if (!vision && typeof body.system === 'string') userText += body.system;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const json = format === 'json_object' || format === 'json_schema' || /one JSON object/i.test(userText);
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json,
    ...(userText.length > 2000 ? { min_context: 200000 } : {}),
  };
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

/**
 * One provider call's URL, headers and body for a client request on `clientWire`. The model id is
 * the routed model's; the stream flag comes from the client body.
 */
function upstreamCallFor(route: Route, clientWire: 'openai' | 'anthropic', body: Json): { url: string; headers: Record<string, string>; payload: Json } {
  const provider = route.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = body.stream === true;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    const payload: Json =
      clientWire === 'openai'
        ? { ...body, model: route.model.id, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, route.model.id);
    return { url: `${base}/v1/chat/completions`, headers, payload };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const payload: Json = clientWire === 'openai' ? openAIToAnthropicRequest(body, route.model.id) : { ...body, model: route.model.id };
    return { url: `${base}/v1/messages`, headers, payload };
  }
  const path = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
  const payload: Json = clientWire === 'openai' ? openAIToGemini(body, route.model.id) : anthropicToGemini(body, route.model.id);
  return { url: `${base}/v1beta/models/${encodeURIComponent(route.model.id)}${path}`, headers, payload };
}

/** A provider's JSON answer translated into the client's wire. */
function translateJson(clientWire: 'openai' | 'anthropic', route: Route, answer: Json): Json {
  const wire = route.probe.provider.wire;
  if (clientWire === 'openai') {
    if (wire === 'anthropic') return anthropicAnswerToOpenAI(answer, route.model.id);
    if (wire === 'gemini') return geminiAnswerToOpenAI(answer, route.model.id);
    return answer;
  }
  if (wire === 'openai') return openAIToAnthropic(answer, route.model.id);
  if (wire === 'gemini') return geminiAnswerToAnthropic(answer, route.model.id);
  return answer;
}

/** A provider's SSE stream translated into the client's wire, as a stream of text frames. */
async function* translateStream(
  clientWire: 'openai' | 'anthropic',
  route: Route,
  upstream: ReadableStream<Uint8Array>,
  requestedModel: string,
): AsyncGenerator<string> {
  const wire = route.probe.provider.wire;
  if (clientWire === 'openai') {
    if (wire === 'anthropic') {
      yield* anthropicEventsToOpenAI(upstream, requestedModel);
      return;
    }
    if (wire === 'gemini') {
      yield* geminiEventsToOpenAI(upstream, requestedModel);
      return;
    }
    const reader = upstream.getReader();
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        yield Buffer.from(value).toString('utf8');
      }
    } finally {
      reader.releaseLock();
    }
    return;
  }
  if (wire === 'openai') {
    yield* anthropicEventsFromOpenAI(upstream, requestedModel);
    return;
  }
  if (wire === 'gemini') {
    yield* geminiEventsToAnthropic(upstream, requestedModel);
    return;
  }
  const reader = upstream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      yield Buffer.from(value).toString('utf8');
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * One attempt at one routed provider: fetch, and on a 2xx translate the answer into the client's
 * wire and send it. Returns true when the client has its answer (the one successful provider call
 * for this request); false when the provider refused or the call failed, and the next candidate
 * should be tried. Nothing is written to the client on a failure.
 */
async function tryRoute(
  route: Route,
  clientWire: 'openai' | 'anthropic',
  body: Json,
  res: ServerResponse,
): Promise<boolean> {
  const state = stateOf(route.probe.provider.id);
  const call = upstreamCallFor(route, clientWire, body);
  let upstream: Response;
  try {
    upstream = await fetch(call.url, {
      method: 'POST',
      headers: call.headers,
      body: JSON.stringify(call.payload),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch (err) {
    state.calls += 1;
    state.consecRefusals += 1;
    if (state.consecRefusals >= DEAD_AFTER_REFUSALS) state.dead = true;
    log(`${route.probe.provider.id}/${route.model.id} failed to answer: ${(err as Error).message}`);
    return false;
  }
  // A call that reached the provider with a model it serves spends its budget, whether it was
  // answered or refused; a 400 does not (the provider refuses it before counting), so it is not
  // counted here, and the model is never tried again.
  if (!upstream.ok) {
    const status = upstream.status;
    if (status === 400) {
      badModels.add(`${route.probe.provider.id}/${route.model.id}`);
      log(`${route.probe.provider.id}/${route.model.id} refused with HTTP 400: the request is wrong for it`);
    } else {
      state.calls += 1;
      if (status === 401) {
        state.dead = true;
        log(`${route.probe.provider.id} refused with HTTP 401: its key does not work`);
      } else {
        state.consecRefusals += 1;
        if (state.consecRefusals >= DEAD_AFTER_REFUSALS) state.dead = true;
        const text = await upstream.text().catch(() => '');
        log(`${route.probe.provider.id}/${route.model.id} refused with HTTP ${status}: ${text.slice(0, 200)}`);
      }
    }
    return false;
  }
  state.calls += 1;
  state.consecRefusals = 0;
  const stream = body.stream === true;
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : route.model.id;
  if (stream && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    for await (const frame of translateStream(clientWire, route, upstream.body, requestedModel)) res.write(frame);
    res.end();
    return true;
  }
  const text = await upstream.text();
  let answer: Json;
  try {
    const parsed: unknown = JSON.parse(text);
    answer = isRecord(parsed) ? parsed : {};
  } catch {
    log(`${route.probe.provider.id}/${route.model.id} answered a body that is not JSON`);
    return false;
  }
  sendJson(res, 200, translateJson(clientWire, route, answer));
  return true;
}

/**
 * Serve one client request: the candidates in rating order, skipping providers that look spent;
 * when every live candidate has failed, the spent ones once more, best first. A request the loop
 * cannot serve gets an error in the client's wire.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, requires: Requirement, res: ServerResponse): Promise<void> {
  const ordered = candidatesFor(requires, clientWire, body);
  if (ordered.length === 0) {
    return sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  for (const route of ordered) {
    if (stateOf(route.probe.provider.id).dead) continue;
    if (await tryRoute(route, clientWire, body, res)) return;
  }
  for (const route of ordered) {
    if (!stateOf(route.probe.provider.id).dead) continue;
    if (await tryRoute(route, clientWire, body, res)) return;
  }
  sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, requirementsFor(body), res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, requirementsFor(body), res);
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
