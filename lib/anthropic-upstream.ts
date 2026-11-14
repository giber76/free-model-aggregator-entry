// The Anthropic upstream wire: an OpenAI client request rewritten to the
// Messages API, and a Messages answer rewritten to Chat Completions.
//
// The starter's wire.ts already carries the reverse direction (an
// Anthropic client to an OpenAI provider). This file is its mirror: it
// lets an OpenAI client be served by an Anthropic-wire provider, and an
// Anthropic provider's answer reach an OpenAI client, in JSON and in
// server-sent events.
//
// What the env's Anthropic parser reads (and the client's echo checks):
//   body.system                     -> the system text
//   messages (role "user")          -> the user text (the last one wins)
//   messages[].content image blocks -> the image bytes (base64 or URL)
//   body.tools[].name               -> the tool names
//   body.max_tokens, body.stop_sequences -> max_tokens, the stops
// JSON mode has no Anthropic spelling: the env's parser always reports
// json_mode false for this wire, so an OpenAI request that carries
// response_format is never routed here (main.ts filters it out).
// The env's Anthropic answer is always a text answer with stop_reason
// end_turn and usage; it never returns a tool call.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

/** A fixed epoch second for every chat completion's `created`. */
const FIXED_CREATED = 1767225600;

/** The text of an OpenAI message content: a string, or a part array's text parts. */
function openAIText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that
 * names `model`: the system messages become the system prompt, image_url
 * parts become image blocks, tool calls become tool_use blocks, tool
 * messages become tool_result blocks, and tools and tool_choice are
 * rewritten. The output cap, temperature, top_p, stop sequences and
 * streaming pass through.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role !== 'assistant') continue;
    const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    for (const call of calls) {
      if (isRecord(call) && typeof call.id === 'string' && isRecord(call.function) && typeof call.function.name === 'string') {
        toolNames.set(call.id, call.function.name);
      }
    }
  }
  const messages: Json[] = [];
  const systemParts: string[] = [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = openAIText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = openAIText(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      const content = typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '');
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content,
          },
        ],
      });
      continue;
    }
    // A user message.
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const url = block.image_url.url;
        if (url.startsWith('data:')) {
          const m = /^data:([^;]+);base64,(.*)$/s.exec(url);
          if (m !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } });
        } else {
          blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop)) {
    const stops = body.stop.filter((s) => typeof s === 'string');
    if (stops.length > 0) out.stop_sequences = stops;
  }
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** A Messages answer as a Chat Completions object. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : '',
      type: 'function',
      function: {
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: text !== '' || toolCalls.length === 0 ? text : null };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(body.stop_reason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** One OpenAI chat chunk: the role chunk, a text chunk, the finish chunk or the usage chunk. */
function openAIChunk(model: string, delta: Json, finish: string | null, usage: Json | null): string {
  if (usage !== null) {
    return sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage });
  }
  return sseEvent(null, {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
}

/**
 * An Anthropic Messages SSE stream as OpenAI chat chunks: a role chunk, one
 * chunk per text delta, a finish chunk carrying the message_delta stop
 * reason, a usage chunk with the message_start input tokens and the
 * message_delta output tokens, then [DONE]. Tool deltas are not expected
 * from the trial's providers (their answers are text); one that arrives
 * is carried as a tool_calls delta.
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let sentRole = false;
  let sentFinish = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  let toolIndex = -1;

  const roleEvent = (): string => {
    if (sentRole) return '';
    sentRole = true;
    return openAIChunk(model, { role: 'assistant', content: '' }, null, null);
  };
  const finishEvents = (): string[] => {
    if (sentFinish) return [];
    sentFinish = true;
    return [
      openAIChunk(model, {}, anthropicStopToOpenAI(stopReason), null),
      openAIChunk(model, {}, null, { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens }),
      'data: [DONE]\n\n',
    ];
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    let eventName: string | null = null;
    let dataText: string | null = null;
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith('event:')) eventName = trimmed.slice(6).trim();
      else if (trimmed.startsWith('data:')) dataText = trimmed.slice(5).trim();
    }
    if (dataText === null || dataText === '') return out;
    let data: unknown;
    try {
      data = JSON.parse(dataText);
    } catch {
      return out;
    }
    if (!isRecord(data)) return out;
    if (eventName === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
      return out;
    }
    if (eventName === 'content_block_delta' && isRecord(data.delta)) {
      const delta = data.delta;
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        out.push(roleEvent());
        out.push(openAIChunk(model, { content: delta.text }, null, null));
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
        out.push(roleEvent());
        if (toolIndex === -1) toolIndex = 0;
        out.push(
          openAIChunk(model, { tool_calls: [{ index: toolIndex, function: { arguments: delta.partial_json } }] }, null, null),
        );
      }
      return out;
    }
    if (eventName === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
      return out;
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) {
          if (event !== '') yield event;
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const event of eventsOfFrame(buffer)) {
        if (event !== '') yield event;
      }
    }
  } finally {
    reader.releaseLock();
  }
  if (!sentRole) yield roleEvent();
  for (const event of finishEvents()) yield event;
}
