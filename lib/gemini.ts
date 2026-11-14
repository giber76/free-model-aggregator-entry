// The Gemini upstream wire: request and response translation, JSON and SSE.
//
// The trial's client calls this entry as OpenAI or Anthropic; a Gemini-wire
// provider speaks generateContent / streamGenerateContent. This file builds
// the Gemini request from either client wire and turns the Gemini answer
// back into the wire the client used. Everything here is a pure function of
// its arguments: no clock and no randomness, so a live pass and its replay
// give the same bytes.
//
// What the env's Gemini parser reads (and the client's echo checks):
//   systemInstruction.parts[].text  -> the system text
//   contents (role != "model")      -> the user text (the last one wins)
//   contents[].parts[].inlineData   -> the image bytes (base64)
//   tools[].functionDeclarations[]  -> the tool names
//   generationConfig.maxOutputTokens, .stopSequences, .responseMimeType
//     -> max_tokens, the stops, and JSON mode (application/json)
// The env's Gemini answer is always a text answer with finishReason STOP
// and usageMetadata; it never returns a tool call.

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

/** The text of an Anthropic message content: a string, or a block array's text blocks. */
function anthropicText(content: unknown): string {
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

/** One OpenAI image_url part as a Gemini image part: inlineData for a data URL, fileData for a URL. */
function geminiImagePart(url: string): Json | null {
  if (url.startsWith('data:')) {
    const m = /^data:([^;]+);base64,(.*)$/s.exec(url);
    if (m !== null) return { inlineData: { mimeType: m[1], data: m[2] } };
    return null;
  }
  return { fileData: { fileUri: url, mimeType: 'image/png' } };
}

/** One Anthropic image block as a Gemini image part. */
function geminiImageFromAnthropic(source: Json): Json | null {
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { fileData: { fileUri: source.url, mimeType: 'image/png' } };
  }
  return null;
}

/** The generationConfig of a Gemini request: the output cap, sampling, stops and JSON mode. */
function generationConfig(maxTokens: unknown, temperature: unknown, topP: unknown, stops: unknown, jsonMode: boolean): Json {
  const config: Json = {};
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  const stopList = Array.isArray(stops) ? stops.filter((s) => typeof s === 'string') : typeof stops === 'string' ? [stops] : [];
  if (stopList.length > 0) config.stopSequences = stopList;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/** OpenAI tools as Gemini functionDeclarations. */
function geminiToolsFromOpenAI(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as Gemini functionDeclarations. */
function geminiToolsFromAnthropic(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request:
 * the system prompt becomes systemInstruction, user and assistant messages
 * become contents, image_url parts become inlineData/fileData, tool calls
 * become functionCall parts, tool results become functionResponse parts,
 * and tools become functionDeclarations.
 */
export function openAIToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  let systemText = '';
  const toolNames = new Map<string, string>();
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw) || raw.role !== 'assistant') continue;
    const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
    for (const call of calls) {
      if (isRecord(call) && typeof call.id === 'string' && isRecord(call.function) && typeof call.function.name === 'string') {
        toolNames.set(call.id, call.function.name);
      }
    }
  }
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = openAIText(raw.content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = openAIText(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      const name = toolNames.get(typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '') ?? '';
      const response = typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '');
      contents.push({ role: 'user', parts: [{ functionResponse: { name, response: { result: response } } }] });
      continue;
    }
    // A user message.
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const part = geminiImagePart(block.image_url.url);
          if (part !== null) parts.push(part);
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { model, contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const declarations = geminiToolsFromOpenAI(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  out.generationConfig = generationConfig(
    body.max_tokens ?? body.max_completion_tokens,
    body.temperature,
    body.top_p,
    body.stop,
    format === 'json_object' || format === 'json_schema',
  );
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request: the
 * system prompt becomes systemInstruction, image blocks become inlineData,
 * tool_use blocks become functionCall parts, tool_result blocks become
 * functionResponse parts, and tools become functionDeclarations.
 */
export function anthropicToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const systemText = anthropicText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') contents.push({ role: assistant ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        const part = geminiImageFromAnthropic(block.source);
        if (part !== null) parts.push(part);
      } else if (assistant && block.type === 'tool_use') {
        parts.push({
          functionCall: {
            name: typeof block.name === 'string' ? block.name : '',
            args: isRecord(block.input) ? block.input : {},
          },
        });
      } else if (!assistant && block.type === 'tool_result') {
        const text = typeof block.content === 'string' ? block.content : anthropicText(block.content);
        parts.push({
          functionResponse: {
            name: '',
            response: { result: text !== '' ? text : JSON.stringify(block.content ?? '') },
          },
        });
      }
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  const out: Json = { model, contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const declarations = geminiToolsFromAnthropic(body.tools);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  out.generationConfig = generationConfig(body.max_tokens, body.temperature, body.top_p, body.stop_sequences, false);
  return out;
}

/** A Gemini finishReason as an OpenAI finish_reason. */
export function geminiFinishToOpenAI(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
export function geminiFinishToAnthropic(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The text and usage of a Gemini generateContent answer. */
function geminiAnswerOf(body: Json): { text: string; finish: string; input: number; output: number } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    finish: typeof cand.finishReason === 'string' ? cand.finishReason : 'STOP',
    input: numberOrZero(usage.promptTokenCount),
    output: numberOrZero(usage.candidatesTokenCount),
  };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const answer = geminiAnswerOf(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: geminiFinishToOpenAI(answer.finish),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const answer = geminiAnswerOf(body);
  return {
    id: `msg_${model}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: geminiFinishToAnthropic(answer.finish),
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
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
 * A Gemini streamGenerateContent SSE stream as OpenAI chat chunks: a role
 * chunk, one chunk per text piece, a finish chunk, a usage chunk carrying
 * the final frame's usageMetadata, then [DONE].
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let sentRole = false;
  let sentFinish = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let finish = 'STOP';

  const roleEvent = (): string => {
    if (sentRole) return '';
    sentRole = true;
    return openAIChunk(model, { role: 'assistant', content: '' }, null, null);
  };
  const finishEvents = (): string[] => {
    if (sentFinish) return [];
    sentFinish = true;
    return [
      openAIChunk(model, {}, geminiFinishToOpenAI(finish), null),
      openAIChunk(model, {}, null, { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens }),
      'data: [DONE]\n\n',
    ];
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
      const cand = isRecord(candidates[0]) ? candidates[0] : null;
      if (cand !== null) {
        const content = isRecord(cand.content) ? cand.content : {};
        const parts = Array.isArray(content.parts) ? content.parts : [];
        for (const part of parts) {
          if (isRecord(part) && typeof part.text === 'string' && part.text !== '') {
            out.push(roleEvent());
            out.push(openAIChunk(model, { content: part.text }, null, null));
          }
        }
        if (typeof cand.finishReason === 'string') finish = cand.finishReason;
      }
      if (isRecord(chunk.usageMetadata)) {
        inputTokens = numberOrZero(chunk.usageMetadata.promptTokenCount);
        outputTokens = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
      }
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

/**
 * A Gemini streamGenerateContent SSE stream as the Anthropic event
 * sequence: message_start, one text content block, message_delta with the
 * stop reason and the final frame's output tokens, then message_stop.
 * The usage check on an Anthropic stream reads only the output tokens,
 * so message_start carries input_tokens 0.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let blockOpen = false;
  let finished = false;
  let outputTokens = 0;
  let finish = 'STOP';

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }),
      sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    ];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (blockOpen) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
      blockOpen = false;
    }
    out.push(
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: geminiFinishToAnthropic(finish), stop_sequence: null },
        usage: { output_tokens: outputTokens },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
      const cand = isRecord(candidates[0]) ? candidates[0] : null;
      if (cand !== null) {
        const content = isRecord(cand.content) ? cand.content : {};
        const parts = Array.isArray(content.parts) ? content.parts : [];
        for (const part of parts) {
          if (isRecord(part) && typeof part.text === 'string' && part.text !== '') {
            out.push(...startEvents());
            blockOpen = true;
            out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: part.text } }));
          }
        }
        if (typeof cand.finishReason === 'string') finish = cand.finishReason;
      }
      if (isRecord(chunk.usageMetadata)) {
        outputTokens = numberOrZero(chunk.usageMetadata.candidatesTokenCount);
      }
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
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const event of eventsOfFrame(buffer)) yield event;
    }
  } finally {
    reader.releaseLock();
  }
  if (!started) {
    for (const event of startEvents()) yield event;
    blockOpen = true;
  }
  for (const event of finishEvents()) yield event;
}
