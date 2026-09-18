import { randomUUID } from 'node:crypto';
import { DBOS, Error as DBOSErrors, StatusString } from '@dbos-inc/dbos-sdk';
import type { UIMessageChunk } from 'ai' with { 'resolution-mode': 'import' };
import type { LanguageModelV4FinishReason, LanguageModelV4StreamPart } from '@ai-sdk/provider' with { 'resolution-mode': 'import' };

/** Durable stream config: the DBOS stream key, or the key plus batching limits for the model step's writes. */
export type DurableStreamOptions = string | { key: string; maxBatchParts?: number; maxBatchDelayMs?: number };

/** One DBOS stream value; the reader turns these into AI SDK UI message chunks. */
export type DurableStreamRecord =
  | { kind: 'model'; step: number; attempt: string; parts: LanguageModelV4StreamPart[] }
  | { kind: 'model-end'; step: number; attempt: string; finishReason?: LanguageModelV4FinishReason; aborted?: true }
  | { kind: 'tool'; step: number; attempt: number; toolCallId: string; output?: unknown; errorText?: string }
  | { kind: 'ui'; step?: number; attempt?: number; chunks: UIMessageChunk[] }
  | { kind: 'end'; finishReason: string };

interface ResolvedDurableStream {
  key: string;
  maxBatchParts: number;
  maxBatchDelayMs: number;
}

export function resolveDurableStream(options: DurableStreamOptions | undefined): ResolvedDurableStream | undefined {
  if (options === undefined) return undefined;
  const config = typeof options === 'string' ? { key: options } : options;
  return { key: config.key, maxBatchParts: config.maxBatchParts ?? 20, maxBatchDelayMs: config.maxBatchDelayMs ?? 25 };
}

function stepInfo(): { step: number; attempt: number } {
  return { step: DBOS.stepID ?? -1, attempt: DBOS.stepStatus?.currentAttempt ?? 1 };
}

// Parts the reader can render; framing, metadata and terminal parts are recorded elsewhere or not at all.
function isContentPart(part: LanguageModelV4StreamPart): boolean {
  switch (part.type) {
    case 'stream-start':
    case 'response-metadata':
    case 'finish':
    case 'error':
    case 'raw':
      return false;
    default:
      return true;
  }
}

// Bytes become base64 so a file part stays compact in Postgres.
function encodePart(part: LanguageModelV4StreamPart): LanguageModelV4StreamPart {
  if (part.type === 'file' && part.data.type === 'data' && part.data.data instanceof Uint8Array) {
    return { ...part, data: { type: 'data', data: Buffer.from(part.data.data).toString('base64') } };
  }
  return part;
}

// A transient write error (the SDK already retries offset conflicts) gets a few attempts before it fails the model call.
async function writeWithRetry(key: string, record: DurableStreamRecord): Promise<void> {
  for (let attempt = 1; ; attempt++) {
    try {
      return await DBOS.writeStream(key, record);
    } catch (error) {
      if (attempt === 3) throw error;
      await new Promise((resolve) => setTimeout(resolve, 100 * attempt));
    }
  }
}

/** Batches a live model step's parts into step-scope stream writes; nothing is written on replay because the step body does not run. */
export class ModelStreamWriter {
  private pending: LanguageModelV4StreamPart[] = [];
  private chain: Promise<void> = Promise.resolve();
  private failure: unknown;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private readonly step = DBOS.stepID ?? -1;
  // Unique per execution of the step, so a recovered run's re-execution is distinguishable from the crashed one.
  private readonly attempt = randomUUID();

  constructor(private readonly config: ResolvedDurableStream) {}

  push(part: LanguageModelV4StreamPart): void {
    if (!isContentPart(part)) return;
    this.pending.push(encodePart(part));
    if (this.pending.length >= this.config.maxBatchParts) this.flush();
    else this.timer ??= setTimeout(() => this.flush(), this.config.maxBatchDelayMs);
  }

  /** Flushes, records how the call ended, and resolves once every write is durable; a write that failed after retries fails the call here. */
  async end(outcome: { finishReason: LanguageModelV4FinishReason } | { aborted: true }): Promise<void> {
    this.flush();
    this.write({ kind: 'model-end', step: this.step, attempt: this.attempt, ...outcome });
    await this.chain;
    if (this.failure !== undefined) throw this.failure;
  }

  /** After a failure: flush what streamed so the record matches what the consumer saw; the stream's end then comes from the workflow's status. */
  async abandon(): Promise<void> {
    this.flush();
    await this.chain;
  }

  private flush(): void {
    if (this.timer !== undefined) clearTimeout(this.timer);
    this.timer = undefined;
    if (this.pending.length === 0) return;
    const parts = this.pending;
    this.pending = [];
    this.write({ kind: 'model', step: this.step, attempt: this.attempt, parts });
  }

  // Every link has a handler, so a rejection can never sit unobserved; after one failure later writes are skipped.
  private write(record: DurableStreamRecord): void {
    this.chain = this.chain
      .then(() => (this.failure === undefined ? writeWithRetry(this.config.key, record) : undefined))
      .catch((error: unknown) => {
        this.failure ??= error;
      });
  }
}

/** Records a tool call's outcome from inside its step. */
export function writeToolRecord(key: string, toolCallId: string, outcome: { output: unknown } | { errorText: string }): Promise<void> {
  const record: DurableStreamRecord = { kind: 'tool', ...stepInfo(), toolCallId, ...outcome };
  return DBOS.writeStream(key, record);
}

/**
 * Appends UI message chunks to a durable stream. From a step the write is cheap and at-least-once, so give data parts
 * stable ids; from workflow code it is a checkpointed step, so the number of calls must be deterministic.
 */
export function writeDurableStream(key: string, chunks: UIMessageChunk[]): Promise<void> {
  const status = DBOS.stepStatus;
  const record: DurableStreamRecord = { kind: 'ui', step: status?.stepID, attempt: status?.currentAttempt, chunks };
  return DBOS.writeStream(key, record);
}

/** Marks the end of the turn explicitly and closes the stream; without it the reader infers the end from the last model call or the workflow's status. */
export async function closeDurableStream(key: string, finishReason = 'stop'): Promise<void> {
  const record: DurableStreamRecord = { kind: 'end', finishReason };
  await DBOS.writeStream(key, record);
  await DBOS.closeStream(key);
}

/** What the reader needs from DBOS: the `DBOS` class in a launched process, or a `DBOSClient` anywhere else. */
export interface DurableStreamSource {
  readStream<T>(workflowID: string, key: string, options?: { offset?: number }): AsyncGenerator<T, void, unknown>;
  readStreamOffset<T>(workflowID: string, key: string, offset: number, options?: { timeoutSeconds?: number }): Promise<T>;
  retrieveWorkflow(workflowID: string): { getStatus(): Promise<{ status: string; error?: unknown } | null> };
}

export interface ReadDurableStreamOptions {
  workflowID: string;
  key: string;
  /** Id for the `start` chunk; omitted on a resume (`offset` > 0). */
  messageId?: string;
  /** Number of records already consumed, from the last `data-dbos-offset` chunk. */
  offset?: number;
  /** Defaults to `DBOS`; pass a `DBOSClient` to read from a process that has not launched DBOS. */
  client?: DurableStreamSource;
  /** Emit reasoning parts (default true, as in the AI SDK). */
  sendReasoning?: boolean;
  /** Emit source parts (default false, as in the AI SDK). */
  sendSources?: boolean;
}

/** Reads a durable stream as AI SDK UI message chunks, live or after the fact, resuming from `offset`. */
export function readDurableStream(options: ReadDurableStreamOptions): ReadableStream<UIMessageChunk> {
  const iterator = uiChunks(options);
  return new ReadableStream<UIMessageChunk>({
    async pull(controller) {
      const { value, done } = await iterator.next();
      if (done) controller.close();
      else controller.enqueue(value);
    },
    async cancel() {
      await iterator.return(undefined);
    },
  });
}

async function* uiChunks(options: ReadDurableStreamOptions): AsyncGenerator<UIMessageChunk> {
  const { workflowID, key, sendReasoning = true, sendSources = false } = options;
  const client: DurableStreamSource = options.client ?? DBOS;
  const state: ReaderState = {
    offset: options.offset ?? 0,
    resumed: (options.offset ?? 0) > 0,
    openParts: new Map(),
    ended: false,
  };
  if (state.offset === 0) yield { type: 'start', messageId: options.messageId };
  const emit = (record: DurableStreamRecord) => emitRecord(state, record, { sendReasoning, sendSources });

  // Phase 1: everything already stored, one value per query until an offset is empty; a superseded attempt is skipped whole.
  const history: DurableStreamRecord[] = [];
  for (;;) {
    try {
      history.push(await client.readStreamOffset<DurableStreamRecord>(workflowID, key, state.offset + history.length, { timeoutSeconds: 0 }));
    } catch (error) {
      if (!DBOSErrors.isStreamTimeoutError(error)) throw error;
      break;
    }
  }
  const finalAttempt = new Map<number, string>();
  for (const record of history) {
    if (record.kind === 'model' || record.kind === 'model-end') finalAttempt.set(record.step, record.attempt);
  }
  for (const record of history) {
    const stale = (record.kind === 'model' || record.kind === 'model-end') && finalAttempt.get(record.step) !== record.attempt;
    yield* stale ? skipRecord(state) : emit(record);
    if (state.ended) return;
  }

  // Phase 2: live; a re-executed step shows up as a new attempt and is handed off in place.
  for await (const record of client.readStream<DurableStreamRecord>(workflowID, key, { offset: state.offset })) {
    yield* emit(record);
    if (state.ended) return;
  }

  // No end record: the workflow's status decides how the turn ended (a stream closed while it still runs counts as finished).
  const status = (await client.retrieveWorkflow(workflowID).getStatus())?.status;
  yield* closeStep(state);
  if (status === StatusString.CANCELLED) {
    yield { type: 'abort' };
  } else if (status === undefined || status === StatusString.SUCCESS || status === StatusString.PENDING || status === StatusString.ENQUEUED) {
    // The AI SDK's finish schema has no 'unknown'; a turn with no model call ends with the reason omitted.
    yield state.finishReason === undefined ? { type: 'finish' } : { type: 'finish', finishReason: state.finishReason as UIFinishReason };
  } else {
    const error = (await client.retrieveWorkflow(workflowID).getStatus())?.error;
    yield { type: 'error', errorText: error instanceof Error ? error.message : String(error ?? 'The workflow ended before the response completed.') };
    yield { type: 'finish', finishReason: 'error' };
  }
}

interface ReaderState {
  offset: number;
  resumed: boolean;
  openStep?: number;
  openAttempt?: string;
  // Text/reasoning parts of the open attempt that have started but not ended, by UI part id.
  openParts: Map<string, 'text' | 'reasoning'>;
  finishReason?: string;
  ended: boolean;
}

function offsetChunk(state: ReaderState): UIMessageChunk {
  return { type: 'data-dbos-offset', data: { offset: state.offset }, transient: true } as UIMessageChunk;
}

function* skipRecord(state: ReaderState): Generator<UIMessageChunk> {
  state.offset += 1;
  yield offsetChunk(state);
}

function* closeStep(state: ReaderState): Generator<UIMessageChunk> {
  if (state.openStep !== undefined) yield { type: 'finish-step' };
  state.openStep = undefined;
  state.openAttempt = undefined;
  state.openParts.clear();
}

// A live re-execution of the open step: end the stale attempt's parts and tell the client which ones to discard.
function* supersede(state: ReaderState, attempt: string): Generator<UIMessageChunk> {
  for (const [id, kind] of state.openParts) yield { type: kind === 'text' ? 'text-end' : 'reasoning-end', id };
  yield {
    type: 'data-dbos-superseded',
    data: { attempt: state.openAttempt, parts: [...state.openParts.keys()] },
    transient: true,
  } as UIMessageChunk;
  state.openParts.clear();
  state.openAttempt = attempt;
}

function* emitRecord(
  state: ReaderState,
  record: DurableStreamRecord,
  filter: { sendReasoning: boolean; sendSources: boolean },
): Generator<UIMessageChunk> {
  state.offset += 1;
  switch (record.kind) {
    case 'model': {
      if (state.openStep !== record.step) {
        yield* closeStep(state);
        if (!state.resumed) yield { type: 'start-step' };
        state.resumed = false;
        state.openStep = record.step;
        state.openAttempt = record.attempt;
      } else if (state.openAttempt !== record.attempt) {
        yield* supersede(state, record.attempt);
      }
      for (const part of record.parts) {
        if (!filter.sendReasoning && part.type.startsWith('reasoning-')) continue;
        if (!filter.sendSources && part.type === 'source') continue;
        const chunk = toUIChunk(part, record.attempt);
        if (!chunk) continue;
        if (chunk.type === 'text-start' || chunk.type === 'reasoning-start') state.openParts.set(chunk.id, chunk.type === 'text-start' ? 'text' : 'reasoning');
        if (chunk.type === 'text-end' || chunk.type === 'reasoning-end') state.openParts.delete(chunk.id);
        yield chunk;
      }
      break;
    }
    case 'model-end':
      // The stream outlives the call: the workflow may run more calls, so only its end (or closeDurableStream) ends the turn.
      if (record.attempt === state.openAttempt) state.finishReason = record.aborted ? 'other' : record.finishReason?.unified;
      break;
    case 'tool':
      yield record.errorText !== undefined
        ? { type: 'tool-output-error', toolCallId: record.toolCallId, errorText: record.errorText }
        : { type: 'tool-output-available', toolCallId: record.toolCallId, output: record.output };
      break;
    case 'ui':
      yield* record.chunks;
      break;
    case 'end':
      // The terminal chunk comes last, so the offset goes out first.
      yield offsetChunk(state);
      yield* closeStep(state);
      yield { type: 'finish', finishReason: record.finishReason as UIFinishReason };
      state.ended = true;
      return;
  }
  yield offsetChunk(state);
}

type UIFinishReason = Extract<UIMessageChunk, { type: 'finish' }>['finishReason'];

function parseInput(input: string): unknown {
  try {
    return JSON.parse(input);
  } catch {
    return input;
  }
}

// Text and reasoning ids are only unique within one model call; the attempt id keeps calls, and re-executions, apart in one message.
function toUIChunk(part: LanguageModelV4StreamPart, attempt: string): UIMessageChunk | undefined {
  const id = 'id' in part ? `${attempt}:${part.id}` : '';
  switch (part.type) {
    case 'text-start':
      return { type: 'text-start', id, providerMetadata: part.providerMetadata };
    case 'text-delta':
      return { type: 'text-delta', id, delta: part.delta, providerMetadata: part.providerMetadata };
    case 'text-end':
      return { type: 'text-end', id, providerMetadata: part.providerMetadata };
    case 'reasoning-start':
      return { type: 'reasoning-start', id, providerMetadata: part.providerMetadata };
    case 'reasoning-delta':
      return { type: 'reasoning-delta', id, delta: part.delta, providerMetadata: part.providerMetadata };
    case 'reasoning-end':
      return { type: 'reasoning-end', id, providerMetadata: part.providerMetadata };
    case 'tool-input-start':
      return {
        type: 'tool-input-start',
        toolCallId: part.id,
        toolName: part.toolName,
        providerExecuted: part.providerExecuted,
        dynamic: part.dynamic,
        title: part.title,
        providerMetadata: part.providerMetadata,
      };
    case 'tool-input-delta':
      return { type: 'tool-input-delta', toolCallId: part.id, inputTextDelta: part.delta };
    case 'tool-call':
      return {
        type: 'tool-input-available',
        toolCallId: part.toolCallId,
        toolName: part.toolName,
        input: parseInput(part.input),
        providerExecuted: part.providerExecuted,
        dynamic: part.dynamic,
        providerMetadata: part.providerMetadata,
      };
    case 'tool-result':
      return part.isError
        ? { type: 'tool-output-error', toolCallId: part.toolCallId, errorText: JSON.stringify(part.result), providerExecuted: true, dynamic: part.dynamic }
        : { type: 'tool-output-available', toolCallId: part.toolCallId, output: part.result, providerExecuted: true, dynamic: part.dynamic, preliminary: part.preliminary };
    case 'source':
      return part.sourceType === 'url'
        ? { type: 'source-url', sourceId: part.id, url: part.url, title: part.title, providerMetadata: part.providerMetadata }
        : { type: 'source-document', sourceId: part.id, mediaType: part.mediaType, title: part.title, filename: part.filename, providerMetadata: part.providerMetadata };
    case 'file': {
      const url = part.data.type === 'url' ? String(part.data.url) : part.data.type === 'data' ? `data:${part.mediaType};base64,${String(part.data.data)}` : undefined;
      return url === undefined ? undefined : { type: 'file', url, mediaType: part.mediaType, providerMetadata: part.providerMetadata };
    }
    default:
      return undefined;
  }
}
