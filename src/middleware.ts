import { DBOS, StepConfig } from '@dbos-inc/dbos-sdk';
// Public signatures use ai's middleware aliases: ai is the single peer instance, so the types always
// match the consumer's wrap* calls. @ai-sdk/provider (dev-only) never appears in the published types —
// the AI SDK ecosystem exact-pins it, and duplicate copies don't unify.
import type { EmbeddingModelMiddleware, ImageModelMiddleware, LanguageModelMiddleware } from 'ai' with { 'resolution-mode': 'import' };
import type {
  EmbeddingModelV4,
  ImageModelV4,
  ImageModelV4Result,
  LanguageModelV4,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4Reasoning,
  LanguageModelV4ResponseMetadata,
  LanguageModelV4StreamPart,
  LanguageModelV4Text,
  LanguageModelV4Usage,
  SharedV4ProviderMetadata,
  SharedV4Warning,
} from '@ai-sdk/provider' with { 'resolution-mode': 'import' };
import { assertNotInTransaction, isInWorkflowFunction, withErrorClassification } from './internal';

// In-flight durable model calls per workflow; concurrent calls have a nondeterministic DBOS step order on replay, so we reject them.
const inflightModelCalls = new Map<string, number>();

function enterDurableModelCall(operation: 'generate' | 'stream' | 'embed'): string {
  const workflowID = DBOS.workflowID!;
  const inflight = inflightModelCalls.get(workflowID) ?? 0;
  if (inflight > 0) {
    // embedMany parallelizes its batches; maxParallelCalls: 1 serializes them deterministically. Other callers use child workflows.
    const remedy =
      operation === 'embed'
        ? 'pass maxParallelCalls: 1 to embedMany, or run each call in its own child workflow with DBOS.startWorkflow'
        : 'run each call in its own child workflow with DBOS.startWorkflow';
    throw new Error(
      `Concurrent durable model calls in workflow "${workflowID}" are not supported because their step order is nondeterministic on replay; ${remedy}.`,
    );
  }
  inflightModelCalls.set(workflowID, inflight + 1);
  return workflowID;
}

function exitDurableModelCall(workflowID: string): void {
  const inflight = (inflightModelCalls.get(workflowID) ?? 1) - 1;
  if (inflight > 0) {
    inflightModelCalls.set(workflowID, inflight);
  } else {
    inflightModelCalls.delete(workflowID);
  }
}

/** AI SDK language-model middleware that runs each model call as a durable, checkpointed DBOS step (replayed on recovery); outside a workflow it calls the model directly. */
export function durableCalls(options: StepConfig = {}): LanguageModelMiddleware {
  const stepConfig = withErrorClassification(options);
  return {
    specificationVersion: 'v4',

    wrapGenerate: async ({ doGenerate, model }) => {
      assertNotInTransaction('generate');
      if (!isInWorkflowFunction()) {
        return await doGenerate();
      }
      const workflowID = enterDurableModelCall('generate');
      try {
        return await DBOS.runStep(async () => encodeBinaryContent(await doGenerate()), {
          ...stepConfig,
          name: stepConfig.name ?? stepName(model, 'generate'),
        });
      } finally {
        exitDurableModelCall(workflowID);
      }
    },

    wrapStream: async ({ doStream, params, model }) => {
      assertNotInTransaction('stream');
      if (!isInWorkflowFunction()) {
        return await doStream();
      }
      const workflowID = enterDurableModelCall('stream');
      // An aborted consumer is done with this call, like a cancelled one: the AI SDK ends the stream gracefully on
      // abort, so a post-abort failure must checkpoint as a (partial) success or replay would fail where the live run didn't.
      const aborted = () => params.abortSignal?.aborted === true;

      let executed = false;
      let cancelled = false;
      let emittedLive = false;
      let controller!: ReadableStreamDefaultController<LanguageModelV4StreamPart>;
      let step!: Promise<LanguageModelV4GenerateResult>;
      const stream = new ReadableStream<LanguageModelV4StreamPart>({
        start(c) {
          controller = c;
        },
        // Await the step so an early cancel still blocks until the model result is checkpointed.
        async cancel() {
          cancelled = true;
          // Post-cancel failures normally checkpoint as a success; anything else (e.g. a failed checkpoint write) is only visible here.
          await step.catch((error: unknown) =>
            DBOS.logger.warn(`Durable model call step failed after consumer cancel: ${String(error)}`),
          );
        },
      });
      const emit = (part: LanguageModelV4StreamPart) => {
        if (!cancelled) {
          controller.enqueue(part);
          // stream-start and response-metadata merge idempotently downstream, so they alone don't preclude a retry.
          if (part.type !== 'stream-start' && part.type !== 'response-metadata') emittedLive = true;
        }
      };

      // Once any output part has streamed live, a retry would re-stream from scratch and duplicate output, so stop retrying.
      const streamStepConfig: StepConfig = {
        ...stepConfig,
        shouldRetry: async (error: unknown) =>
          !emittedLive && (stepConfig.shouldRetry ? await stepConfig.shouldRetry(error) : true),
      };

      try {
        step = DBOS.runStep(
          async () => {
            executed = true;
            const accumulator = new StreamAccumulator();
            // A timed-out attempt is abandoned by DBOS (its outcome is discarded) but keeps running; stop it so it can't emit alongside a retry.
            const timeoutSignal = DBOS.stepStatus?.timeoutSignal;
            let streamResult: Awaited<ReturnType<typeof doStream>> | undefined;
            let reader: ReadableStreamDefaultReader<LanguageModelV4StreamPart> | undefined;
            let sawFinish = false;
            const abandon = () => void reader?.cancel().catch(() => {});
            timeoutSignal?.addEventListener('abort', abandon, { once: true });
            try {
              streamResult = await doStream();
              reader = streamResult.stream.getReader();
              for (;;) {
                const { done, value: part } = await reader.read();
                if (timeoutSignal?.aborted) throw (timeoutSignal.reason ?? new Error('step attempt timed out'));
                if (done) break;
                if (part.type === 'error') {
                  // A cancelled or aborted consumer abandoned this call; don't let a late failure become the step outcome, or replay would fail where the live run succeeded.
                  if (cancelled || aborted()) break;
                  throw toStepError(part.error);
                }
                // Stream deltas live, but withhold 'finish' until the checkpoint is durable: the AI SDK runs tool calls
                // (and thus downstream durable steps) on 'finish', which must not checkpoint before this model step.
                if (part.type === 'finish') sawFinish = true;
                else emit(part);
                accumulator.add(part);
              }
              // No terminal part and no output: fail (retryably) like the AI SDK's NoOutputGeneratedError, instead of checkpointing a permanent empty success.
              if (!sawFinish && !accumulator.hasContent && !cancelled && !aborted()) {
                throw new Error('Model stream ended without a finish part or any output.');
              }
            } catch (error) {
              // Same rule for stream-level failures (doStream or a read rejecting) after a cancel or abort.
              if (!cancelled && !aborted()) throw error;
            } finally {
              timeoutSignal?.removeEventListener('abort', abandon);
              // Tear down the provider stream on early exits (error part, post-cancel break); a no-op after a clean drain.
              void reader?.cancel().catch(() => {});
            }
            return encodeBinaryContent(accumulator.result(streamResult?.request, streamResult?.response));
          },
          { ...streamStepConfig, name: stepConfig.name ?? stepName(model, 'stream') },
        );
      } catch (error) {
        // runStep can throw synchronously (e.g. a shutdown race); don't leak the guard entry.
        exitDurableModelCall(workflowID);
        throw error;
      }

      // Drive the returned stream from the settled step: a live run emits only the withheld 'finish' (deltas already
      // streamed); a recovered run synthesizes the whole stream from the checkpoint. Either way consumers finish
      // only after the result is durable.
      void step
        .then(
          (recorded) => {
            if (cancelled) return;
            if (executed) {
              emit({
                type: 'finish',
                finishReason: recorded.finishReason,
                usage: recorded.usage,
                providerMetadata: recorded.providerMetadata,
              });
            } else {
              for (const part of replayParts(recorded)) emit(part);
            }
            controller.close();
          },
          (error: unknown) => {
            if (!cancelled) controller.error(error);
          },
        )
        .finally(() => exitDurableModelCall(workflowID));

      return { stream };
    },
  };
}

/** AI SDK embedding-model middleware that runs each embedding call as a durable DBOS step, like {@link durableCalls}. */
export function durableEmbeddingCalls(options: StepConfig = {}): EmbeddingModelMiddleware {
  const stepConfig = withErrorClassification(options);
  return {
    specificationVersion: 'v4',
    wrapEmbed: async ({ doEmbed, model }) => {
      assertNotInTransaction('embed');
      if (!isInWorkflowFunction()) {
        return await doEmbed();
      }
      const workflowID = enterDurableModelCall('embed');
      try {
        return await DBOS.runStep(async () => doEmbed(), {
          ...stepConfig,
          name: stepConfig.name ?? stepName(model, 'embed'),
        });
      } finally {
        exitDurableModelCall(workflowID);
      }
    },
  };
}

/**
 * AI SDK image-model middleware that runs each image generation as a durable DBOS step, like {@link durableCalls}.
 * No concurrency guard: generateImage splits `n > maxImagesPerCall` into batches it dispatches synchronously (no
 * await before doGenerate), so their step order is deterministic on replay — unlike embedMany's parallel batches.
 */
export function durableImageCalls(options: StepConfig = {}): ImageModelMiddleware {
  const stepConfig = withErrorClassification(options);
  return {
    specificationVersion: 'v4',
    wrapGenerate: async ({ doGenerate, model }) => {
      assertNotInTransaction('generateImage');
      if (!isInWorkflowFunction()) {
        return await doGenerate();
      }
      return await DBOS.runStep(async () => encodeImageResult(await doGenerate()), {
        ...stepConfig,
        name: stepConfig.name ?? stepName(model, 'image'),
      });
    },
  };
}

/** Convert generated image bytes (Uint8Array) to base64 (spec-allowed) to keep checkpoints compact. */
function encodeImageResult(result: ImageModelV4Result): ImageModelV4Result {
  const images = result.images as (string | Uint8Array)[];
  if (images.every((image) => typeof image === 'string')) {
    return result;
  }
  return {
    ...result,
    images: images.map((image) => (typeof image === 'string' ? image : Buffer.from(image).toString('base64'))),
  };
}

function stepName(model: LanguageModelV4 | EmbeddingModelV4 | ImageModelV4, operation: string): string {
  return `${model.provider}.${model.modelId}.${operation}`;
}

/** Normalize an error-part payload to an Error, preserving the payload (JSON message, cause, isRetryable). */
function toStepError(error: unknown): Error {
  if (error instanceof Error) return error;
  let message: string;
  try {
    message = typeof error === 'string' ? error : (JSON.stringify(error) ?? String(error));
  } catch {
    message = String(error);
  }
  const result = new Error(message, { cause: error });
  const isRetryable = (error as { isRetryable?: unknown } | null | undefined)?.isRetryable;
  return isRetryable === undefined ? result : Object.assign(result, { isRetryable });
}

/** Convert generated-file bytes (Uint8Array) to base64 (spec-allowed) to keep checkpoints compact. */
function encodeBinaryContent(result: LanguageModelV4GenerateResult): LanguageModelV4GenerateResult {
  const content = result.content.map(encodeBinaryPart);
  return content.some((part, i) => part !== result.content[i]) ? { ...result, content } : result;
}

function encodeBinaryPart(part: LanguageModelV4Content): LanguageModelV4Content {
  if (
    (part.type === 'file' || part.type === 'reasoning-file') &&
    part.data.type === 'data' &&
    part.data.data instanceof Uint8Array
  ) {
    return { ...part, data: { type: 'data', data: Buffer.from(part.data.data).toString('base64') } };
  }
  return part;
}

/** Assembles stream parts into a LanguageModelV4GenerateResult so a stream can be checkpointed as one step result. */
class StreamAccumulator {
  private readonly content: LanguageModelV4Content[] = [];
  private finishReason: LanguageModelV4FinishReason = { unified: 'other', raw: undefined };
  private usage: LanguageModelV4Usage = {
    inputTokens: { total: undefined, noCache: undefined, cacheRead: undefined, cacheWrite: undefined },
    outputTokens: { total: undefined, text: undefined, reasoning: undefined },
  };
  private readonly warnings: SharedV4Warning[] = [];
  private providerMetadata?: SharedV4ProviderMetadata;
  private responseMetadata?: LanguageModelV4ResponseMetadata;
  // Text/reasoning content objects (also in content), appended at -start and mutated in place so content keeps arrival order.
  private readonly textBlocks = new Map<string, LanguageModelV4Text>();
  private readonly reasoningBlocks = new Map<string, LanguageModelV4Reasoning>();

  add(part: LanguageModelV4StreamPart): void {
    switch (part.type) {
      case 'stream-start':
        this.warnings.push(...part.warnings);
        break;
      case 'text-start': {
        const block: LanguageModelV4Text = { type: 'text', text: '', providerMetadata: part.providerMetadata };
        this.textBlocks.set(part.id, block);
        this.content.push(block);
        break;
      }
      case 'text-delta': {
        const block = this.getOrCreateText(part.id);
        block.text += part.delta;
        if (part.providerMetadata) block.providerMetadata = part.providerMetadata;
        break;
      }
      case 'text-end': {
        const block = this.textBlocks.get(part.id);
        if (block && part.providerMetadata) block.providerMetadata = part.providerMetadata;
        this.textBlocks.delete(part.id);
        break;
      }
      case 'reasoning-start': {
        const block: LanguageModelV4Reasoning = {
          type: 'reasoning',
          text: '',
          providerMetadata: part.providerMetadata,
        };
        this.reasoningBlocks.set(part.id, block);
        this.content.push(block);
        break;
      }
      case 'reasoning-delta': {
        const block = this.getOrCreateReasoning(part.id);
        block.text += part.delta;
        if (part.providerMetadata) block.providerMetadata = part.providerMetadata;
        break;
      }
      case 'reasoning-end': {
        const block = this.reasoningBlocks.get(part.id);
        if (block && part.providerMetadata) block.providerMetadata = part.providerMetadata;
        this.reasoningBlocks.delete(part.id);
        break;
      }
      case 'response-metadata':
        // Merge per-field like the AI SDK: later parts override only the fields they carry.
        this.responseMetadata = {
          id: part.id ?? this.responseMetadata?.id,
          timestamp: part.timestamp ?? this.responseMetadata?.timestamp,
          modelId: part.modelId ?? this.responseMetadata?.modelId,
        };
        break;
      case 'finish':
        this.finishReason = part.finishReason;
        this.usage = part.usage;
        this.providerMetadata = part.providerMetadata;
        break;
      case 'tool-input-start':
      case 'tool-input-delta':
      case 'tool-input-end':
      case 'raw':
      case 'error':
        // Transient parts; the tool-call part carries the complete input.
        break;
      default:
        // Complete content parts: tool-call, tool-result, tool-approval-request, file, reasoning-file, source, custom.
        this.content.push(part);
        break;
    }
  }

  get hasContent(): boolean {
    return this.content.length > 0;
  }

  // If a delta arrives with no preceding start, create the block in arrival position rather than dropping the text.
  private getOrCreateText(id: string): LanguageModelV4Text {
    let block = this.textBlocks.get(id);
    if (!block) {
      block = { type: 'text', text: '' };
      this.textBlocks.set(id, block);
      this.content.push(block);
    }
    return block;
  }

  private getOrCreateReasoning(id: string): LanguageModelV4Reasoning {
    let block = this.reasoningBlocks.get(id);
    if (!block) {
      block = { type: 'reasoning', text: '' };
      this.reasoningBlocks.set(id, block);
      this.content.push(block);
    }
    return block;
  }

  result(
    request?: { body?: unknown },
    response?: { headers?: Record<string, string> },
  ): LanguageModelV4GenerateResult {
    return {
      content: this.content,
      finishReason: this.finishReason,
      usage: this.usage,
      warnings: this.warnings,
      providerMetadata: this.providerMetadata,
      request,
      response: this.responseMetadata ? { ...this.responseMetadata, ...response } : response,
    };
  }
}

/** Synthesizes a stream from a checkpointed result on recovery; text/reasoning come back as one delta per block. */
function* replayParts(result: LanguageModelV4GenerateResult): Generator<LanguageModelV4StreamPart> {
  yield { type: 'stream-start', warnings: result.warnings ?? [] };
  if (result.response?.id !== undefined || result.response?.timestamp !== undefined || result.response?.modelId !== undefined) {
    yield {
      type: 'response-metadata',
      id: result.response.id,
      timestamp: result.response.timestamp,
      modelId: result.response.modelId,
    };
  }
  let blockIndex = 0;
  for (const part of result.content) {
    const id = `replay-${blockIndex++}`;
    if (part.type === 'text') {
      yield { type: 'text-start', id, providerMetadata: part.providerMetadata };
      if (part.text.length > 0) yield { type: 'text-delta', id, delta: part.text };
      yield { type: 'text-end', id, providerMetadata: part.providerMetadata };
    } else if (part.type === 'reasoning') {
      yield { type: 'reasoning-start', id, providerMetadata: part.providerMetadata };
      if (part.text.length > 0) yield { type: 'reasoning-delta', id, delta: part.text };
      yield { type: 'reasoning-end', id, providerMetadata: part.providerMetadata };
    } else if (part.type === 'tool-call') {
      // Re-synthesize the tool-input grammar: consumer tool callbacks (onInputStart/onInputAvailable) are keyed off tool-input-start.
      const { toolCallId, toolName, providerExecuted, dynamic } = part;
      yield { type: 'tool-input-start', id: toolCallId, toolName, providerExecuted, dynamic };
      if (part.input.length > 0) yield { type: 'tool-input-delta', id: toolCallId, delta: part.input };
      yield { type: 'tool-input-end', id: toolCallId };
      yield part;
    } else {
      yield part;
    }
  }
  yield {
    type: 'finish',
    finishReason: result.finishReason,
    usage: result.usage,
    providerMetadata: result.providerMetadata,
  };
}
