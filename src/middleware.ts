import { DBOS, StepConfig } from '@dbos-inc/dbos-sdk';
import type {
  EmbeddingModelV4,
  EmbeddingModelV4Middleware,
  LanguageModelV4,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4Middleware,
  LanguageModelV4ResponseMetadata,
  LanguageModelV4StreamPart,
  LanguageModelV4Usage,
  SharedV4ProviderMetadata,
  SharedV4Warning,
} from '@ai-sdk/provider' with { 'resolution-mode': 'import' };

/**
 * Options for {@link durableCalls} and {@link durableEmbeddingCalls}.
 *
 * Extends the DBOS {@link StepConfig}, so all step options
 * (retriesAllowed, maxAttempts, intervalSeconds, backoffRate, shouldRetry, timeoutMS, name)
 * apply to the wrapped model calls.
 */
export interface DurableCallsOptions extends StepConfig {
  /**
   * If set, every raw stream part produced by streaming calls (`doStream`) is also
   * written to the DBOS workflow stream with this key, so tokens can be consumed
   * from outside the workflow with `DBOS.readStream(workflowID, streamKey)`
   * (e.g., an HTTP handler streaming to a browser while the workflow runs elsewhere).
   *
   * Note that stream writes from steps are not checkpointed: if the step is retried,
   * parts from failed attempts remain in the stream (at-least-once delivery).
   */
  streamKey?: string;
}

function isInWorkflowFunction(): boolean {
  // True only in workflow code proper: not in a step (the enclosing step provides
  // durability), not in a transaction, and not outside DBOS entirely.
  return DBOS.isInWorkflow();
}

function assertNotInTransaction(operation: string) {
  if (DBOS.isInTransaction()) {
    throw new Error(
      `Cannot call ${operation} inside a DBOS transaction. AI model calls perform network I/O; move this call to workflow or step code.`,
    );
  }
}

/**
 * Returns AI SDK language-model middleware that makes model calls durable by
 * running them as DBOS steps. Once a call succeeds, its result is checkpointed
 * in the DBOS system database; if the workflow is interrupted and recovers, the
 * checkpointed result is used instead of calling the model again.
 *
 * Usage:
 * ```ts
 * const model = wrapLanguageModel({
 *   model: openai('gpt-5'),
 *   middleware: durableCalls({ retriesAllowed: true, maxAttempts: 5 }),
 * });
 * ```
 *
 * Used outside a DBOS workflow (or inside another step), the middleware calls
 * the model directly without checkpointing, so the same wrapped model works anywhere.
 */
export function durableCalls(options: DurableCallsOptions = {}): LanguageModelV4Middleware {
  const { streamKey, ...stepConfig } = options;
  return {
    specificationVersion: 'v4',

    wrapGenerate: async ({ doGenerate, model }) => {
      assertNotInTransaction('generate');
      if (!isInWorkflowFunction()) {
        return await doGenerate();
      }
      return await DBOS.runStep(async () => encodeBinaryContent(await doGenerate()), {
        ...stepConfig,
        name: stepConfig.name ?? stepName(model, 'generate'),
      });
    },

    wrapStream: async ({ doStream, model }) => {
      assertNotInTransaction('stream');
      if (!isInWorkflowFunction()) {
        return await doStream();
      }

      // With retries enabled, a failed attempt may have already produced parts, so
      // live pass-through would deliver output from multiple attempts. Instead,
      // buffer and emit the (synthesized) parts only after the step succeeds.
      const buffered = stepConfig.retriesAllowed === true;

      let executed = false;
      let cancelled = false;
      let controller!: ReadableStreamDefaultController<LanguageModelV4StreamPart>;
      const stream = new ReadableStream<LanguageModelV4StreamPart>({
        start(c) {
          controller = c;
        },
        cancel() {
          cancelled = true;
        },
      });
      const emit = (part: LanguageModelV4StreamPart) => {
        if (!cancelled) controller.enqueue(part);
      };

      const step = DBOS.runStep(
        async () => {
          executed = true;
          const streamResult = await doStream();
          const accumulator = new StreamAccumulator();
          const reader = streamResult.stream.getReader();
          for (;;) {
            const { done, value: part } = await reader.read();
            if (done) break;
            if (part.type === 'error') {
              throw part.error instanceof Error ? part.error : new Error(String(part.error));
            }
            if (!buffered) emit(part);
            if (streamKey !== undefined) await DBOS.writeStream(streamKey, part);
            accumulator.add(part);
          }
          return encodeBinaryContent(accumulator.result(streamResult.request, streamResult.response));
        },
        { ...stepConfig, name: stepConfig.name ?? stepName(model, 'stream') },
      );

      // Don't await the step before returning: parts must flow to the consumer
      // while the model call runs. The stream closes only after the step is
      // checkpointed, so consumers finish strictly after the result is durable.
      // If the step was replayed from a checkpoint (or ran buffered), the closure
      // above never emitted, so synthesize the parts from the recorded result.
      void step.then(
        (recorded) => {
          if ((buffered || !executed) && !cancelled) {
            for (const part of replayParts(recorded)) emit(part);
          }
          if (!cancelled) controller.close();
        },
        (error: unknown) => {
          if (!cancelled) controller.error(error);
        },
      );

      return { stream };
    },
  };
}

/**
 * Returns AI SDK embedding-model middleware that makes embedding calls durable
 * by running them as DBOS steps, analogous to {@link durableCalls}.
 */
export function durableEmbeddingCalls(options: StepConfig = {}): EmbeddingModelV4Middleware {
  return {
    specificationVersion: 'v4',
    wrapEmbed: async ({ doEmbed, model }) => {
      assertNotInTransaction('embed');
      if (!isInWorkflowFunction()) {
        return await doEmbed();
      }
      return await DBOS.runStep(async () => doEmbed(), {
        ...options,
        name: options.name ?? stepName(model, 'embed'),
      });
    },
  };
}

function stepName(model: LanguageModelV4 | EmbeddingModelV4, operation: string): string {
  return `${model.provider}.${model.modelId}.${operation}`;
}

/**
 * DBOS serializes step results with superjson, which round-trips Date, URL, Map,
 * Set, and Buffer — but not raw Uint8Array. Generated files may carry raw bytes,
 * so convert them to base64 strings, which the provider spec explicitly allows.
 */
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

/**
 * Assembles the parts of a model stream into a `LanguageModelV4GenerateResult`
 * so the completed stream can be checkpointed as a single step result.
 */
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
  private readonly textBlocks = new Map<string, { text: string; providerMetadata?: SharedV4ProviderMetadata }>();
  private readonly reasoningBlocks = new Map<string, { text: string; providerMetadata?: SharedV4ProviderMetadata }>();

  add(part: LanguageModelV4StreamPart): void {
    switch (part.type) {
      case 'stream-start':
        this.warnings.push(...part.warnings);
        break;
      case 'text-start':
        this.textBlocks.set(part.id, { text: '' });
        break;
      case 'text-delta': {
        const block = this.textBlocks.get(part.id) ?? { text: '' };
        block.text += part.delta;
        this.textBlocks.set(part.id, block);
        break;
      }
      case 'text-end': {
        const block = this.textBlocks.get(part.id) ?? { text: '' };
        this.content.push({ type: 'text', text: block.text, providerMetadata: part.providerMetadata });
        this.textBlocks.delete(part.id);
        break;
      }
      case 'reasoning-start':
        this.reasoningBlocks.set(part.id, { text: '' });
        break;
      case 'reasoning-delta': {
        const block = this.reasoningBlocks.get(part.id) ?? { text: '' };
        block.text += part.delta;
        this.reasoningBlocks.set(part.id, block);
        break;
      }
      case 'reasoning-end': {
        const block = this.reasoningBlocks.get(part.id) ?? { text: '' };
        this.content.push({ type: 'reasoning', text: block.text, providerMetadata: part.providerMetadata });
        this.reasoningBlocks.delete(part.id);
        break;
      }
      case 'response-metadata':
        this.responseMetadata = { id: part.id, timestamp: part.timestamp, modelId: part.modelId };
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
        // Complete content parts: tool-call, tool-result, tool-approval-request,
        // file, reasoning-file, source, custom.
        this.content.push(part);
        break;
    }
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

/**
 * Synthesizes a stream of parts from a checkpointed generate result, for replaying
 * a recorded model stream during workflow recovery. Text and reasoning come back
 * as a single delta per block; the real-time token stream already happened during
 * the original execution.
 */
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
      yield { type: 'text-start', id };
      if (part.text.length > 0) yield { type: 'text-delta', id, delta: part.text };
      yield { type: 'text-end', id, providerMetadata: part.providerMetadata };
    } else if (part.type === 'reasoning') {
      yield { type: 'reasoning-start', id };
      if (part.text.length > 0) yield { type: 'reasoning-delta', id, delta: part.text };
      yield { type: 'reasoning-end', id, providerMetadata: part.providerMetadata };
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
