import { DBOS, StepConfig } from '@dbos-inc/dbos-sdk';
import type {
  EmbeddingModelV4,
  EmbeddingModelV4Middleware,
  LanguageModelV4,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4Middleware,
  LanguageModelV4Reasoning,
  LanguageModelV4ResponseMetadata,
  LanguageModelV4StreamPart,
  LanguageModelV4Text,
  LanguageModelV4Usage,
  SharedV4ProviderMetadata,
  SharedV4Warning,
} from '@ai-sdk/provider' with { 'resolution-mode': 'import' };

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

// Count of durable model-call steps currently executing within each workflow,
// keyed by workflow ID. DBOS derives a step's replay identity from a synchronous
// per-workflow counter captured in the order steps are reached — but the AI SDK
// can issue model calls concurrently (parallel generateText/streamText, embedMany
// over more values than the model's per-call limit, or parallel tool sub-agents),
// and it reaches them in a nondeterministic order. On recovery that order can
// differ, binding a checkpoint to the wrong call: silently for same-model calls
// (identical step names), or as a replay crash for different ones. The middleware
// cannot make the AI SDK deterministic, so it refuses to start a second concurrent
// durable call rather than risk corruption; run each in its own child workflow.
const inflightModelCalls = new Map<string, number>();

function enterDurableModelCall(): string {
  const workflowID = DBOS.workflowID!;
  const inflight = inflightModelCalls.get(workflowID) ?? 0;
  if (inflight > 0) {
    throw new Error(
      `Concurrent durable model calls detected in workflow "${workflowID}". The Vercel AI SDK ` +
        `issues concurrent calls (e.g. Promise.all over generateText/streamText, embedMany over ` +
        `inputs larger than the model's per-call limit, or parallel tool sub-agents) in a ` +
        `nondeterministic order, which is unsafe for DBOS replay and can silently bind a checkpoint ` +
        `to the wrong call on recovery. Run each concurrent model call in its own child workflow with ` +
        `DBOS.startWorkflow instead. See the "Concurrency" section of the README.`,
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
export function durableCalls(options: StepConfig = {}): LanguageModelV4Middleware {
  const stepConfig = options;
  return {
    specificationVersion: 'v4',

    wrapGenerate: async ({ doGenerate, model }) => {
      assertNotInTransaction('generate');
      if (!isInWorkflowFunction()) {
        return await doGenerate();
      }
      const workflowID = enterDurableModelCall();
      try {
        return await DBOS.runStep(async () => encodeBinaryContent(await doGenerate()), {
          ...stepConfig,
          name: stepConfig.name ?? stepName(model, 'generate'),
        });
      } finally {
        exitDurableModelCall(workflowID);
      }
    },

    wrapStream: async ({ doStream, model }) => {
      assertNotInTransaction('stream');
      if (!isInWorkflowFunction()) {
        return await doStream();
      }
      const workflowID = enterDurableModelCall();

      // With retries enabled, a failed attempt may have already produced parts, so
      // live pass-through would deliver output from multiple attempts. Instead,
      // buffer and emit the (synthesized) parts only after the step succeeds.
      const buffered = stepConfig.retriesAllowed === true;

      let executed = false;
      let cancelled = false;
      let controller!: ReadableStreamDefaultController<LanguageModelV4StreamPart>;
      let step!: Promise<LanguageModelV4GenerateResult>;
      const stream = new ReadableStream<LanguageModelV4StreamPart>({
        start(c) {
          controller = c;
        },
        // On cancellation the step keeps draining the model to a checkpoint. Await
        // it so a workflow that cancels early still blocks until the result is
        // durable, instead of racing the checkpoint against workflow completion.
        async cancel() {
          cancelled = true;
          await step.catch(() => {});
        },
      });
      const emit = (part: LanguageModelV4StreamPart) => {
        if (!cancelled) controller.enqueue(part);
      };

      step = DBOS.runStep(
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
      void step
        .then(
          (recorded) => {
            if ((buffered || !executed) && !cancelled) {
              for (const part of replayParts(recorded)) emit(part);
            }
            if (!cancelled) controller.close();
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
      const workflowID = enterDurableModelCall();
      try {
        return await DBOS.runStep(async () => doEmbed(), {
          ...options,
          name: options.name ?? stepName(model, 'embed'),
        });
      } finally {
        exitDurableModelCall(workflowID);
      }
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
  // These map a block id to the text/reasoning content object that is ALSO already
  // in `content`. Blocks are appended to `content` at their `-start` and mutated in
  // place, so `content` preserves arrival order — matching the AI SDK's own stream
  // recorder. (Deferring the push to `-end` would place any part that arrives while
  // a block is open ahead of that block, and would drop a block that never closes.)
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

  // Fallbacks for a malformed stream that emits a delta without a preceding start:
  // create the block in arrival position rather than dropping the text.
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
