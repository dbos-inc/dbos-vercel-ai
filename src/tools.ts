import { AsyncLocalStorage } from 'node:async_hooks';
import { DBOS, StepConfig } from '@dbos-inc/dbos-sdk';
import type { ToolSet, UIMessageChunk, UIMessageStreamWriter } from 'ai' with { 'resolution-mode': 'import' };
import { assertNotInTransaction, isAsyncIterable, isInWorkflowFunction, runDurableStep, withErrorClassification } from './internal';
import { AGENT_TOOL } from './agent-tool';
import { writeDurableStream, writeToolRecord } from './durable-stream';

export interface DurableToolsOptions extends StepConfig {
  /** Per-tool step config overriding the defaults; `false` leaves that tool non-durable. */
  tools?: Record<string, StepConfig | false>;
  /** Write each tool call's output (or error) to this durable stream from inside its step. */
  durableStream?: string;
  /** Receives chunks tools write via `toolWriter()`; those that are part of the message are checkpointed and re-emitted on replay, before the tool's output. */
  writer?: UIMessageStreamWriter;
}

const currentWriter = new AsyncLocalStorage<UIMessageStreamWriter>();

/** Returns a UI message stream writer bound to the current `durableTools` tool call. */
export function toolWriter(): UIMessageStreamWriter {
  const writer = currentWriter.getStore();
  if (!writer) throw new Error('toolWriter() can only be called from a tool wrapped by durableTools.');
  return writer;
}

// Tool output plus the message chunks its call wrote, as checkpointed; a bare output is a call that wrote none.
interface ToolEnvelope {
  __dbosToolChunks: 1;
  output: unknown;
  chunks: UIMessageChunk[];
}

function isToolEnvelope(value: unknown): value is ToolEnvelope {
  return typeof value === 'object' && value !== null && (value as Partial<ToolEnvelope>).__dbosToolChunks === 1;
}

function isTransient(chunk: UIMessageChunk): boolean {
  return (chunk as { transient?: boolean }).transient === true;
}

// Outside a step: every chunk goes straight to the writer, or nowhere.
function liveWriter(writer: UIMessageStreamWriter | undefined): UIMessageStreamWriter {
  return writer ?? { write() {}, merge() {}, onError: undefined };
}

/** One attempt's writer: transient chunks go out live, the rest wait for the step to succeed. */
class StepWriter implements UIMessageStreamWriter {
  readonly chunks: UIMessageChunk[] = [];
  private pending: Promise<unknown>[] = [];

  constructor(
    private readonly writer: UIMessageStreamWriter | undefined,
    private readonly durableStream: string | undefined,
  ) {}

  get onError() {
    return this.writer?.onError;
  }

  write(chunk: UIMessageChunk): void {
    if (!isTransient(chunk)) {
      this.chunks.push(chunk);
      return;
    }
    this.writer?.write(chunk);
    if (this.durableStream) this.track(writeDurableStream(this.durableStream, [chunk]));
  }

  merge(stream: ReadableStream<UIMessageChunk>): void {
    this.track(
      (async () => {
        for await (const chunk of stream) this.write(chunk);
      })(),
    );
  }

  // Observed now so a failure after the tool throws is not an unhandled rejection; settle still sees it.
  private track(promise: Promise<unknown>): void {
    promise.catch(() => {});
    this.pending.push(promise);
  }

  /** Waits for merges and live writes, including any started while waiting; a failed one fails the call. */
  async settle(): Promise<void> {
    while (this.pending.length > 0) {
      const batch = this.pending;
      this.pending = [];
      await Promise.all(batch);
    }
  }
}

// Loose view of a tool's execute; the AI SDK validates input and supplies the options.
type ToolExecute = (input: unknown, options: { toolCallId: string; abortSignal?: AbortSignal }) => unknown;

/**
 * Wraps each tool's `execute` so that, inside a workflow, every tool call runs as a durable DBOS step named
 * `<tool>.<toolCallId>` and replays from its checkpoint on recovery; outside a workflow tools run unchanged.
 * Retries are off by default (the AI SDK never retries tools); opt in per tool with `retriesAllowed`.
 */
export function durableTools<TOOLS extends ToolSet>(tools: TOOLS, options: DurableToolsOptions = {}): TOOLS {
  const { tools: perTool, durableStream, writer, ...defaults } = options;
  const live = liveWriter(writer);
  const durable: ToolSet = {};
  for (const [name, definition] of Object.entries(tools)) {
    const override = perTool?.[name];
    // An agent tool is a child workflow, not a step: leave it unwrapped, binding this durable stream to it.
    const bindAgentTool = (definition as { [AGENT_TOOL]?: (key: string) => ToolSet[string] })[AGENT_TOOL];
    if (bindAgentTool) {
      durable[name] = durableStream ? bindAgentTool(durableStream) : definition;
      continue;
    }
    if (typeof definition.execute !== 'function') {
      durable[name] = definition;
      continue;
    }
    const execute = definition.execute as ToolExecute;
    if (override === false) {
      durable[name] = {
        ...definition,
        execute: (input: unknown, execOptions: Parameters<ToolExecute>[1]) => currentWriter.run(live, () => execute(input, execOptions)),
      } as ToolSet[string];
      continue;
    }
    const merged: StepConfig = { ...defaults, ...override };
    // Default classification (aborts and provider-declared non-retryable errors are terminal), but retries stay opt-in.
    const stepConfig: StepConfig = { ...withErrorClassification(merged), retriesAllowed: merged.retriesAllowed ?? false };
    const prefix = stepConfig.name ?? name;
    durable[name] = {
      ...definition,
      execute: (input: unknown, execOptions: Parameters<ToolExecute>[1]) => {
        assertNotInTransaction(name);
        if (!isInWorkflowFunction()) return currentWriter.run(live, () => execute(input, execOptions));
        const signal = execOptions.abortSignal;
        // An aborted call is done whatever the failure looks like; a retry would re-run a cancelled side effect.
        const callConfig: StepConfig = {
          ...stepConfig,
          shouldRetry: async (error: unknown) => !signal?.aborted && (await stepConfig.shouldRetry!(error)),
        };
        // The tool call id comes from the checkpointed model result, so a reordered parallel step fails replay instead of swapping results.
        return runDurableStep(
          `${prefix}.${execOptions.toolCallId}`,
          async () => {
            // A timed-out attempt is abandoned by DBOS but keeps running; forward its signal so the tool stops too.
            const timeoutSignal = DBOS.stepStatus?.timeoutSignal;
            const abortSignal = timeoutSignal && signal ? AbortSignal.any([signal, timeoutSignal]) : (timeoutSignal ?? signal);
            const stepWriter = new StepWriter(writer, durableStream);
            let output: unknown;
            try {
              output = await currentWriter.run(stepWriter, async () => {
                const value = await execute(input, abortSignal === signal ? execOptions : { ...execOptions, abortSignal });
                // A streaming execute can't checkpoint mid-flight; drain it and record the final value (the last yield).
                if (!isAsyncIterable(value)) return value;
                let last: unknown;
                for await (last of value);
                return last;
              });
              await stepWriter.settle();
            } catch (error) {
              if (durableStream) await writeToolRecord(durableStream, execOptions.toolCallId, { errorText: errorMessage(error) });
              throw error;
            }
            const chunks = stepWriter.chunks;
            if (durableStream) await writeToolRecord(durableStream, execOptions.toolCallId, { output }, chunks);
            return chunks.length > 0 ? ({ __dbosToolChunks: 1, output, chunks } satisfies ToolEnvelope) : output;
          },
          callConfig,
        ).then((result) => {
          if (!isToolEnvelope(result)) return result;
          // Runs on first execution and on replay alike, so the workflow's message gets the same parts either way.
          for (const chunk of result.chunks) writer?.write(chunk);
          return result.output;
        });
      },
    } as ToolSet[string];
  }
  return durable as TOOLS;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
