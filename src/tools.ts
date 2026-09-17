import { DBOS, StepConfig } from '@dbos-inc/dbos-sdk';
import type { ToolSet } from 'ai' with { 'resolution-mode': 'import' };
import { assertNotInTransaction, isAsyncIterable, isInWorkflowFunction, runDurableStep, withErrorClassification } from './internal';

export interface DurableToolsOptions extends StepConfig {
  /** Per-tool step config overriding the defaults; `false` leaves that tool non-durable. */
  tools?: Record<string, StepConfig | false>;
}

// Loose view of a tool's execute; the AI SDK validates input and supplies the options.
type ToolExecute = (input: unknown, options: { toolCallId: string; abortSignal?: AbortSignal }) => unknown;

/**
 * Wraps each tool's `execute` so that, inside a workflow, every tool call runs as a durable DBOS step named
 * `<tool>.<toolCallId>` and replays from its checkpoint on recovery; outside a workflow tools run unchanged.
 * Retries are off by default (the AI SDK never retries tools); opt in per tool with `retriesAllowed`.
 */
export function durableTools<TOOLS extends ToolSet>(tools: TOOLS, options: DurableToolsOptions = {}): TOOLS {
  const { tools: perTool, ...defaults } = options;
  const durable: ToolSet = {};
  for (const [name, definition] of Object.entries(tools)) {
    const override = perTool?.[name];
    if (typeof definition.execute !== 'function' || override === false) {
      durable[name] = definition;
      continue;
    }
    const execute = definition.execute as ToolExecute;
    const merged: StepConfig = { ...defaults, ...override };
    // Default classification (aborts and provider-declared non-retryable errors are terminal), but retries stay opt-in.
    const stepConfig: StepConfig = { ...withErrorClassification(merged), retriesAllowed: merged.retriesAllowed ?? false };
    const prefix = stepConfig.name ?? name;
    durable[name] = {
      ...definition,
      execute: (input: unknown, execOptions: Parameters<ToolExecute>[1]) => {
        assertNotInTransaction(name);
        if (!isInWorkflowFunction()) return execute(input, execOptions);
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
            const output = await execute(input, abortSignal === signal ? execOptions : { ...execOptions, abortSignal });
            // A streaming execute can't checkpoint mid-flight; drain it and record the final value (the last yield).
            if (isAsyncIterable(output)) {
              let last: unknown;
              for await (last of output);
              return last;
            }
            return output;
          },
          callConfig,
        );
      },
    } as ToolSet[string];
  }
  return durable as TOOLS;
}
