import { DBOS, StepConfig } from '@dbos-inc/dbos-sdk';

export function isInWorkflowFunction(): boolean {
  // True only in workflow code proper: not in a step, transaction, or outside DBOS.
  return DBOS.isInWorkflow();
}

export function assertNotInTransaction(operation: string): void {
  if (DBOS.isInTransaction()) {
    throw new Error(`Cannot call ${operation} inside a DBOS transaction; run it in workflow or step code.`);
  }
}

// Aborts/timeouts are deliberate cancellations, never transient; retrying just re-runs an already-cancelled call.
function isAbortError(error: unknown): boolean {
  const name = (error as { name?: unknown } | null)?.name;
  return name === 'AbortError' || name === 'TimeoutError';
}

// AI SDK errors (APICallError, GatewayError) expose an isRetryable flag; treat an explicit false, and any abort, as terminal.
function isNonRetryable(error: unknown): boolean {
  try {
    return (
      isAbortError(error) ||
      (typeof error === 'object' &&
        error !== null &&
        'isRetryable' in error &&
        (error as { isRetryable?: unknown }).isRetryable === false)
    );
  } catch {
    // A throwing accessor must not replace the step's real error; treat as retryable.
    return false;
  }
}

// Default to DBOS-owned retries so a transient provider error is absorbed inside one step (never checkpointed as an
// error that replay would re-run); the default shouldRetry skips provider-declared non-retryable errors and aborts.
// A caller's retriesAllowed/shouldRetry wins, but an explicit `undefined` falls back to the default.
export function withErrorClassification(options: StepConfig): StepConfig {
  return {
    ...options,
    retriesAllowed: options.retriesAllowed ?? true,
    shouldRetry: options.shouldRetry ?? ((error: unknown) => !isNonRetryable(error)),
  };
}

