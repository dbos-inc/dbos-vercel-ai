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

// AI SDK errors (APICallError, GatewayError) expose an isRetryable flag; treat an explicit false as terminal.
function isNonRetryable(error: unknown): boolean {
  return (
    typeof error === 'object' &&
    error !== null &&
    'isRetryable' in error &&
    (error as { isRetryable?: unknown }).isRetryable === false
  );
}

// Inject a default shouldRetry that skips provider-declared non-retryable errors (e.g. a 401); a caller-provided one wins, but an explicit `undefined` falls back to the default.
export function withErrorClassification(options: StepConfig): StepConfig {
  return { ...options, shouldRetry: options.shouldRetry ?? ((error: unknown) => !isNonRetryable(error)) };
}
