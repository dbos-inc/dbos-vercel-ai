import { DBOS } from '@dbos-inc/dbos-sdk';
import type { FlexibleSchema, ModelMessage, Tool } from 'ai' with { 'resolution-mode': 'import' };
import { writeDurableStream } from './durable-stream';
import { isInWorkflowFunction } from './internal';

/** Marks a tool built by agentTool: durableTools leaves it unwrapped (it is a child workflow, not a step) and binds its durable stream. */
export const AGENT_TOOL: unique symbol = Symbol.for('@dbos-inc/vercel-ai/agentTool');

// What agentTool needs from an agent: the AI SDK's Agent interface, structurally.
type StreamingAgent = {
  stream(options: { prompt: string } | { messages: ModelMessage[] }): PromiseLike<{ consumeStream(): PromiseLike<void>; readonly text: PromiseLike<string> }>;
};

export interface AgentToolOptions<INPUT, AGENT extends StreamingAgent, OUTPUT> {
  /** Name of the child workflow; must be unique. */
  name: string;
  description: string;
  inputSchema: FlexibleSchema<INPUT>;
  agent: AGENT;
  /** Turns the tool's input into the sub-agent's prompt. */
  prompt: (input: INPUT) => string | ModelMessage[];
  /** Turns the sub-agent's result into the tool's output (default: its final text); must be serializable. */
  output?: (result: Awaited<ReturnType<AGENT['stream']>>) => OUTPUT | Promise<OUTPUT>;
  /** Record the call in this durable stream: a `data-dbos-subagent` part naming the child workflow, then its output. */
  durableStream?: string;
  /** Name of a DBOS queue to run each child on, e.g. to bound how many sub-agents run at once. */
  queue?: string;
  /** Workflow timeout for each child. */
  timeoutMS?: number;
}

export type AgentTool<INPUT, OUTPUT> = Tool<INPUT, OUTPUT> & {
  /** The registered child workflow; call it directly to run the sub-agent without a model in the loop. */
  workflow: (input: INPUT) => Promise<OUTPUT>;
};

/**
 * Wraps an agent as a tool whose every call runs as a child workflow: durable at model-call granularity, with its own
 * concurrency guard, safe to call in parallel, and visible as a child in the parent's step list. Call it at module load,
 * before `DBOS.launch()`, since it registers the child workflow.
 */
export function agentTool<INPUT, AGENT extends StreamingAgent, OUTPUT = string>(
  options: AgentToolOptions<INPUT, AGENT, OUTPUT>,
): AgentTool<INPUT, OUTPUT> {
  const { name, agent, prompt, output } = options;
  const run = async (input: INPUT): Promise<OUTPUT> => {
    const request = prompt(input);
    // stream, not generate: only streamed calls write to a durable stream.
    const result = await agent.stream(typeof request === 'string' ? { prompt: request } : { messages: request });
    await result.consumeStream();
    return output ? await output(result as Awaited<ReturnType<AGENT['stream']>>) : ((await result.text) as OUTPUT);
  };
  const workflow = DBOS.registerWorkflow(run, { name });
  return build(options, workflow, options.durableStream);
}

function build<INPUT, AGENT extends StreamingAgent, OUTPUT>(
  options: AgentToolOptions<INPUT, AGENT, OUTPUT>,
  workflow: (input: INPUT) => Promise<OUTPUT>,
  durableStream: string | undefined,
): AgentTool<INPUT, OUTPUT> {
  const { name, description, inputSchema, queue, timeoutMS } = options;
  const execute = async (input: INPUT, execOptions: { toolCallId: string; abortSignal?: AbortSignal }): Promise<OUTPUT> => {
    if (!isInWorkflowFunction()) return workflow(input);
    const { toolCallId } = execOptions;
    // The tool call id comes from the checkpointed model output, so the child id is the same on replay and known for cancellation.
    const childID = `${DBOS.workflowID}-${toolCallId}`;
    let invoke = () => workflow(input);
    if (timeoutMS !== undefined) invoke = ((inner) => () => DBOS.withWorkflowTimeout(timeoutMS, inner))(invoke);
    if (queue) invoke = ((inner) => () => DBOS.withWorkflowQueue(queue, inner))(invoke);
    // Invoke first: the direct call reserves the child's function ids synchronously, so parallel calls replay in order.
    const pending = DBOS.withNextWorkflowID(childID, invoke);
    pending.catch(() => {});
    const cancel = () => void DBOS.cancelWorkflow(childID).catch(() => {});
    execOptions.abortSignal?.addEventListener('abort', cancel, { once: true });
    try {
      if (durableStream) {
        await writeDurableStream(durableStream, [
          { type: 'data-dbos-subagent', id: toolCallId, data: { toolCallId, workflowID: childID, name } },
        ]);
      }
      const result = await pending;
      if (durableStream) await writeDurableStream(durableStream, [{ type: 'tool-output-available', toolCallId, output: result }]);
      return result;
    } catch (error) {
      if (durableStream) {
        const errorText = error instanceof Error ? error.message : String(error);
        await writeDurableStream(durableStream, [{ type: 'tool-output-error', toolCallId, errorText }]);
      }
      throw error;
    } finally {
      execOptions.abortSignal?.removeEventListener('abort', cancel);
    }
  };
  return {
    description,
    inputSchema,
    execute,
    workflow,
    [AGENT_TOOL]: (key: string) => build(options, workflow, key),
  } as unknown as AgentTool<INPUT, OUTPUT>;
}
