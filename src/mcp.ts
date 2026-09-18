import { StepConfig } from '@dbos-inc/dbos-sdk';
import type { ToolSet } from 'ai' with { 'resolution-mode': 'import' };
import { isAsyncIterable, runDurableStep, withErrorClassification } from './internal';

// Structural type for an MCP client (e.g. from @ai-sdk/mcp) — deliberately loose: the AI SDK ecosystem
// exact-pins @ai-sdk/provider-utils, so precise Tool types fail to match across skewed copies.
export interface MCPClientLike {
  tools(options?: unknown): Promise<Record<string, unknown>>;
  close?(): Promise<void>;
}

// Duck-typed view of a client tool; every field is verified at runtime before use.
interface MCPToolLike {
  description?: unknown;
  title?: unknown;
  metadata?: ToolSet[string]['metadata'];
  _meta?: unknown;
  toModelOutput?: unknown;
  inputSchema?: unknown;
  execute?: (input: unknown, options: unknown) => unknown;
}

export interface DurableMCPToolsOptions extends StepConfig {
  /** Forwarded to client.tools() on listing and on each call (e.g. { schemas } for subsetting and output schemas). */
  toolOptions?: unknown;
}

interface DurableToolDef {
  description?: string;
  title?: string;
  metadata?: ToolSet[string]['metadata'];
  meta?: unknown;
  convertsOutput: boolean;
  inputJsonSchema: unknown;
}

type ToolModelOutput = Awaited<ReturnType<NonNullable<ToolSet[string]['toModelOutput']>>>;

// Mirror of @ai-sdk/mcp's toModelOutput: MCP content becomes model content (text stays text, images become files).
function mcpToolOutput(output: unknown): ToolModelOutput {
  const result = output as { content?: unknown };
  if (result === null || typeof result !== 'object' || !Array.isArray(result.content)) {
    return { type: 'json', value: output } as ToolModelOutput;
  }
  return {
    type: 'content',
    value: result.content.map((part: { type?: string; text?: string; data?: string; mimeType?: string }) => {
      if (part.type === 'text' && typeof part.text === 'string') {
        return { type: 'text' as const, text: part.text };
      }
      if (part.type === 'image' && part.data !== undefined && part.mimeType !== undefined) {
        return { type: 'file' as const, mediaType: part.mimeType, data: { type: 'data' as const, data: part.data } };
      }
      return { type: 'text' as const, text: JSON.stringify(part) };
    }),
  };
}

/**
 * Wraps an MCP client so its tool listing and each tool call run as durable DBOS steps: the tool
 * list (JSON schemas) is checkpointed so recovery needs no live connection, and each tool call is
 * checkpointed so a recovered workflow replays results instead of re-invoking the tool.
 */
export async function durableMCPTools(client: MCPClientLike, options: DurableMCPToolsOptions = {}): Promise<ToolSet> {
  const { toolOptions, ...stepOptions } = options;
  const stepConfig = withErrorClassification(stepOptions);
  const { asSchema, dynamicTool, jsonSchema } = await import('ai');
  const run = <T>(name: string, fn: () => Promise<T>, config: StepConfig = stepConfig): Promise<T> =>
    runDurableStep(name, fn, config);

  // Checkpoint the tool list as plain JSON schemas, so replay reconstructs tools without the live client.
  const listed = await run('mcp.listTools', async () => {
    const tools = await client.tools(toolOptions);
    const defs: Record<string, DurableToolDef> = {};
    for (const [name, rawTool] of Object.entries(tools)) {
      const tool = rawTool as MCPToolLike;
      defs[name] = {
        description: typeof tool.description === 'string' ? tool.description : undefined,
        title: typeof tool.title === 'string' ? tool.title : undefined,
        metadata: tool.metadata,
        meta: tool._meta,
        convertsOutput: typeof tool.toModelOutput === 'function',
        // Await: a Schema's jsonSchema may be a Promise, which would otherwise checkpoint as {} and yield an empty schema.
        inputJsonSchema: await asSchema(tool.inputSchema as Parameters<typeof asSchema>[0]).jsonSchema,
      };
    }
    return defs;
  });

  const durable: ToolSet = {};
  for (const [name, def] of Object.entries(listed)) {
    const reconstructed = dynamicTool({
      description: def.description,
      title: def.title,
      metadata: def.metadata,
      inputSchema: jsonSchema(def.inputJsonSchema as Parameters<typeof jsonSchema>[0]),
      // MCP clients convert results via a pure toModelOutput; reapply an equivalent so results reach the model as content, not raw JSON.
      toModelOutput: def.convertsOutput ? ({ output }) => mcpToolOutput(output) : undefined,
      // Re-fetch the live tool inside the step (its execute closure can't be checkpointed); replay returns the recorded result.
      execute: (input: unknown, execOptions) => {
        const { abortSignal: signal, toolCallId } = (execOptions ?? {}) as { abortSignal?: AbortSignal; toolCallId?: string };
        // An aborted consumer is done with this call, whatever the failure looks like; a retry would re-run a cancelled side effect.
        const callConfig: StepConfig = {
          ...stepConfig,
          shouldRetry: async (error: unknown) =>
            !signal?.aborted && (stepConfig.shouldRetry ? await stepConfig.shouldRetry(error) : true),
        };
        // The tool call id comes from the checkpointed model result, so a reordered parallel step fails replay instead of swapping results.
        return run(
          `mcp.tool.${name}.${toolCallId ?? 'call'}`,
          async () => {
            const tool = (await client.tools(toolOptions))[name] as MCPToolLike | undefined;
            if (typeof tool?.execute !== 'function') throw new Error(`MCP tool "${name}" is not executable.`);
            const output = await tool.execute(input, execOptions);
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
    });
    // @ai-sdk/mcp spreads the MCP _meta onto the tool object; preserve it for consumers that read it.
    durable[name] = def.meta === undefined ? reconstructed : Object.assign(reconstructed, { _meta: def.meta });
  }
  return durable;
}
