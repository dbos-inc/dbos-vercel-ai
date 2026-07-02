import { DBOS, StepConfig } from '@dbos-inc/dbos-sdk';
import type { ToolSet } from 'ai' with { 'resolution-mode': 'import' };
import { isInWorkflowFunction, withErrorClassification } from './internal';

// Structural type for an MCP client (e.g. from @ai-sdk/mcp or ai's experimental_createMCPClient) — only what we use.
export interface MCPClientLike {
  tools(options?: unknown): Promise<ToolSet>;
  close?(): Promise<void>;
}

interface DurableToolDef {
  description?: string;
  inputJsonSchema: unknown;
}

/**
 * Wraps an MCP client so its tool listing and each tool call run as durable DBOS steps: the tool
 * list (JSON schemas) is checkpointed so recovery needs no live connection, and each tool call is
 * checkpointed so a recovered workflow replays results instead of re-invoking the tool.
 */
export async function durableMCPTools(client: MCPClientLike, options: StepConfig = {}): Promise<ToolSet> {
  const stepConfig = withErrorClassification(options);
  const { asSchema, dynamicTool, jsonSchema } = await import('ai');
  const run = <T>(name: string, fn: () => Promise<T>): Promise<T> =>
    isInWorkflowFunction() ? DBOS.runStep(fn, { ...stepConfig, name }) : fn();

  // Checkpoint the tool list as plain JSON schemas, so replay reconstructs tools without the live client.
  const listed = await run('mcp.listTools', async () => {
    const tools = await client.tools();
    const defs: Record<string, DurableToolDef> = {};
    for (const [name, tool] of Object.entries(tools)) {
      const description = typeof tool.description === 'string' ? tool.description : undefined;
      // Await: a Schema's jsonSchema may be a Promise, which would otherwise checkpoint as {} and yield an empty schema.
      defs[name] = { description, inputJsonSchema: await asSchema(tool.inputSchema).jsonSchema };
    }
    return defs;
  });

  const durable: ToolSet = {};
  for (const [name, def] of Object.entries(listed)) {
    durable[name] = dynamicTool({
      description: def.description ?? '',
      inputSchema: jsonSchema(def.inputJsonSchema as Parameters<typeof jsonSchema>[0]),
      // Re-fetch the live tool inside the step (its execute closure can't be checkpointed); replay returns the recorded result.
      execute: (input: unknown, execOptions) =>
        run(`mcp.tool.${name}`, async () => {
          const tool = (await client.tools())[name];
          if (!tool?.execute) throw new Error(`MCP tool "${name}" is not executable.`);
          return tool.execute(input, execOptions);
        }),
    });
  }
  return durable;
}
