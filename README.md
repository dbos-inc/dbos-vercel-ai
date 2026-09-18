# @dbos-inc/vercel-ai

[DBOS](https://docs.dbos.dev/) durable execution for the [Vercel AI SDK](https://ai-sdk.dev/).

This package makes AI SDK **agents** durable, backed by your Postgres database.
All you have to do is wrap your model with `durableCalls` and run your generation inside a DBOS workflow.
Then, this integration automatically checkpoints every action your agents take in Postgres.
If your process crashes mid-agent, DBOS replays the completed steps from their checkpoints and the agent resumes exactly where it left off.

This package is implemented as standard AI SDK [middleware](https://ai-sdk.dev/docs/ai-sdk-core/middleware), so you keep your provider, your model configuration, and the familiar APIs like `generateText`, `streamText`, and `ToolLoopAgent`.
Durability is transparent to your agent code.

```ts
import { DBOS } from '@dbos-inc/dbos-sdk';
import { generateText, wrapLanguageModel } from 'ai';
import { openai } from '@ai-sdk/openai';
import { durableCalls } from '@dbos-inc/vercel-ai';

const model = wrapLanguageModel({
  model: openai('gpt-5'),
  middleware: durableCalls({ retriesAllowed: true, maxAttempts: 5 }),
});

const researchAgent = DBOS.registerWorkflow(
  async (question: string) => {
    const { text } = await generateText({
      model,
      prompt: question,
      system: 'You are a helpful research assistant.',
    });
    return text;
  },
  { name: 'researchAgent' },
);

DBOS.setConfig({ name: 'my-agent', systemDatabaseUrl: process.env.DBOS_SYSTEM_DATABASE_URL });
await DBOS.launch();

console.log(await researchAgent('Why did the agent cross the road?'));
```

## Installation

```sh
npm install @dbos-inc/vercel-ai @dbos-inc/dbos-sdk ai
```

Requires DBOS v4.21+ or v5, AI SDK v7+, and a Postgres database for DBOS.

## How it works

When an agent runs inside a DBOS workflow, DBOS makes three things durable:

- **Every model call.** `durableCalls()` is AI SDK middleware that intercepts `doGenerate`/`doStream` and runs each call through [`DBOS.runStep`](https://docs.dbos.dev/typescript/tutorials/step-tutorial). The complete result (content, usage, finish reason, response metadata) is checkpointed in Postgres. On recovery, completed calls replay from their checkpoints without contacting the model provider.
- **The agent loop.** Because DBOS workflows replay deterministically on recovery and each model call replays from its checkpoint, a multi-step, tool-calling agent resumes from the first unfinished step instead of restarting from the beginning.
- **Tool calls.** Your own tools are checkpointed when wrapped with [`durableTools`](#tools), and MCP tools via [`durableMCPTools`](#mcp-tools). On recovery, completed tool calls replay their recorded output instead of re-running.

Outside a workflow (or inside another step) the wrapped model calls the provider directly with no checkpointing, so the same model works anywhere in your app.

All DBOS step options are accepted and apply per model call:

```ts
durableCalls({
  retriesAllowed: true,   // retry failed model calls (default: true)
  maxAttempts: 5,         // total attempts when retries are allowed (default: 3)
  intervalSeconds: 1,     // delay before first retry (default: 1)
  backoffRate: 2,         // exponential backoff multiplier (default: 2)
  shouldRetry: (error) => true,  // per-error retry predicate (default: skip provider-declared non-retryable errors and aborts)
  timeoutMS: 60000,       // per-attempt timeout
  name: 'my-model-call',  // step name (default: "<provider>.<modelId>.<operation>")
});
```

Retries are on by default so that a transient provider error is absorbed inside a single durable step.
The default `shouldRetry` treats errors the provider marks non-retryable (an AI SDK `APICallError`/`GatewayError` with `isRetryable === false`, e.g. a 401 or an invalid-request 400) and aborts/timeouts as terminal, so they fail fast instead of retrying `maxAttempts` times. 
Pass your own `shouldRetry` to override it, or `retriesAllowed: false` to disable step retries.

Because DBOS owns retries by default, pass `maxRetries: 0` to the AI SDK call so retry behavior is governed in one place; otherwise the two compose multiplicatively and each AI SDK retry is a fresh step.

## Durable streams

You can **durably stream** agent or model output so it can be read by an external client or UI.
To do this, configure `durableCalls` or `durableTools`/`durableMCPTools` with a durable stream name:

```ts
import { createUIMessageStreamResponse, streamText } from 'ai';
import { durableCalls, durableTools, readDurableStream } from '@dbos-inc/vercel-ai';

const model = wrapLanguageModel({ model: openai('gpt-5'), middleware: durableCalls({ durableStream: 'ui' }) });
const tools = durableTools(myTools, { durableStream: 'ui' });

const chatTurn = DBOS.registerWorkflow(async (messages: ModelMessage[]) => {
  const result = streamText({ model, messages, tools, stopWhen: stepCountIs(10) });
  return await result.text;
}, { name: 'chatTurn' });

// POST: start the turn and stream it. GET: reconnect from the last offset the client saw.
const handle = await DBOS.startWorkflow(chatTurn)(messages);
return createUIMessageStreamResponse({
  stream: readDurableStream({ workflowID: handle.workflowID, key: 'ui', messageId }),
  headers: { 'x-dbos-workflow-id': handle.workflowID },
});
```

You can also write your own data to a stream with `writeDurableStream(key, chunks)`.
Your streams are closed when your workflow finishes; you can also close a stream early using `closeDurableStream`.

You can read from a durable stream using `readDurableStream`, for example to stream it to a UI.
It emits a stream of AI SDK `UIMessageChunk`s.
After every record it emits a transient `data-dbos-offset` chunk; to reconnect an interrupted stream, pass the last `offset` a client saw back as `readDurableStream({ ..., offset })` and the stream resumes from there.
You can also pass a `DBOSClient` into `readDurableStream` to read it from a different process.

## Tools

Model calls in a tool-calling loop are each checkpointed individually, so a recovered agent resumes mid-loop.
Wrap your tools with `durableTools` so each tool call is checkpointed too: on recovery, completed tool calls replay their recorded output (or error) instead of re-running.

```ts
import { tool, stepCountIs } from 'ai';
import { durableTools } from '@dbos-inc/vercel-ai';
import { z } from 'zod';

const tools = durableTools({
  getWeather: tool({
    description: 'Get the weather for a city',
    inputSchema: z.object({ city: z.string() }),
    execute: ({ city }) => fetchWeather(city),
  }),
});

const agent = DBOS.registerWorkflow(async (question: string) => {
  const result = await generateText({ model, prompt: question, tools, stopWhen: stepCountIs(10) });
  return result.text;
}, { name: 'weatherAgent' });
```

You can pass step configuration (such as timeouts or retries) to `durableTools`.
You can set default for all tools or configure tools individually.
Retries are off by default.

```ts
const tools = durableTools(myTools, {
  timeoutMS: 30_000,
  tools: {
    getWeather: { retriesAllowed: true, maxAttempts: 3 },
  },
});
```

### MCP tools

`durableMCPTools` wraps an [MCP](https://modelcontextprotocol.io/) client (e.g. from [`@ai-sdk/mcp`](https://www.npmjs.com/package/@ai-sdk/mcp)) so both the tool listing and every tool call run as durable steps.
Each tool call is checkpointed as a step named `mcp.tool.<tool>.<toolCallId>`, so recovery replays results instead of re-invoking the tool:

```ts
import { createMCPClient } from '@ai-sdk/mcp';
import { durableMCPTools } from '@dbos-inc/vercel-ai';

const agent = DBOS.registerWorkflow(async (question: string) => {
  const mcpClient = await createMCPClient({ transport: { type: 'http', url: MCP_URL } });
  const tools = await durableMCPTools(mcpClient);
  const result = await generateText({ model, prompt: question, tools, stopWhen: stepCountIs(10) });
  return result.text;
}, { name: 'mcpAgent' });
```

To use the client's explicit-schema mode (tool subsetting, typed inputs, output schemas), pass `toolOptions`; it is forwarded to `client.tools()` for both the listing and each tool call:

```ts
const tools = await durableMCPTools(mcpClient, {
  toolOptions: { schemas: { 'get-weather': { inputSchema: z.object({ city: z.string() }) } } },
});
```

## Embeddings

`durableEmbeddingCalls` enables durable calls to embedding models:

```ts
import { embedMany, wrapEmbeddingModel } from 'ai';
import { durableEmbeddingCalls } from '@dbos-inc/vercel-ai';

const embeddingModel = wrapEmbeddingModel({
  model: openai.textEmbeddingModel('text-embedding-3-small'),
  middleware: durableEmbeddingCalls({ retriesAllowed: true }),
});

const { embeddings } = await embedMany({ model: embeddingModel, values: chunks });
```

## Images

`durableImageCalls` makes image generation durable:

```ts
import { generateImage, wrapImageModel } from 'ai';
import { durableImageCalls } from '@dbos-inc/vercel-ai';

const imageModel = wrapImageModel({ model: openai.imageModel('gpt-image-1'), middleware: durableImageCalls() });

const { images } = await generateImage({ model: imageModel, prompt: 'a durable cat' });
```

