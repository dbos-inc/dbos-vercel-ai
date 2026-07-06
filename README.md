# @dbos-inc/vercel-ai

[DBOS](https://docs.dbos.dev/) durable execution for the [Vercel AI SDK](https://ai-sdk.dev/).

This package makes AI SDK **agents** durable.
All you have to do is wrap your model with `durableCalls` and run your generation inside a DBOS workflow.
Then, this integration automatically checkpoints every action your agents take in Postgres.
If your process crashes mid-agent, DBOS replays the completed steps from their checkpoints and the agent resumes exactly where it left off.

This integration works as standard AI SDK [middleware](https://ai-sdk.dev/docs/ai-sdk-core/middleware), so you keep your provider, your model configuration, and the familiar `generateText` / `streamText` API.
The durability is transparent to your agent code.

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

Requires `ai` v7+ and a Postgres database for DBOS.

## How it works

Running an agent inside a DBOS workflow makes three things durable:

- **Every model call.** `durableCalls()` is AI SDK middleware that intercepts `doGenerate`/`doStream` and runs each call through [`DBOS.runStep`](https://docs.dbos.dev/typescript/tutorials/step-tutorial). The complete result (content, usage, finish reason, response metadata) is checkpointed in Postgres; on recovery, a completed call returns its recorded result without contacting the model.
- **The agent loop.** Because the workflow re-executes deterministically on recovery and each model call replays from its checkpoint, a multi-step, tool-calling agent resumes from the first unfinished step instead of restarting from the beginning.
- **Tool calls.** MCP tools (via [`durableMCPTools`](#mcp-tools)) are checkpointed automatically. Your own tools' side effects are durable when you wrap their `execute` in `DBOS.runStep` (see [Tools](#tools)).

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

## Streaming

You can stream model responses in a workflow with `streamText`.
When streaming model output in a workflow, only the final output is checkpointed, not individual deltas.
As a consequence:

- You can safely forward streamed deltas to a UI or print them to a terminal, but you should not perform durable actions on them. Instead, wait until the stream is complete before calling tools or otherwise progressing your workflow.
- Do not break out of a stream before it is complete. Instead, either explicitly abort the stream or wait for it to complete before progressing your workflow.

```ts
const streamingAgent = DBOS.registerWorkflow(async (prompt: string) => {
  const result = streamText({ model, prompt });
  for await (const delta of result.textStream) {
    process.stdout.write(delta);
  }
  return await result.text;
}, { name: 'streamingAgent' });
```

## Tools

Model calls in a tool-calling loop are each checkpointed individually, so a recovered agent resumes mid-loop. A tool's `execute` is your own code, though: wrap its side effects in a DBOS step so they're checkpointed too.

```ts
import { tool, stepCountIs } from 'ai';
import { z } from 'zod';

const agent = DBOS.registerWorkflow(async (question: string) => {
  const result = await generateText({
    model,
    prompt: question,
    tools: {
      getWeather: tool({
        description: 'Get the weather for a city',
        inputSchema: z.object({ city: z.string() }),
        execute: ({ city }) => DBOS.runStep(() => fetchWeather(city), { name: 'getWeather' }),
      }),
    },
    stopWhen: stepCountIs(10),
  });
  return result.text;
}, { name: 'weatherAgent' });
```

### MCP tools

`durableMCPTools` wraps an [MCP](https://modelcontextprotocol.io/) client (e.g. from [`@ai-sdk/mcp`](https://www.npmjs.com/package/@ai-sdk/mcp)) so both the tool listing and every tool call run as durable steps. The tool list is checkpointed as JSON schemas, so a recovered workflow reconstructs the tools without the live connection, and each tool call is checkpointed so recovery replays results instead of re-invoking the tool:

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

## Concurrency

Run **one durable model call at a time within a single workflow**. DBOS derives each step's replay identity from the order steps are reached, but the AI SDK issues concurrent model calls in a nondeterministic order — so on recovery a checkpoint could be bound to the wrong call, silently returning one call's result for another. To prevent this, the middleware throws if it detects a second durable model call starting while one is already in flight in the same workflow.

Sequential calls (including a normal tool-calling loop, where each model call completes before the next begins) are unaffected.

To fan out model calls in parallel, give each its own **child workflow**, which gets an independent, deterministic step-ID space:

```ts
const summarizeOne = DBOS.registerWorkflow(
  async (doc: string) => (await generateText({ model, prompt: `Summarize: ${doc}` })).text,
  { name: 'summarizeOne' },
);

const summarizeAll = DBOS.registerWorkflow(async (docs: string[]) => {
  const handles = await Promise.all(
    docs.map((doc) => DBOS.startWorkflow(summarizeOne)(doc)),
  );
  return Promise.all(handles.map((h) => h.getResult()));
}, { name: 'summarizeAll' });
```

## Embeddings

`durableEmbeddingCalls` does the same for embedding models:

```ts
import { embedMany, wrapEmbeddingModel } from 'ai';
import { durableEmbeddingCalls } from '@dbos-inc/vercel-ai';

const embeddingModel = wrapEmbeddingModel({
  model: openai.textEmbeddingModel('text-embedding-3-small'),
  middleware: durableEmbeddingCalls({ retriesAllowed: true }),
});

const { embeddings } = await embedMany({ model: embeddingModel, values: chunks, maxParallelCalls: 1 });
```

Pass `maxParallelCalls: 1` when embedding more values than the model's per-call limit. `embedMany` otherwise splits the input into batches and runs them concurrently, which the concurrency guard rejects (their step order would be nondeterministic on replay); `maxParallelCalls: 1` runs the batches sequentially, keeping them durable and replay-safe.

## Images

`durableImageCalls` makes image generation durable. Generated image bytes are base64-encoded before checkpointing (the `.uint8Array`/`.base64` accessors on the result work either way):

```ts
import { generateImage, wrapImageModel } from 'ai';
import { durableImageCalls } from '@dbos-inc/vercel-ai';

const imageModel = wrapImageModel({ model: openai.imageModel('gpt-image-1'), middleware: durableImageCalls() });

const { images } = await generateImage({ model: imageModel, prompt: 'a durable cat' });
```

