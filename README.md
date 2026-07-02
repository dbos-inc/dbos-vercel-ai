# @dbos-inc/vercel-ai

[DBOS](https://docs.dbos.dev/) durable execution for the [Vercel AI SDK](https://ai-sdk.dev/).

This package makes AI SDK model calls **durable**: each call to a language model runs as a DBOS step whose result is checkpointed in Postgres. If your program crashes or restarts mid-agent, DBOS recovers the workflow and replays completed model calls from their checkpoints instead of calling the model again — no repeated LLM spend, no lost progress, and tool calls resume exactly where they left off.

It works as standard AI SDK [middleware](https://ai-sdk.dev/docs/ai-sdk-core/middleware), so you keep your provider, your model configuration, and the familiar `generateText` / `streamText` API.

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

console.log(await researchAgent('Why did the DBOS integration cross the road?'));
```

## Installation

```sh
npm install @dbos-inc/vercel-ai @dbos-inc/dbos-sdk ai
```

Requires `ai` v7+ and a Postgres database for DBOS.

## How it works

`durableCalls()` returns AI SDK language-model middleware that intercepts `doGenerate` and `doStream`. Inside a DBOS workflow, each model call runs through [`DBOS.runStep`](https://docs.dbos.dev/typescript/tutorials/step-tutorial):

- On first execution, the model is called and the complete result (content, usage, finish reason, response metadata) is checkpointed in the DBOS system database.
- If the workflow is interrupted and recovered, checkpointed calls return their recorded results without contacting the model, and execution resumes from the first incomplete step.
- Outside a workflow (or inside another step), the middleware calls the model directly with no checkpointing, so the same wrapped model works anywhere in your app.

All DBOS step options are accepted and apply per model call:

```ts
durableCalls({
  retriesAllowed: true,   // retry failed model calls (default: false)
  maxAttempts: 5,         // total attempts when retries are allowed (default: 3)
  intervalSeconds: 1,     // delay before first retry (default: 1)
  backoffRate: 2,         // exponential backoff multiplier (default: 2)
  timeoutMS: 60000,       // per-attempt timeout
  name: 'my-model-call',  // step name (default: "<provider>.<modelId>.<operation>")
});
```

## Streaming

`streamText` works inside workflows. On first execution, stream parts are passed through to your code live as the model produces them, and the assembled result is checkpointed when the stream completes. On recovery, the checkpointed result is replayed as a short synthetic stream (one delta per text block), so your workflow code runs identically either way.

```ts
const streamingAgent = DBOS.registerWorkflow(async (prompt: string) => {
  const result = streamText({ model, prompt });
  for await (const delta of result.textStream) {
    process.stdout.write(delta);
  }
  return await result.text;
}, { name: 'streamingAgent' });
```

Two notes on streaming:

- **Consuming tokens from another process.** If the workflow runs on a queue worker but you want to stream tokens to a browser, set `streamKey`. Every raw stream part is then also written to a [DBOS workflow stream](https://docs.dbos.dev/typescript/tutorials/workflow-tutorial#workflow-streaming) that any process can read:

  ```ts
  const model = wrapLanguageModel({
    model: openai('gpt-5'),
    middleware: durableCalls({ streamKey: 'llm-stream' }),
  });

  // In an HTTP handler, possibly in a different process:
  for await (const part of DBOS.readStream(workflowID, 'llm-stream')) {
    // forward text-delta parts to the client
  }
  ```

- **Retries and streaming.** When `retriesAllowed` is set, live pass-through is disabled for streaming calls: a failed attempt may already have produced partial output, so parts are instead emitted all at once after an attempt succeeds. Non-streaming calls are unaffected.

## Tools

Model calls in a tool-calling loop are each checkpointed individually, so a recovered agent resumes mid-loop. Make tool side effects durable by running them as steps:

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

## Embeddings

`durableEmbeddingCalls` does the same for embedding models:

```ts
import { embedMany, wrapEmbeddingModel } from 'ai';
import { durableEmbeddingCalls } from '@dbos-inc/vercel-ai';

const embeddingModel = wrapEmbeddingModel({
  model: openai.textEmbeddingModel('text-embedding-3-small'),
  middleware: durableEmbeddingCalls({ retriesAllowed: true }),
});

const { embeddings } = await embedMany({ model: embeddingModel, values: chunks });
```

## Serialization

DBOS checkpoints step results with a superjson-based serializer, so `Date`, `URL`, `Map`, `Set`, and `Buffer` values in model responses survive recovery intact (e.g., `response.timestamp` stays a `Date`). Binary file content generated by models (`Uint8Array`) is transparently converted to base64 — a representation the AI SDK accepts natively — before checkpointing.

## Development

Tests require a local Postgres database (set `DBOS_TEST_DB_URL` to override the default `postgresql://postgres@localhost:5432/dbos_vercel_ai_test_dbos_sys`):

```sh
npm install
npm test
```
