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

Note that the AI SDK has its own retry layer (`maxRetries` on `generateText` et al., default 2), which composes multiplicatively with DBOS step retries — each AI SDK retry is a fresh step. If you enable DBOS retries, consider passing `maxRetries: 0` to the AI SDK call so retry behavior is governed in one place.

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

**Retries and streaming.** When `retriesAllowed` is set, live pass-through is disabled for streaming calls: a failed attempt may already have produced partial output, so parts are instead emitted all at once after an attempt succeeds. Non-streaming calls are unaffected.

To stream tokens to another process (e.g. the workflow runs on a queue worker and an HTTP handler streams to a browser), write parts to a [DBOS workflow stream](https://docs.dbos.dev/typescript/tutorials/workflow-tutorial#workflow-streaming) from your own consumer loop and read them elsewhere with `DBOS.readStream`.

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

## Concurrency

Run **one durable model call at a time within a single workflow**. DBOS derives each step's replay identity from the order steps are reached, but the AI SDK issues concurrent model calls in a nondeterministic order — so on recovery a checkpoint could be bound to the wrong call, silently returning one call's result for another. To prevent this, the middleware throws if it detects a second durable model call starting while one is already in flight in the same workflow. This covers `Promise.all` over `generateText`/`streamText`, `embedMany` on inputs larger than the model's per-call limit (which the SDK batches in parallel — see [Embeddings](#embeddings) for the `maxParallelCalls: 1` remedy), and parallel tool calls that themselves invoke models.

Sequential calls — including a normal tool-calling loop, where each model call completes before the next begins — are unaffected.

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

## Serialization

DBOS checkpoints step results with a superjson-based serializer, so `Date`, `URL`, `Map`, `Set`, and `Buffer` values in model responses survive recovery intact (e.g., `response.timestamp` stays a `Date`). Binary file content generated by models (`Uint8Array`) is transparently converted to base64 — a representation the AI SDK accepts natively — before checkpointing.

## Development

See [DEVELOPING.md](./DEVELOPING.md) for building, testing (requires a local Postgres database), and the release process.
