import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { after, before, test } from 'node:test';
import { DBOS } from '@dbos-inc/dbos-sdk';
import { APICallError } from '@ai-sdk/provider';
import { GatewayRateLimitError } from '@ai-sdk/gateway';
import { Client as PgClient } from 'pg';
import {
  asSchema,
  embed,
  embedMany,
  generateImage,
  generateText,
  stepCountIs,
  streamText,
  tool,
  wrapEmbeddingModel,
  wrapImageModel,
  wrapLanguageModel,
} from 'ai';
import { z } from 'zod';
import {
  durableCalls,
  durableEmbeddingCalls,
  durableImageCalls,
  durableMCPTools,
  type MCPClientLike,
} from '../src/index.js';
import { restoreAISDKErrorIdentity } from '../src/internal.js';
import {
  contentResponse,
  IMAGE_BYTES,
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
  MockLateAbortStreamModel,
  MockMCPClient,
  RichMockMCPClient,
  textResponse,
  textResponseNoMetadata,
  textResponseNullMetadata,
  textStreamParts,
  textStreamPartsNoMetadata,
  toolCallResponse,
  toolCallsResponse,
  usage,
} from './mock-models.mjs';

const systemDatabaseUrl =
  process.env.DBOS_TEST_DB_URL ?? 'postgresql://postgres@localhost:5432/dbos_vercel_ai_test_dbos_sys';

// Models and workflows are set up at module load, before DBOS.launch().

const generateMock = new MockLanguageModel();
const generateModel = wrapLanguageModel({ model: generateMock, middleware: durableCalls() });

const generateWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const result = await generateText({ model: generateModel, prompt });
    // A second step so forkWorkflow can start after the model-call checkpoint.
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return {
      text: result.text,
      timestamp: result.response.timestamp,
      inputTokens: result.usage.inputTokens ?? 0,
    };
  },
  { name: 'generateWorkflow' },
);

const toolMock = new MockLanguageModel();
const toolModel = wrapLanguageModel({ model: toolMock, middleware: durableCalls() });
let toolExecutions = 0;

const toolWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const result = await generateText({
      model: toolModel,
      prompt,
      tools: {
        getWeather: tool({
          description: 'Get the weather for a city',
          inputSchema: z.object({ city: z.string() }),
          execute: async ({ city }) =>
            DBOS.runStep(
              async () => {
                toolExecutions++;
                return `sunny in ${city}`;
              },
              { name: 'getWeather' },
            ),
        }),
      },
      stopWhen: stepCountIs(5),
    });
    return result.text;
  },
  { name: 'toolWorkflow' },
);

const streamMock = new MockLanguageModel();
const streamModel = wrapLanguageModel({ model: streamMock, middleware: durableCalls() });

const streamWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const result = streamText({ model: streamModel, prompt });
    const deltas: string[] = [];
    for await (const delta of result.textStream) {
      deltas.push(delta);
    }
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return { deltas, text: await result.text, finishReason: await result.finishReason };
  },
  { name: 'streamWorkflow' },
);

const streamToolMock = new MockLanguageModel();
const streamToolModel = wrapLanguageModel({ model: streamToolMock, middleware: durableCalls() });
let streamToolExecutions = 0;

const streamToolWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const result = streamText({
      model: streamToolModel,
      prompt,
      tools: {
        getWeather: tool({
          description: 'Get the weather for a city',
          inputSchema: z.object({ city: z.string() }),
          execute: async ({ city }) =>
            DBOS.runStep(
              async () => {
                streamToolExecutions++;
                return `rainy in ${city}`;
              },
              { name: 'getWeather' },
            ),
        }),
      },
      stopWhen: stepCountIs(5),
    });
    const text = await result.text;
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return text;
  },
  { name: 'streamToolWorkflow' },
);

const embedMock = new MockEmbeddingModel();
const embedModel = wrapEmbeddingModel({ model: embedMock, middleware: durableEmbeddingCalls() });

const embedWorkflow = DBOS.registerWorkflow(
  async (values: string[]) => {
    const result = await embedMany({ model: embedModel, values });
    return { count: result.embeddings.length, first: result.embeddings[0] };
  },
  { name: 'embedWorkflow' },
);

const embedReplayMock = new MockEmbeddingModel();
const embedReplayModel = wrapEmbeddingModel({ model: embedReplayMock, middleware: durableEmbeddingCalls() });

const embedReplayWorkflow = DBOS.registerWorkflow(
  async (values: string[]) => {
    const result = await embedMany({ model: embedReplayModel, values });
    await DBOS.runStep(async () => 'noop', { name: 'noop' }); // trailing step to fork past (replays the embed step)
    return { count: result.embeddings.length, first: result.embeddings[0] };
  },
  { name: 'embedReplayWorkflow' },
);

// Finite per-call limit makes embedMany split inputs into batches (parallel by default).
const batchEmbedMock = new MockEmbeddingModel(2);
const batchEmbedModel = wrapEmbeddingModel({ model: batchEmbedMock, middleware: durableEmbeddingCalls() });

const parallelEmbedWorkflow = DBOS.registerWorkflow(
  async (values: string[]) => (await embedMany({ model: batchEmbedModel, values })).embeddings.length,
  { name: 'parallelEmbedWorkflow' },
);

const serialEmbedWorkflow = DBOS.registerWorkflow(
  async (values: string[]) => {
    const result = await embedMany({ model: batchEmbedModel, values, maxParallelCalls: 1 });
    return { count: result.embeddings.length, first: result.embeddings[0] };
  },
  { name: 'serialEmbedWorkflow' },
);

const retryMock = new MockLanguageModel();
const retryModel = wrapLanguageModel({
  model: retryMock,
  middleware: durableCalls({ retriesAllowed: true, maxAttempts: 3, intervalSeconds: 0 }),
});

const retryWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    // maxRetries: 0 disables the AI SDK's own retry layer so the test observes DBOS step retries in isolation.
    const result = await generateText({ model: retryModel, prompt, maxRetries: 0 });
    return result.text;
  },
  { name: 'retryWorkflow' },
);

const errorMock = new MockLanguageModel();
const errorModel = wrapLanguageModel({ model: errorMock, middleware: durableCalls() });

const errorWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const result = await generateText({ model: errorModel, prompt, maxRetries: 0 });
    return result.text;
  },
  { name: 'errorWorkflow' },
);

// retriesAllowed: false delegates retries to the AI SDK, so a transient error checkpoints (rather than being
// absorbed inside the step) and the AI SDK's own retry — which keys off APICallError.isInstance — runs across it.
const identityMock = new MockLanguageModel();
const identityModel = wrapLanguageModel({ model: identityMock, middleware: durableCalls({ retriesAllowed: false }) });

const identityWorkflow = DBOS.registerWorkflow(
  // No maxRetries: 0 here, so the AI SDK's retry layer is active.
  async () => (await generateText({ model: identityModel, prompt: 'hi' })).text,
  { name: 'identityWorkflow' },
);

// GatewayError (from the Vercel AI Gateway) is the OTHER half of the AI SDK's retry predicate; its markers use a
// different namespace than APICallError, so restoreAISDKErrorIdentity must handle it too.
const gatewayMock = new MockLanguageModel();
const gatewayModel = wrapLanguageModel({ model: gatewayMock, middleware: durableCalls({ retriesAllowed: false }) });
const gatewayWorkflow = DBOS.registerWorkflow(
  async () => (await generateText({ model: gatewayModel, prompt: 'hi' })).text,
  { name: 'gatewayWorkflow' },
);

// Embed and image share generate's APICallError-keyed retry, so their restore wiring needs the same replay coverage.
const embedIdentityMock = new MockEmbeddingModel();
const embedIdentityModel = wrapEmbeddingModel({ model: embedIdentityMock, middleware: durableEmbeddingCalls({ retriesAllowed: false }) });
const embedIdentityWorkflow = DBOS.registerWorkflow(
  async () => (await embed({ model: embedIdentityModel, value: 'hi' })).embedding.length,
  { name: 'embedIdentityWorkflow' },
);

const imageIdentityMock = new MockImageModel();
const imageIdentityModel = wrapImageModel({ model: imageIdentityMock, middleware: durableImageCalls({ retriesAllowed: false }) });
const imageIdentityWorkflow = DBOS.registerWorkflow(
  async () => (await generateImage({ model: imageIdentityModel, prompt: 'draw' })).images.length,
  { name: 'imageIdentityWorkflow' },
);

// Providers that omit response metadata: the middleware must give the response a durable id/timestamp so the AI
// SDK's generateId()/new Date() fallback (which runs outside the step) doesn't produce a fresh value on replay.
const idGenMock = new MockLanguageModel();
const idGenModel = wrapLanguageModel({ model: idGenMock, middleware: durableCalls() });
const idGenWorkflow = DBOS.registerWorkflow(
  async () => {
    const result = await generateText({ model: idGenModel, prompt: 'hi', maxRetries: 0 });
    await DBOS.runStep(async () => 'noop', { name: 'noop' }); // so forkWorkflow can start after the model step
    return { id: result.response.id, timestamp: result.response.timestamp.toISOString() };
  },
  { name: 'idGenWorkflow' },
);

const idStreamMock = new MockLanguageModel();
const idStreamModel = wrapLanguageModel({ model: idStreamMock, middleware: durableCalls() });
const idStreamWorkflow = DBOS.registerWorkflow(
  async () => {
    const result = streamText({ model: idStreamModel, prompt: 'hi', maxRetries: 0 });
    await result.consumeStream();
    const response = await result.response;
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return { id: response.id, timestamp: response.timestamp.toISOString() };
  },
  { name: 'idStreamWorkflow' },
);

const fileMock = new MockLanguageModel();
const fileModel = wrapLanguageModel({ model: fileMock, middleware: durableCalls() });

const fileWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const result = await generateText({ model: fileModel, prompt });
    const file = result.files[0]!;
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return { text: result.text, mediaType: file.mediaType, bytes: Array.from(file.uint8Array) };
  },
  { name: 'fileWorkflow' },
);

const reasoningMock = new MockLanguageModel();
const reasoningModel = wrapLanguageModel({ model: reasoningMock, middleware: durableCalls() });

const reasoningWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const result = streamText({ model: reasoningModel, prompt });
    const text = await result.text;
    const reasoning = await result.reasoningText;
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return { text, reasoning };
  },
  { name: 'reasoningWorkflow' },
);

const retryStreamMock = new MockLanguageModel();
const retryStreamModel = wrapLanguageModel({
  model: retryStreamMock,
  middleware: durableCalls({ maxAttempts: 3, intervalSeconds: 0 }),
});

const retryStreamWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const result = streamText({ model: retryStreamModel, prompt });
    const deltas: string[] = [];
    for await (const delta of result.textStream) {
      deltas.push(delta);
    }
    return { deltas, text: await result.text };
  },
  { name: 'retryStreamWorkflow' },
);

const userPrompt = [{ role: 'user' as const, content: [{ type: 'text' as const, text: 'hi' }] }];

const failStreamMock = new MockLanguageModel();
const failStreamModel = wrapLanguageModel({ model: failStreamMock, middleware: durableCalls() });

const failStreamWorkflow = DBOS.registerWorkflow(
  async () => {
    const streamResult = await failStreamModel.doStream({ prompt: userPrompt });
    const reader = streamResult.stream.getReader();
    for (;;) {
      const { done } = await reader.read();
      if (done) break;
    }
    return 'unreachable';
  },
  { name: 'failStreamWorkflow' },
);

const cancelMock = new MockLanguageModel();
const cancelModel = wrapLanguageModel({ model: cancelMock, middleware: durableCalls() });

const cancelWorkflow = DBOS.registerWorkflow(
  async () => {
    const streamResult = await cancelModel.doStream({ prompt: userPrompt });
    const reader = streamResult.stream.getReader();
    await reader.read();
    await reader.read();
    // cancel() awaits the in-flight step, so the model call is checkpointed before the workflow proceeds (no sleep needed).
    await reader.cancel();
    return 'cancelled early';
  },
  { name: 'cancelWorkflow' },
);

const cancelErrorMock = new MockLanguageModel();
const cancelErrorModel = wrapLanguageModel({ model: cancelErrorMock, middleware: durableCalls() });

// Reads two parts, cancels, then the model stream errors. The step must record a success (not the
// post-cancel error), so a fork replays identically instead of failing where the live run succeeded.
const cancelErrorWorkflow = DBOS.registerWorkflow(
  async () => {
    const streamResult = await cancelErrorModel.doStream({ prompt: userPrompt });
    const reader = streamResult.stream.getReader();
    await reader.read();
    await reader.read();
    await reader.cancel();
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return 'cancelled before error';
  },
  { name: 'cancelErrorWorkflow' },
);

const cancelRejectMock = new MockLanguageModel();
const cancelRejectModel = wrapLanguageModel({ model: cancelRejectMock, middleware: durableCalls() });

// Like cancelErrorWorkflow, but the model stream rejects (stream-level failure) after the cancel instead of sending an error part.
const cancelRejectWorkflow = DBOS.registerWorkflow(
  async () => {
    const streamResult = await cancelRejectModel.doStream({ prompt: userPrompt });
    const reader = streamResult.stream.getReader();
    await reader.read();
    await reader.read();
    await reader.cancel();
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return 'cancelled before rejection';
  },
  { name: 'cancelRejectWorkflow' },
);

const earlyCancelMock = new MockLanguageModel();
const earlyCancelModel = wrapLanguageModel({ model: earlyCancelMock, middleware: durableCalls() });

// Cancels before the model call settles; a later doStream rejection must not become the step outcome.
const earlyCancelWorkflow = DBOS.registerWorkflow(
  async () => {
    const streamResult = await earlyCancelModel.doStream({ prompt: userPrompt });
    await streamResult.stream.cancel();
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return 'cancelled immediately';
  },
  { name: 'earlyCancelWorkflow' },
);

const recoveryMock = new MockLanguageModel();
const recoveryModel = wrapLanguageModel({ model: recoveryMock, middleware: durableCalls() });

const recoveryWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const result = await generateText({ model: recoveryModel, prompt });
    const go = await DBOS.recv<string>('go', 30);
    return { text: result.text, go };
  },
  { name: 'recoveryWorkflow' },
);

const concurrentMock = new MockLanguageModel();
const concurrentModel = wrapLanguageModel({ model: concurrentMock, middleware: durableCalls() });

const concurrentWorkflow = DBOS.registerWorkflow(
  async () => {
    const [a, b] = await Promise.all([
      generateText({ model: concurrentModel, prompt: 'A' }),
      generateText({ model: concurrentModel, prompt: 'B' }),
    ]);
    return [a.text, b.text];
  },
  { name: 'concurrentWorkflow' },
);

// Streams a source part while a text block is still open, to check the accumulator keeps arrival order ([text, source]).
const orderingMock = new MockLanguageModel();
const orderingModel = wrapLanguageModel({ model: orderingMock, middleware: durableCalls() });

const orderingWorkflow = DBOS.registerWorkflow(
  async () => {
    const result = streamText({ model: orderingModel, prompt: 'hi' });
    const types: string[] = [];
    for (const part of await result.content) {
      types.push(part.type);
    }
    await DBOS.runStep(async () => 'noop', { name: 'noop' }); // trailing step so a fork replays the stream from its checkpoint
    return { types, text: await result.text };
  },
  { name: 'orderingWorkflow' },
);

const imageMock = new MockImageModel();
const imageModel = wrapImageModel({ model: imageMock, middleware: durableImageCalls() });

const imageWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const { images } = await generateImage({ model: imageModel, prompt });
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return { count: images.length, bytes: Array.from(images[0]!.uint8Array) };
  },
  { name: 'imageWorkflow' },
);

// n > maxImagesPerCall (1) makes generateImage dispatch parallel batches; the last byte of each image is its call ordinal.
const multiImageMock = new MockImageModel();
const multiImageModel = wrapImageModel({ model: multiImageMock, middleware: durableImageCalls() });

const multiImageWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const { images } = await generateImage({ model: multiImageModel, prompt, n: 3 });
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return images.map((img) => img.uint8Array.at(-1)!);
  },
  { name: 'multiImageWorkflow' },
);

const nonRetryMock = new MockLanguageModel();
const nonRetryModel = wrapLanguageModel({
  model: nonRetryMock,
  middleware: durableCalls({ retriesAllowed: true, maxAttempts: 5, intervalSeconds: 0 }),
});

const nonRetryWorkflow = DBOS.registerWorkflow(
  async () => (await generateText({ model: nonRetryModel, prompt: 'hi', maxRetries: 0 })).text,
  { name: 'nonRetryWorkflow' },
);

const abortMock = new MockLanguageModel();
const abortModel = wrapLanguageModel({
  model: abortMock,
  middleware: durableCalls({ maxAttempts: 5, intervalSeconds: 0 }),
});
const abortWorkflow = DBOS.registerWorkflow(
  async () => (await generateText({ model: abortModel, prompt: 'hi', maxRetries: 0 })).text,
  { name: 'abortWorkflow' },
);

const mcpToolMock = new MockLanguageModel();
const mcpToolModel = wrapLanguageModel({ model: mcpToolMock, middleware: durableCalls() });
const mcpClient = new MockMCPClient();

const mcpWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const tools = await durableMCPTools(mcpClient);
    const result = await generateText({ model: mcpToolModel, prompt, tools, stopWhen: stepCountIs(5), maxRetries: 0 });
    await DBOS.runStep(async () => 'noop', { name: 'noop' }); // trailing step to fork past (replays all model/tool steps)
    return result.text;
  },
  { name: 'mcpWorkflow' },
);

const parallelMcpMock = new MockLanguageModel();
const parallelMcpModel = wrapLanguageModel({ model: parallelMcpMock, middleware: durableCalls() });
const parallelMcpClient = new MockMCPClient();

const parallelMcpWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const tools = await durableMCPTools(parallelMcpClient);
    const result = await generateText({ model: parallelMcpModel, prompt, tools, stopWhen: stepCountIs(5), maxRetries: 0 });
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return result.text;
  },
  { name: 'parallelMcpWorkflow' },
);

const subsetMcpMock = new MockLanguageModel();
const subsetMcpModel = wrapLanguageModel({ model: subsetMcpMock, middleware: durableCalls() });
const subsetMcpClient = new MockMCPClient();
const subsetSchemas = { schemas: { getWeather: {} } };

const subsetMcpWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const tools = await durableMCPTools(subsetMcpClient, { toolOptions: subsetSchemas });
    const result = await generateText({ model: subsetMcpModel, prompt, tools, stopWhen: stepCountIs(5), maxRetries: 0 });
    return { text: result.text, toolNames: Object.keys(tools) };
  },
  { name: 'subsetMcpWorkflow' },
);

const richMcpMock = new MockLanguageModel();
const richMcpModel = wrapLanguageModel({ model: richMcpMock, middleware: durableCalls() });
const richMcpClient = new RichMockMCPClient();

const richMcpWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const tools = await durableMCPTools(richMcpClient);
    const result = await generateText({ model: richMcpModel, prompt, tools, stopWhen: stepCountIs(5), maxRetries: 0 });
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    const screenshot = tools.screenshot!;
    return {
      text: result.text,
      title: screenshot.title,
      metadata: screenshot.metadata,
      meta: (screenshot as { _meta?: unknown })._meta,
      converts: typeof screenshot.toModelOutput === 'function',
    };
  },
  { name: 'richMcpWorkflow' },
);

// #4: an explicit `shouldRetry: undefined` must still fall back to the default classification.
const undefinedRetryMock = new MockLanguageModel();
const undefinedRetryModel = wrapLanguageModel({
  model: undefinedRetryMock,
  middleware: durableCalls({ retriesAllowed: true, maxAttempts: 5, intervalSeconds: 0, shouldRetry: undefined }),
});
const undefinedRetryWorkflow = DBOS.registerWorkflow(
  async () => (await generateText({ model: undefinedRetryModel, prompt: 'hi', maxRetries: 0 })).text,
  { name: 'undefinedRetryWorkflow' },
);

// A caller-provided shouldRetry must win over the default (here: never retry, even a retryable error).
const overrideRetryMock = new MockLanguageModel();
const overrideRetryModel = wrapLanguageModel({
  model: overrideRetryMock,
  middleware: durableCalls({ retriesAllowed: true, maxAttempts: 3, intervalSeconds: 0, shouldRetry: () => false }),
});
const overrideRetryWorkflow = DBOS.registerWorkflow(
  async () => (await generateText({ model: overrideRetryModel, prompt: 'hi', maxRetries: 0 })).text,
  { name: 'overrideRetryWorkflow' },
);

// #7: an MCP tool whose schema exposes its JSON Schema asynchronously (a PromiseLike jsonSchema).
const asyncSchemaClient = {
  async tools() {
    return {
      ping: {
        description: 'ping a host',
        inputSchema: {
          [Symbol.for('vercel.ai.schema')]: true,
          jsonSchema: Promise.resolve({ type: 'object', properties: { host: { type: 'string' } }, required: ['host'] }),
          validate: undefined,
        },
        execute: async () => 'pong',
      },
    };
  },
  async close() {},
} as unknown as MCPClientLike;

const asyncSchemaWorkflow = DBOS.registerWorkflow(
  async () => {
    const tools = await durableMCPTools(asyncSchemaClient);
    return await asSchema((tools.ping as { inputSchema: Parameters<typeof asSchema>[0] }).inputSchema).jsonSchema;
  },
  { name: 'asyncSchemaWorkflow' },
);

// A structured (non-Error) error-part payload must keep its JSON message and isRetryable classification.
const structuredErrorMock = new MockLanguageModel();
const structuredErrorModel = wrapLanguageModel({
  model: structuredErrorMock,
  middleware: durableCalls({ maxAttempts: 3, intervalSeconds: 0 }),
});
const structuredErrorWorkflow = DBOS.registerWorkflow(
  async () => {
    const streamResult = await structuredErrorModel.doStream({ prompt: userPrompt });
    const reader = streamResult.stream.getReader();
    for (;;) {
      const { done } = await reader.read();
      if (done) break;
    }
    return 'unreachable';
  },
  { name: 'structuredErrorWorkflow' },
);

const abortPartMock = new MockLanguageModel();
const abortPartModel = wrapLanguageModel({
  model: abortPartMock,
  middleware: durableCalls({ maxAttempts: 3, intervalSeconds: 0 }),
});
const abortPartWorkflow = DBOS.registerWorkflow(
  async () => {
    const streamResult = await abortPartModel.doStream({ prompt: userPrompt });
    const reader = streamResult.stream.getReader();
    for (;;) {
      const { done } = await reader.read();
      if (done) break;
    }
    return 'unreachable';
  },
  { name: 'abortPartWorkflow' },
);

// A stream ending with no finish part and no output must fail the attempt (retryably), not checkpoint an empty success.
const emptyStreamMock = new MockLanguageModel();
const emptyStreamModel = wrapLanguageModel({
  model: emptyStreamMock,
  middleware: durableCalls({ maxAttempts: 3, intervalSeconds: 0 }),
});
const emptyStreamWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const result = streamText({ model: emptyStreamModel, prompt, maxRetries: 0 });
    const deltas: string[] = [];
    for await (const delta of result.textStream) {
      deltas.push(delta);
    }
    return { deltas, text: await result.text };
  },
  { name: 'emptyStreamWorkflow' },
);

// Collects raw stream parts so replay-grammar tests can inspect exactly what the middleware emits.
const replayPartsMock = new MockLanguageModel();
const replayPartsModel = wrapLanguageModel({ model: replayPartsMock, middleware: durableCalls() });
type CollectedPart = { type: string; id?: string; toolName?: string; delta?: string; providerMetadata?: unknown };
const replayPartsWorkflow = DBOS.registerWorkflow(
  async () => {
    const streamResult = await replayPartsModel.doStream({ prompt: userPrompt });
    const reader = streamResult.stream.getReader();
    const parts: CollectedPart[] = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      const part = value as CollectedPart;
      parts.push({
        type: part.type,
        id: part.id,
        toolName: part.toolName,
        delta: part.delta,
        providerMetadata: part.providerMetadata,
      });
    }
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return parts;
  },
  { name: 'replayPartsWorkflow' },
);

// Split response-metadata parts must merge per-field (like the AI SDK), not clobber earlier fields.
const metadataMock = new MockLanguageModel();
const metadataModel = wrapLanguageModel({ model: metadataMock, middleware: durableCalls() });
const metadataWorkflow = DBOS.registerWorkflow(
  async () => {
    const result = streamText({ model: metadataModel, prompt: 'hi', maxRetries: 0 });
    const text = await result.text;
    const response = await result.response;
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return { id: response.id, modelId: response.modelId, text };
  },
  { name: 'metadataWorkflow' },
);

// maxImagesPerCall 2 keeps n=2 in one call, so a single (spec-violating) mixed string/bytes batch reaches the encoder.
const mixedImageMock = new MockImageModel(2);
const mixedImageModel = wrapImageModel({ model: mixedImageMock, middleware: durableImageCalls() });
const mixedImageWorkflow = DBOS.registerWorkflow(
  async () => {
    const { images } = await generateImage({ model: mixedImageModel, prompt: 'mixed', n: 2 });
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return images.map((image) => Array.from(image.uint8Array));
  },
  { name: 'mixedImageWorkflow' },
);

// An MCP tool whose execute streams (returns an AsyncIterable): the checkpoint must record the final value.
const streamingToolClient = {
  async tools() {
    return {
      countdown: {
        description: 'count down and lift off',
        inputSchema: {
          [Symbol.for('vercel.ai.schema')]: true,
          jsonSchema: { type: 'object', properties: {}, additionalProperties: false },
          validate: undefined,
        },
        execute: async function* () {
          yield { status: 'counting' };
          yield 'lift off';
        },
      },
    };
  },
  async close() {},
} as unknown as MCPClientLike;

const streamingToolMock = new MockLanguageModel();
const streamingToolModel = wrapLanguageModel({ model: streamingToolMock, middleware: durableCalls() });
const streamingToolWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const tools = await durableMCPTools(streamingToolClient);
    const result = await generateText({
      model: streamingToolModel,
      prompt,
      tools,
      stopWhen: stepCountIs(5),
      maxRetries: 0,
    });
    return result.text;
  },
  { name: 'streamingToolWorkflow' },
);

// Aborting mid-stream must checkpoint the partial output as a success, so recovery replays the graceful abort.
const abortStreamMock = new MockLanguageModel();
const abortStreamModel = wrapLanguageModel({ model: abortStreamMock, middleware: durableCalls() });
const abortStreamWorkflow = DBOS.registerWorkflow(
  async () => {
    const abortController = new AbortController();
    const result = streamText({ model: abortStreamModel, prompt: 'hi', abortSignal: abortController.signal, maxRetries: 0 });
    const deltas: string[] = [];
    for await (const delta of result.textStream) {
      deltas.push(delta);
      if (deltas.length === 2) abortController.abort();
    }
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return deltas;
  },
  { name: 'abortStreamWorkflow' },
);

// Issue #5: an abort detaches the consumer, but the durable model call must still record the complete response —
// otherwise recovery replays the truncated answer as if it were the whole one. The gate parks the provider
// mid-stream so the abort lands at a deterministic split point (no timing race).
const abortRecoveryMock = new MockLanguageModel();
const abortRecoveryModel = wrapLanguageModel({ model: abortRecoveryMock, middleware: durableCalls() });
const newGate = () => {
  let release!: () => void;
  const promise = new Promise<void>((resolve) => (release = resolve));
  return { promise, release };
};
let abortRecoveryGate = newGate();
let abortRecoveryAborts = true;
const abortRecoveryWorkflow = DBOS.registerWorkflow(
  async () => {
    const controller = new AbortController();
    const result = streamText({
      model: abortRecoveryModel,
      prompt: 'hi',
      abortSignal: controller.signal,
      maxRetries: 0,
    });
    let text = '';
    for await (const delta of result.textStream) {
      text += delta;
      // Abort while the provider is parked at the gate, then release it: the model has more to send either way.
      if (abortRecoveryAborts) controller.abort();
      abortRecoveryGate.release();
    }
    await DBOS.runStep(async () => 'noop', { name: 'noop' });
    return text;
  },
  { name: 'abortRecoveryWorkflow' },
);

// A follow-up durable call after aborting a stream must not be rejected as concurrent: the stream step is
// already sequenced (its funcID is assigned), so the follow-up is deterministic on replay even while the
// aborted stream step drains and checkpoints in the background.
const guardRaceMock = new MockLateAbortStreamModel();
const guardRaceModel = wrapLanguageModel({ model: guardRaceMock, middleware: durableCalls() });
const guardRaceWorkflow = DBOS.registerWorkflow(
  async () => {
    const controller = new AbortController();
    const result = streamText({ model: guardRaceModel, prompt: 'hi', abortSignal: controller.signal, maxRetries: 0 });
    try {
      for await (const _delta of result.textStream) {
        controller.abort(); // stop after the first delta
      }
    } catch {
      /* aborted */
    }
    // Fires while the aborted stream step is still checkpointing; before the fix this tripped the in-flight guard.
    const follow = await generateText({ model: guardRaceModel, prompt: 'summarize', maxRetries: 0 });
    return follow.text;
  },
  { name: 'guardRaceWorkflow' },
);

// A timed-out (abandoned) stream attempt must stop reading/emitting so it can't interleave with its retry.
const timeoutStreamMock = new MockLanguageModel();
const timeoutStreamModel = wrapLanguageModel({
  model: timeoutStreamMock,
  middleware: durableCalls({ timeoutMS: 100, maxAttempts: 2, intervalSeconds: 0 }),
});
const timeoutLiveDeltas: string[] = [];
const timeoutStreamWorkflow = DBOS.registerWorkflow(
  async () => {
    const result = streamText({ model: timeoutStreamModel, prompt: 'hi', maxRetries: 0 });
    for await (const delta of result.textStream) {
      timeoutLiveDeltas.push(delta);
    }
    return { text: await result.text };
  },
  { name: 'timeoutStreamWorkflow' },
);

// An aborted MCP tool call must checkpoint its real error (never a TypeError from mutating a getter-only
// DOMException message) and must not be retried, since the consumer is already gone.
let slowToolStarted: (() => void) | undefined;
let slowToolExecutions = 0;

// Builds an MCP client whose one tool hangs until aborted, then rejects with rejectWith(signal).
const makeAbortToolClient = (rejectWith: (signal: AbortSignal) => unknown): MCPClientLike =>
  ({
    async tools() {
      return {
        slowTool: {
          description: 'a slow tool',
          inputSchema: {
            [Symbol.for('vercel.ai.schema')]: true,
            jsonSchema: { type: 'object', properties: {} },
            validate: undefined,
          },
          execute: (_input: unknown, options: { abortSignal?: AbortSignal }) =>
            new Promise((_resolve, reject) => {
              slowToolExecutions++;
              slowToolStarted?.();
              options.abortSignal?.addEventListener('abort', () => reject(rejectWith(options.abortSignal!)));
            }),
        },
      };
    },
    async close() {},
  }) as unknown as MCPClientLike;

let toolAbort: (() => void) | undefined;

// A tool aborted with a real DOMException reason — signal.reason has a getter-only `message`, which the
// removed marker-tagging path mutated and threw TypeError on. C1 regression.
const domAbortMock = new MockLanguageModel();
const domAbortModel = wrapLanguageModel({ model: domAbortMock, middleware: durableCalls() });
const domAbortClient = makeAbortToolClient((signal) => signal.reason);
const domAbortWorkflow = DBOS.registerWorkflow(
  async () => {
    const controller = new AbortController();
    toolAbort = () => controller.abort();
    const tools = await durableMCPTools(domAbortClient);
    try {
      const result = await generateText({
        model: domAbortModel,
        prompt: 'hi',
        tools,
        abortSignal: controller.signal,
        stopWhen: stepCountIs(5),
        maxRetries: 0,
      });
      return result.text;
    } catch (error) {
      return `caught:${(error as Error).name}`;
    }
  },
  { name: 'domAbortWorkflow' },
);

// A tool aborted with a GENERIC error (name "Error", no isRetryable): the old classifier keyed on the name
// and retried it; the fix declines a retry whenever the signal is aborted. #9 regression.
const genericAbortMock = new MockLanguageModel();
const genericAbortModel = wrapLanguageModel({ model: genericAbortMock, middleware: durableCalls() });
const genericAbortClient = makeAbortToolClient(() => new Error('connection reset by peer'));
const genericAbortWorkflow = DBOS.registerWorkflow(
  async () => {
    const controller = new AbortController();
    toolAbort = () => controller.abort();
    const tools = await durableMCPTools(genericAbortClient, { maxAttempts: 3, intervalSeconds: 0 });
    try {
      const result = await generateText({
        model: genericAbortModel,
        prompt: 'hi',
        tools,
        abortSignal: controller.signal,
        stopWhen: stepCountIs(5),
        maxRetries: 0,
      });
      return result.text;
    } catch (error) {
      return `caught:${(error as Error).name}`;
    }
  },
  { name: 'genericAbortWorkflow' },
);

// A tool that fails generically with NO abort: the retry fix is scoped to aborts, so this must still retry.
const retryToolMock = new MockLanguageModel();
const retryToolModel = wrapLanguageModel({ model: retryToolMock, middleware: durableCalls() });
const retryToolClient = {
  async tools() {
    return {
      slowTool: {
        description: 'a failing tool',
        inputSchema: { [Symbol.for('vercel.ai.schema')]: true, jsonSchema: { type: 'object', properties: {} }, validate: undefined },
        execute: () => {
          slowToolExecutions++;
          throw new Error('connection reset by peer');
        },
      },
    };
  },
  async close() {},
} as unknown as MCPClientLike;
const retryToolWorkflow = DBOS.registerWorkflow(
  async () => {
    const tools = await durableMCPTools(retryToolClient, { maxAttempts: 3, intervalSeconds: 0 });
    const result = await generateText({
      model: retryToolModel,
      prompt: 'hi',
      tools,
      stopWhen: stepCountIs(5),
      maxRetries: 0,
    });
    return result.text;
  },
  { name: 'retryToolWorkflow' },
);

// A throwing isRetryable accessor must not replace the step's real error or disable retries.
const evilRetryMock = new MockLanguageModel();
const evilRetryModel = wrapLanguageModel({
  model: evilRetryMock,
  middleware: durableCalls({ maxAttempts: 2, intervalSeconds: 0 }),
});
const evilRetryWorkflow = DBOS.registerWorkflow(
  async () => (await generateText({ model: evilRetryModel, prompt: 'hi', maxRetries: 0 })).text,
  { name: 'evilRetryWorkflow' },
);

before(async () => {
  DBOS.setConfig({ name: 'dbos-vercel-ai-test', systemDatabaseUrl });
  await DBOS.launch();
});

after(async () => {
  await DBOS.shutdown();
});

test('generateText runs as a durable step inside a workflow', async () => {
  generateMock.generateResults.push(textResponse('Hello from DBOS'));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(generateWorkflow, { workflowID })('hi');
  const result = await handle.getResult();

  assert.equal(result.text, 'Hello from DBOS');
  assert.equal(generateMock.generateCalls, 1);
  // The step result round-trips through the DBOS serializer; Dates must survive.
  assert.ok(result.timestamp instanceof Date);
  assert.equal(result.timestamp.toISOString(), '2026-07-02T12:00:00.000Z');
  assert.equal(result.inputTokens, 10);

  const steps = await DBOS.listWorkflowSteps(workflowID);
  assert.ok(steps !== undefined);
  assert.equal(steps.length, 2);
  assert.equal(steps[0]!.name, 'mock.mock-model.generate');
});

test('replayed workflows use the checkpointed model result instead of calling the model', async () => {
  generateMock.generateResults.push(textResponse('checkpointed answer'));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(generateWorkflow, { workflowID })('hi');
  const original = await handle.getResult();
  assert.equal(generateMock.generateCalls, 2);

  // Fork after the model-call step: the workflow re-executes but the model call replays from its checkpoint (no responses queued, so a real call would throw).
  const forked = await DBOS.forkWorkflow<ReturnType<typeof generateWorkflow>>(workflowID, 1);
  const replayed = (await forked.getResult()) as Awaited<ReturnType<typeof generateWorkflow>>;

  assert.equal(replayed.text, original.text);
  assert.ok(replayed.timestamp instanceof Date);
  assert.equal(generateMock.generateCalls, 2);
});

test('tool-calling loop with a durable tool step', async () => {
  toolMock.generateResults.push(toolCallResponse('getWeather', '{"city":"Tokyo"}'), textResponse('It is sunny in Tokyo.'));
  const handle = await DBOS.startWorkflow(toolWorkflow, { workflowID: randomUUID() })('weather in Tokyo?');
  const text = await handle.getResult();

  assert.equal(text, 'It is sunny in Tokyo.');
  assert.equal(toolMock.generateCalls, 2);
  assert.equal(toolExecutions, 1);
});

test('streamText streams live parts through a durable step', async () => {
  streamMock.streamPartLists.push(textStreamParts(['Hello', ' from', ' DBOS']));
  const handle = await DBOS.startWorkflow(streamWorkflow, { workflowID: randomUUID() })('hi');
  const result = await handle.getResult();

  // Live pass-through delivers each model delta individually, not one merged block.
  assert.deepEqual(result.deltas, ['Hello', ' from', ' DBOS']);
  assert.equal(result.text, 'Hello from DBOS');
  assert.equal(result.finishReason, 'stop');
  assert.equal(streamMock.streamCalls, 1);
});

test('replayed workflows synthesize the stream from the checkpoint', async () => {
  streamMock.streamPartLists.push(textStreamParts(['A', 'B', 'C']));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(streamWorkflow, { workflowID })('hi');
  const original = await handle.getResult();
  assert.equal(original.text, 'ABC');
  assert.equal(streamMock.streamCalls, 2);

  const forked = await DBOS.forkWorkflow<ReturnType<typeof streamWorkflow>>(workflowID, 1);
  const replayed = (await forked.getResult()) as Awaited<ReturnType<typeof streamWorkflow>>;

  // The model is not called again; the recorded result is replayed as one delta.
  assert.equal(streamMock.streamCalls, 2);
  assert.equal(replayed.text, 'ABC');
  assert.deepEqual(replayed.deltas, ['ABC']);
  assert.equal(replayed.finishReason, 'stop');
});

test('streaming tool call runs as a durable step ordered after the model step, and replays without re-executing', async () => {
  streamToolMock.streamPartLists.push(
    [
      { type: 'stream-start', warnings: [] },
      { type: 'tool-input-start', id: 'call-1', toolName: 'getWeather' },
      { type: 'tool-input-delta', id: 'call-1', delta: '{"city":"Oslo"}' },
      { type: 'tool-input-end', id: 'call-1' },
      { type: 'tool-call', toolCallId: 'call-1', toolName: 'getWeather', input: '{"city":"Oslo"}' },
      { type: 'finish', finishReason: { unified: 'tool-calls', raw: undefined }, usage: usage() },
    ],
    textStreamParts(['Rainy in Oslo.']),
  );
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(streamToolWorkflow, { workflowID })('weather in Oslo?');
  assert.equal(await handle.getResult(), 'Rainy in Oslo.');
  assert.equal(streamToolExecutions, 1);
  assert.equal(streamToolMock.streamCalls, 2);

  const steps = await DBOS.listWorkflowSteps(workflowID);
  const streamStep = steps!.find((s) => s.name === 'mock.mock-model.stream')!;
  const toolStep = steps!.find((s) => s.name === 'getWeather')!;
  // The tool runs on 'finish', which is withheld until the model step is durable, so its step is ordered after it.
  assert.ok(streamStep.functionID < toolStep.functionID);

  // Fork past every model/tool step: all replay from checkpoints, so nothing is re-invoked.
  const noopStep = steps!.find((s) => s.name === 'noop')!;
  const forked = await DBOS.forkWorkflow<ReturnType<typeof streamToolWorkflow>>(workflowID, noopStep.functionID);
  assert.equal(await forked.getResult(), 'Rainy in Oslo.');
  assert.equal(streamToolExecutions, 1);
  assert.equal(streamToolMock.streamCalls, 2);
});

test('embedMany runs as a durable step inside a workflow', async () => {
  const handle = await DBOS.startWorkflow(embedWorkflow, { workflowID: randomUUID() })(['a', 'b']);
  const result = await handle.getResult();

  assert.equal(result.count, 2);
  assert.deepEqual(result.first, [0, 0.5, 0.25]);
  assert.equal(embedMock.embedCalls, 1);
});

test('embedMany checkpoints as a durable step and replays from the checkpoint without re-calling the model', async () => {
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(embedReplayWorkflow, { workflowID })(['a', 'b']);
  const original = await handle.getResult();
  assert.equal(original.count, 2);
  assert.equal(embedReplayMock.embedCalls, 1);

  // The embed call must be recorded as a durable step (not silently run live).
  const steps = await DBOS.listWorkflowSteps(workflowID);
  assert.ok(steps?.some((s) => s.name === 'mock.mock-embed.embed'), 'embedding recorded as a durable step');

  // Fork past the embed step: it replays from its checkpoint, so the model is not re-called.
  const forked = await DBOS.forkWorkflow<ReturnType<typeof embedReplayWorkflow>>(workflowID, 1);
  const replayed = (await forked.getResult()) as Awaited<ReturnType<typeof embedReplayWorkflow>>;
  assert.deepEqual(replayed, original);
  assert.equal(embedReplayMock.embedCalls, 1); // not re-called on replay
});

test('multi-batch embedMany (parallel batches) trips the guard with a remedy in the message', async () => {
  const handle = await DBOS.startWorkflow(parallelEmbedWorkflow, { workflowID: randomUUID() })(['a', 'b', 'c', 'd']);
  await assert.rejects(handle.getResult(), /Concurrent durable model calls.*maxParallelCalls: 1/s);
});

test('multi-batch embedMany with maxParallelCalls: 1 is durable and correct', async () => {
  const handle = await DBOS.startWorkflow(serialEmbedWorkflow, { workflowID: randomUUID() })(['a', 'b', 'c', 'd']);
  const result = await handle.getResult();
  assert.equal(result.count, 4);
  assert.deepEqual(result.first, [0, 0.5, 0.25]);
});

test('wrapped models work outside DBOS workflows without checkpointing', async () => {
  generateMock.generateResults.push(textResponse('outside a workflow'));
  const before = generateMock.generateCalls;
  const result = await generateText({ model: generateModel, prompt: 'hi' });
  assert.equal(result.text, 'outside a workflow');
  assert.equal(generateMock.generateCalls, before + 1);
});

test('streaming outside a workflow passes through live', async () => {
  streamMock.streamPartLists.push(textStreamParts(['no', ' workflow']));
  const before = streamMock.streamCalls;
  const result = streamText({ model: streamModel, prompt: 'hi' });
  const deltas: string[] = [];
  for await (const delta of result.textStream) {
    deltas.push(delta);
  }
  assert.deepEqual(deltas, ['no', ' workflow']);
  assert.equal(streamMock.streamCalls, before + 1);
});

test('failed model calls are retried durably until success', async () => {
  retryMock.generateResults.push(new Error('transient upstream failure'), textResponse('recovered'));
  const handle = await DBOS.startWorkflow(retryWorkflow, { workflowID: randomUUID() })('hi');
  assert.equal(await handle.getResult(), 'recovered');
  assert.equal(retryMock.generateCalls, 2);
});

test('permanent (non-retryable) model failures propagate and fail the workflow', async () => {
  errorMock.generateResults.push(Object.assign(new Error('model exploded'), { isRetryable: false }));
  const handle = await DBOS.startWorkflow(errorWorkflow, { workflowID: randomUUID() })('hi');
  await assert.rejects(handle.getResult(), /model exploded/);
  assert.equal(errorMock.generateCalls, 1);
});

test('recovery resumes mid-tool-loop without repeating completed work', async () => {
  toolMock.generateResults.push(toolCallResponse('getWeather', '{"city":"Oslo"}'), textResponse('Rainy in Oslo.'));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(toolWorkflow, { workflowID })('weather in Oslo?');
  assert.equal(await handle.getResult(), 'Rainy in Oslo.');
  const callsBefore = toolMock.generateCalls;
  const toolExecutionsBefore = toolExecutions;

  // Fork after step 0 (first model call) and step 1 (tool step), mid-loop: both replay and only the second model call re-executes.
  toolMock.generateResults.push(textResponse('Recovered: rainy in Oslo.'));
  const forked = await DBOS.forkWorkflow<ReturnType<typeof toolWorkflow>>(workflowID, 2);
  assert.equal(await forked.getResult(), 'Recovered: rainy in Oslo.');
  assert.equal(toolMock.generateCalls, callsBefore + 1);
  assert.equal(toolExecutions, toolExecutionsBefore);
});

test('binary file content survives checkpointing and replay', async () => {
  const bytes = [137, 80, 78, 71, 3, 250];
  fileMock.generateResults.push(
    contentResponse([
      { type: 'file', mediaType: 'image/png', data: { type: 'data', data: new Uint8Array(bytes) } },
      { type: 'text', text: 'made you an image' },
    ]),
  );
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(fileWorkflow, { workflowID })('draw');
  const original = await handle.getResult();
  assert.deepEqual(original.bytes, bytes);
  assert.equal(original.mediaType, 'image/png');
  assert.equal(original.text, 'made you an image');

  const forked = await DBOS.forkWorkflow<ReturnType<typeof fileWorkflow>>(workflowID, 1);
  const replayed = (await forked.getResult()) as Awaited<ReturnType<typeof fileWorkflow>>;
  assert.deepEqual(replayed.bytes, bytes);
  assert.equal(fileMock.generateCalls, 1);
});

test('reasoning content is checkpointed and replayed', async () => {
  reasoningMock.streamPartLists.push([
    { type: 'stream-start', warnings: [] },
    { type: 'reasoning-start', id: 'r1' },
    { type: 'reasoning-delta', id: 'r1', delta: 'thinking...' },
    { type: 'reasoning-end', id: 'r1' },
    { type: 'text-start', id: 't1' },
    { type: 'text-delta', id: 't1', delta: 'The answer is 42.' },
    { type: 'text-end', id: 't1' },
    { type: 'finish', finishReason: { unified: 'stop', raw: undefined }, usage: usage() },
  ]);
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(reasoningWorkflow, { workflowID })('hi');
  const original = await handle.getResult();
  assert.equal(original.text, 'The answer is 42.');
  assert.equal(original.reasoning, 'thinking...');

  const forked = await DBOS.forkWorkflow<ReturnType<typeof reasoningWorkflow>>(workflowID, 1);
  const replayed = (await forked.getResult()) as Awaited<ReturnType<typeof reasoningWorkflow>>;
  assert.equal(replayed.text, 'The answer is 42.');
  assert.equal(replayed.reasoning, 'thinking...');
  assert.equal(reasoningMock.streamCalls, 1);
});

test('a pre-stream failure is retried durably, then the successful attempt streams live', async () => {
  // doStream rejects before any part is emitted, so the retry re-streams cleanly with nothing leaked.
  retryStreamMock.streamCallErrors.push(new Error('connection reset'));
  retryStreamMock.streamPartLists.push(textStreamParts(['Good', ' answer']));
  const handle = await DBOS.startWorkflow(retryStreamWorkflow, { workflowID: randomUUID() })('hi');
  const result = await handle.getResult();
  // The successful attempt streams live, one delta per part (not collapsed as a buffered replay would).
  assert.deepEqual(result.deltas, ['Good', ' answer']);
  assert.equal(result.text, 'Good answer');
  assert.equal(retryStreamMock.streamCalls, 2);
});

test('a mid-stream error fails the model call and is not retried once output has streamed live', async () => {
  failStreamMock.streamPartLists.push([
    { type: 'stream-start', warnings: [] },
    { type: 'text-start', id: 't1' },
    { type: 'text-delta', id: 't1', delta: 'about to fail' },
    { type: 'error', error: new Error('boom') },
  ]);
  const handle = await DBOS.startWorkflow(failStreamWorkflow, { workflowID: randomUUID() })();
  await assert.rejects(handle.getResult(), /boom/);
  // Retries are on by default, but parts already streamed live, so re-streaming would duplicate output: no retry.
  assert.equal(failStreamMock.streamCalls, 1);
  // The abandoned model stream is torn down, not left draining the provider connection.
  assert.equal(failStreamMock.streamCancellations, 1);
});

// Reads the checkpointed output of the stream step so cancel tests can assert what was actually recorded.
async function recordedStreamStep(workflowID: string) {
  const steps = await DBOS.listWorkflowSteps(workflowID);
  const step = steps!.find((s) => s.name === 'mock.mock-model.stream')!;
  return { step, output: step.output as { content: { type: string; text?: string }[]; finishReason: { unified: string } } };
}

test('cancelling the consumer stream still checkpoints the full model call', async () => {
  cancelMock.streamPartLists.push(textStreamParts(['Hello', ' world']));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(cancelWorkflow, { workflowID })();
  assert.equal(await handle.getResult(), 'cancelled early');
  assert.equal(cancelMock.streamCalls, 1);
  // The step must record the full model output, not a truncated/empty result from stopping at the cancel.
  const { step, output } = await recordedStreamStep(workflowID);
  assert.equal(step.error, null);
  assert.equal(output.content.length, 1);
  assert.equal(output.content[0]!.text, 'Hello world');
  assert.equal(output.finishReason.unified, 'stop');
});

test('cancel then model failure records a success, so a fork replays identically instead of failing', async () => {
  cancelErrorMock.streamPartLists.push([
    { type: 'stream-start', warnings: [] },
    { type: 'text-start', id: 't1' },
    { type: 'text-delta', id: 't1', delta: 'a' },
    { type: 'text-delta', id: 't1', delta: 'b' },
    { type: 'error', error: new Error('upstream failure after cancel') },
  ]);
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(cancelErrorWorkflow, { workflowID })();
  assert.equal(await handle.getResult(), 'cancelled before error');
  assert.equal(cancelErrorMock.streamCalls, 1);
  // The post-cancel error is swallowed: the partial content read before it is recorded as a successful step.
  const { step, output } = await recordedStreamStep(workflowID);
  assert.equal(step.error, null);
  assert.equal(output.content[0]!.text, 'ab');

  // Fork after the stream step: it replays from its checkpoint. Before the fix the step was
  // recorded as an error and the fork threw; now it's a success and replays identically.
  const forked = await DBOS.forkWorkflow<ReturnType<typeof cancelErrorWorkflow>>(workflowID, 1);
  assert.equal(await forked.getResult(), 'cancelled before error');
  assert.equal(cancelErrorMock.streamCalls, 1);
});

test('cancel then stream-level rejection records a success, so a fork replays identically', async () => {
  cancelRejectMock.streamPartLists.push([
    { type: 'stream-start', warnings: [] },
    { type: 'text-start', id: 't1' },
    { type: 'text-delta', id: 't1', delta: 'a' },
    { type: 'text-delta', id: 't1', delta: 'b' },
    new Error('connection reset after cancel'),
  ]);
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(cancelRejectWorkflow, { workflowID })();
  assert.equal(await handle.getResult(), 'cancelled before rejection');
  assert.equal(cancelRejectMock.streamCalls, 1);
  // The post-cancel rejection is swallowed too: the partial content read before it is recorded as a success.
  const { step, output } = await recordedStreamStep(workflowID);
  assert.equal(step.error, null);
  assert.equal(output.content[0]!.text, 'ab');

  // Fork after the stream step: before the fix the rejection was recorded as the step outcome and the fork threw here.
  const forked = await DBOS.forkWorkflow<ReturnType<typeof cancelRejectWorkflow>>(workflowID, 1);
  assert.equal(await forked.getResult(), 'cancelled before rejection');
  assert.equal(cancelRejectMock.streamCalls, 1);
});

test('cancel before doStream settles: a later rejection still checkpoints a success', async () => {
  earlyCancelMock.streamCallErrors.push(new Error('connect failed after cancel'));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(earlyCancelWorkflow, { workflowID })();
  assert.equal(await handle.getResult(), 'cancelled immediately');

  // Checkpoint-level assertion: this consumer shape masks a recorded error live, so inspect the step directly.
  const { step, output } = await recordedStreamStep(workflowID);
  assert.equal(step.error, null); // a success, not the post-cancel doStream rejection
  assert.deepEqual(output.content, []); // cancelled before any content arrived

  const forked = await DBOS.forkWorkflow<ReturnType<typeof earlyCancelWorkflow>>(workflowID, 1);
  assert.equal(await forked.getResult(), 'cancelled immediately');
});

test('invoking a workflow twice with the same ID does not repeat model calls', async () => {
  generateMock.generateResults.push(textResponse('only once'));
  const workflowID = randomUUID();
  const first = await DBOS.startWorkflow(generateWorkflow, { workflowID })('hi');
  const original = await first.getResult();
  const callsAfter = generateMock.generateCalls;

  const second = await DBOS.startWorkflow(generateWorkflow, { workflowID })('hi');
  const repeated = await second.getResult();
  assert.equal(repeated.text, original.text);
  assert.equal(generateMock.generateCalls, callsAfter);
});

test('concurrent durable model calls in one workflow are refused, not silently corrupted', async () => {
  concurrentMock.generateResults.push(textResponse('A result'), textResponse('B result'));
  const handle = await DBOS.startWorkflow(concurrentWorkflow, { workflowID: randomUUID() })();
  await assert.rejects(handle.getResult(), /Concurrent durable model calls/);
});

test('accumulator preserves arrival order when a part interleaves an open text block', async () => {
  orderingMock.streamPartLists.push([
    { type: 'stream-start', warnings: [] },
    { type: 'text-start', id: 't1' },
    { type: 'text-delta', id: 't1', delta: 'According to ' },
    { type: 'source', sourceType: 'url', id: 's1', url: 'https://example.com', title: 'Example' },
    { type: 'text-delta', id: 't1', delta: 'the docs' },
    { type: 'text-end', id: 't1' },
    { type: 'finish', finishReason: { unified: 'stop', raw: undefined }, usage: usage() },
  ]);
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(orderingWorkflow, { workflowID })();
  const result = await handle.getResult();
  // The source arrived while the text block was open; content must stay in arrival order (text then source), not [source, text].
  assert.deepEqual(result.types, ['text', 'source']);
  assert.equal(result.text, 'According to the docs');

  // Fork past the stream step: the checkpointed content replays via replayParts, which must preserve arrival order too.
  const forked = await DBOS.forkWorkflow<ReturnType<typeof orderingWorkflow>>(workflowID, 1);
  const replayed = (await forked.getResult()) as Awaited<ReturnType<typeof orderingWorkflow>>;
  assert.deepEqual(replayed.types, ['text', 'source']);
  assert.equal(replayed.text, 'According to the docs');
});

test('generateImage runs as a durable step, bytes survive, and replay does not re-call the model', async () => {
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(imageWorkflow, { workflowID })('a durable cat');
  const result = await handle.getResult();
  assert.equal(result.count, 1);
  assert.deepEqual(result.bytes, [...IMAGE_BYTES, 1]); // image tagged with call ordinal 1
  assert.equal(imageMock.generateCalls, 1);

  const forked = await DBOS.forkWorkflow<ReturnType<typeof imageWorkflow>>(workflowID, 1);
  const replayed = (await forked.getResult()) as Awaited<ReturnType<typeof imageWorkflow>>;
  assert.deepEqual(replayed.bytes, [...IMAGE_BYTES, 1]);
  assert.equal(imageMock.generateCalls, 1);
});

test('multi-image generateImage: parallel batches are durable and replay in the same order (no guard needed)', async () => {
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(multiImageWorkflow, { workflowID })('three durable cats');
  const original = await handle.getResult();
  assert.equal(original.length, 3);
  assert.equal(multiImageMock.generateCalls, 3); // n=3 with maxImagesPerCall=1 → 3 parallel batches, no guard throw

  // Fork past all image steps: each replays from its checkpoint, so the image order is byte-identical and nothing is re-called.
  const steps = await DBOS.listWorkflowSteps(workflowID);
  const noopStep = steps!.find((s) => s.name === 'noop')!;
  const forked = await DBOS.forkWorkflow<ReturnType<typeof multiImageWorkflow>>(workflowID, noopStep.functionID);
  const replayed = (await forked.getResult()) as Awaited<ReturnType<typeof multiImageWorkflow>>;
  assert.deepEqual(replayed, original); // deterministic: same batch → same funcID → same checkpoint → same order
  assert.equal(multiImageMock.generateCalls, 3);
});

test('an abort error is treated as terminal and not retried', async () => {
  abortMock.generateResults.push(Object.assign(new Error('the operation was aborted'), { name: 'AbortError' }));
  const handle = await DBOS.startWorkflow(abortWorkflow, { workflowID: randomUUID() })();
  await assert.rejects(handle.getResult(), /aborted/);
  // maxAttempts is 5, but an abort must not be retried.
  assert.equal(abortMock.generateCalls, 1);
});

test('a provider non-retryable error is not retried even with retriesAllowed', async () => {
  nonRetryMock.generateResults.push(Object.assign(new Error('invalid request'), { isRetryable: false }));
  const handle = await DBOS.startWorkflow(nonRetryWorkflow, { workflowID: randomUUID() })();
  await assert.rejects(handle.getResult(), /invalid request/);
  // Without error classification this would be called maxAttempts (5) times.
  assert.equal(nonRetryMock.generateCalls, 1);
});

test('MCP tools list and execute as durable steps; replay does not re-execute the tool', async () => {
  mcpToolMock.generateResults.push(
    toolCallResponse('getWeather', '{"city":"Paris"}'),
    textResponse('It is sunny in Paris.'),
  );
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(mcpWorkflow, { workflowID })('weather in Paris?');
  assert.equal(await handle.getResult(), 'It is sunny in Paris.');
  assert.equal(mcpClient.executeCalls, 1);

  const steps = await DBOS.listWorkflowSteps(workflowID);
  assert.ok(steps?.some((s) => s.name === 'mcp.listTools'), 'tool listing recorded as a durable step');
  assert.ok(steps?.some((s) => s.name === 'mcp.tool.getWeather'), 'tool call recorded as a durable step');
  const generateCallsBefore = mcpToolMock.generateCalls;

  // Fork past every model/tool step: they all replay from checkpoints, so nothing is re-invoked.
  const noopStep = steps!.find((s) => s.name === 'noop')!;
  const forked = await DBOS.forkWorkflow<ReturnType<typeof mcpWorkflow>>(workflowID, noopStep.functionID);
  assert.equal(await forked.getResult(), 'It is sunny in Paris.');
  assert.equal(mcpClient.executeCalls, 1); // tool not re-invoked on replay
  assert.equal(mcpToolMock.generateCalls, generateCallsBefore); // model not re-called on replay
});

test('parallel MCP tool calls each execute durably and replay without re-executing', async () => {
  // One model turn returns two tool calls; the AI SDK runs them in parallel (Promise.all).
  parallelMcpMock.generateResults.push(
    toolCallsResponse([
      { toolName: 'getWeather', input: '{"city":"Paris"}' },
      { toolName: 'getTime', input: '{"city":"Paris"}' },
    ]),
    textResponse('Fetched weather and time for Paris.'),
  );
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(parallelMcpWorkflow, { workflowID })('weather and time in Paris?');
  assert.equal(await handle.getResult(), 'Fetched weather and time for Paris.');
  assert.equal(parallelMcpClient.weatherCalls, 1);
  assert.equal(parallelMcpClient.timeCalls, 1);

  const steps = await DBOS.listWorkflowSteps(workflowID);
  assert.ok(steps?.some((s) => s.name === 'mcp.tool.getWeather'), 'first parallel tool recorded as a step');
  assert.ok(steps?.some((s) => s.name === 'mcp.tool.getTime'), 'second parallel tool recorded as a step');

  // Fork past both parallel tool steps: they replay from checkpoints (would throw DBOSUnexpectedStepError if reordered).
  const noopStep = steps!.find((s) => s.name === 'noop')!;
  const forked = await DBOS.forkWorkflow<ReturnType<typeof parallelMcpWorkflow>>(workflowID, noopStep.functionID);
  assert.equal(await forked.getResult(), 'Fetched weather and time for Paris.');
  assert.equal(parallelMcpClient.weatherCalls, 1); // not re-executed on replay
  assert.equal(parallelMcpClient.timeCalls, 1);
});

test('MCP tool results reach the model as converted content; title/metadata/_meta survive checkpointing', async () => {
  richMcpMock.generateResults.push(toolCallResponse('screenshot', '{}'), textResponse('Here is your screenshot.'));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(richMcpWorkflow, { workflowID })('take a screenshot');
  const result = await handle.getResult();
  assert.equal(result.text, 'Here is your screenshot.');
  assert.equal(result.title, 'Screenshot');
  assert.deepEqual(result.metadata, { clientName: 'mock-mcp', toolName: 'screenshot' });
  assert.deepEqual(result.meta, { 'mcp/app': { uri: 'ui://screenshot' } });
  assert.equal(result.converts, true);
  assert.equal(richMcpClient.screenshotCalls, 1);

  // The follow-up model call must see the tool result as converted content (text + file), not raw MCP JSON.
  const followUp = richMcpMock.generateOptions.at(-1)!;
  const toolMessage = followUp.prompt.find((m) => m.role === 'tool')!;
  const resultPart = (toolMessage.content as { type: string; output?: { type: string; value: unknown } }[]).find(
    (p) => p.type === 'tool-result',
  )!;
  assert.equal(resultPart.output!.type, 'content');
  const value = resultPart.output!.value as { type: string; text?: string; mediaType?: string; data?: unknown }[];
  // Lowercase proves the built-in conversion ran (the client's own converter would uppercase).
  assert.deepEqual(value[0], { type: 'text', text: 'took screenshot' });
  assert.equal(value[1]!.type, 'file');
  assert.equal(value[1]!.mediaType, 'image/png');
  assert.deepEqual(value[1]!.data, { type: 'data', data: 'QUJD' });

  // Fork past all steps: tools rebuilt from the checkpointed listing must carry the same fields and converter.
  const steps = await DBOS.listWorkflowSteps(workflowID);
  const noopStep = steps!.find((s) => s.name === 'noop')!;
  const forked = await DBOS.forkWorkflow<ReturnType<typeof richMcpWorkflow>>(workflowID, noopStep.functionID);
  const replayed = (await forked.getResult()) as Awaited<ReturnType<typeof richMcpWorkflow>>;
  assert.equal(replayed.title, 'Screenshot');
  assert.deepEqual(replayed.meta, { 'mcp/app': { uri: 'ui://screenshot' } });
  assert.equal(replayed.converts, true);
  assert.equal(richMcpClient.screenshotCalls, 1); // tool not re-executed on replay
});

test('toolOptions forward to the client on listing and on each tool call', async () => {
  subsetMcpMock.generateResults.push(toolCallResponse('getWeather', '{"city":"Lima"}'), textResponse('Sunny in Lima.'));
  const handle = await DBOS.startWorkflow(subsetMcpWorkflow, { workflowID: randomUUID() })('weather in Lima?');
  const result = await handle.getResult();
  assert.equal(result.text, 'Sunny in Lima.');
  // Schemas subsetting: only the explicitly listed tool is exposed to the model.
  assert.deepEqual(result.toolNames, ['getWeather']);
  assert.equal(subsetMcpClient.weatherCalls, 1);
  // Both the listing and the in-step re-fetch received the same options.
  assert.equal(subsetMcpClient.toolsOptionsLog.length, 2);
  for (const opts of subsetMcpClient.toolsOptionsLog) {
    assert.deepEqual(opts, subsetSchemas);
  }
});

test('tools without toModelOutput are not given one', async () => {
  const tools = await durableMCPTools(new MockMCPClient());
  assert.equal(tools.getWeather!.toModelOutput, undefined);
  assert.equal(tools.getWeather!.title, undefined);
});

test('a description-less MCP tool is reconstructed without an empty-string description', async () => {
  const client: MCPClientLike = {
    tools: async () => ({ ping: tool({ inputSchema: z.object({}), execute: async () => 'pong' }) }),
  };
  const tools = await durableMCPTools(client);
  // Before the fix this was '' (an empty description sent to the model); upstream omits the field instead.
  assert.equal(tools.ping!.description, undefined);
});

test('explicit shouldRetry: undefined falls back to the default classification', async () => {
  undefinedRetryMock.generateResults.push(Object.assign(new Error('bad request'), { isRetryable: false }));
  const handle = await DBOS.startWorkflow(undefinedRetryWorkflow, { workflowID: randomUUID() })();
  await assert.rejects(handle.getResult(), /bad request/);
  // Before the fix, an explicit `shouldRetry: undefined` disabled classification → 5 retries.
  assert.equal(undefinedRetryMock.generateCalls, 1);
});

test('a caller-provided shouldRetry overrides the default', async () => {
  // A plain Error is retryable under the default; shouldRetry: () => false must prevent any retry.
  overrideRetryMock.generateResults.push(new Error('transient'));
  const handle = await DBOS.startWorkflow(overrideRetryWorkflow, { workflowID: randomUUID() })();
  await assert.rejects(handle.getResult(), /transient/);
  assert.equal(overrideRetryMock.generateCalls, 1);
});

test('MCP tool with an async JSON schema is awaited before checkpointing (not stored as an empty Promise)', async () => {
  const handle = await DBOS.startWorkflow(asyncSchemaWorkflow, { workflowID: randomUUID() })();
  const schema = (await handle.getResult()) as { type?: string; properties?: unknown; required?: unknown };
  // Without the await, the Promise checkpoints as {} and this schema would be empty.
  assert.equal(schema.type, 'object');
  assert.deepEqual(schema.properties, { host: { type: 'string' } });
  assert.deepEqual(schema.required, ['host']);
});

test('a structured error-part payload keeps its JSON message and non-retryable classification', async () => {
  structuredErrorMock.streamPartLists.push([
    { type: 'stream-start', warnings: [] },
    { type: 'error', error: { message: 'quota exceeded', code: 'insufficient_quota', isRetryable: false } },
  ]);
  const handle = await DBOS.startWorkflow(structuredErrorWorkflow, { workflowID: randomUUID() })();
  // Before the fix the payload became Error("[object Object]").
  await assert.rejects(handle.getResult(), /quota exceeded[\s\S]*insufficient_quota/);
  // The payload's isRetryable: false is honored: no retry despite maxAttempts 3.
  assert.equal(structuredErrorMock.streamCalls, 1);
});

test('a named abort/timeout error-part is treated as terminal, not retried', async () => {
  // Three identical attempts available; before the fix toStepError dropped the name, so it was classified retryable.
  const part = { type: 'error' as const, error: { name: 'TimeoutError', message: 'upstream timeout' } };
  abortPartMock.streamPartLists.push(
    [{ type: 'stream-start', warnings: [] }, part],
    [{ type: 'stream-start', warnings: [] }, part],
    [{ type: 'stream-start', warnings: [] }, part],
  );
  const handle = await DBOS.startWorkflow(abortPartWorkflow, { workflowID: randomUUID() })();
  await assert.rejects(handle.getResult(), /upstream timeout/);
  assert.equal(abortPartMock.streamCalls, 1); // name preserved → isAbortError → terminal, no retry
});

test('a stream ending with no finish and no output is retried, not checkpointed as an empty success', async () => {
  // Attempt 1 closes cleanly after only stream-start (no finish, no content); attempt 2 succeeds.
  emptyStreamMock.streamPartLists.push([{ type: 'stream-start', warnings: [] }], textStreamParts(['Recovered']));
  const handle = await DBOS.startWorkflow(emptyStreamWorkflow, { workflowID: randomUUID() })('hi');
  const result = await handle.getResult();
  // Before the fix, attempt 1 checkpointed a permanent empty success (text '') and never retried.
  assert.equal(result.text, 'Recovered');
  assert.deepEqual(result.deltas, ['Recovered']);
  assert.equal(emptyStreamMock.streamCalls, 2);
});

test('a stream that never produces output fails the step after retries instead of succeeding empty', async () => {
  const emptyParts = (): Parameters<typeof emptyStreamMock.streamPartLists.push>[0] => [
    { type: 'stream-start', warnings: [] },
  ];
  emptyStreamMock.streamPartLists.push(emptyParts(), emptyParts(), emptyParts());
  const callsBefore = emptyStreamMock.streamCalls;
  const handle = await DBOS.startWorkflow(emptyStreamWorkflow, { workflowID: randomUUID() })('hi');
  await assert.rejects(handle.getResult(), /without a finish part/);
  assert.equal(emptyStreamMock.streamCalls, callsBefore + 3); // retried to maxAttempts
});

test('replay re-synthesizes tool-input parts and start-part metadata from the checkpoint', async () => {
  replayPartsMock.streamPartLists.push([
    { type: 'stream-start', warnings: [] },
    { type: 'text-start', id: 't1', providerMetadata: { mock: { redacted: true } } },
    { type: 'text-delta', id: 't1', delta: 'Calling a tool.' },
    { type: 'text-end', id: 't1' },
    { type: 'tool-input-start', id: 'call-9', toolName: 'getWeather' },
    { type: 'tool-input-delta', id: 'call-9', delta: '{"city":"Nice"}' },
    { type: 'tool-input-end', id: 'call-9' },
    { type: 'tool-call', toolCallId: 'call-9', toolName: 'getWeather', input: '{"city":"Nice"}' },
    { type: 'finish', finishReason: { unified: 'tool-calls', raw: undefined }, usage: usage() },
  ]);
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(replayPartsWorkflow, { workflowID })();
  await handle.getResult();

  const forked = await DBOS.forkWorkflow<ReturnType<typeof replayPartsWorkflow>>(workflowID, 1);
  const replayed = (await forked.getResult()) as Awaited<ReturnType<typeof replayPartsWorkflow>>;
  assert.equal(replayPartsMock.streamCalls, 1);

  // The tool-input grammar is synthesized around the tool-call; ids must match the toolCallId (what consumer callbacks key on).
  const inputStart = replayed.find((p) => p.type === 'tool-input-start')!;
  assert.equal(inputStart.id, 'call-9');
  assert.equal(inputStart.toolName, 'getWeather');
  const types = replayed.map((p) => p.type);
  assert.deepEqual(types.slice(types.indexOf('tool-input-start'), types.indexOf('tool-call') + 1), [
    'tool-input-start',
    'tool-input-delta',
    'tool-input-end',
    'tool-call',
  ]);
  assert.equal(replayed.find((p) => p.type === 'tool-input-delta')!.delta, '{"city":"Nice"}');
  // Start parts carry the checkpointed providerMetadata on replay too.
  assert.deepEqual(replayed.find((p) => p.type === 'text-start')!.providerMetadata, { mock: { redacted: true } });
});

test('split response-metadata parts merge per-field in the checkpoint', async () => {
  metadataMock.streamPartLists.push([
    { type: 'stream-start', warnings: [] },
    { type: 'response-metadata', id: 'resp-split' },
    { type: 'response-metadata', modelId: 'mock-model-9' },
    { type: 'text-start', id: 't1' },
    { type: 'text-delta', id: 't1', delta: 'ok' },
    { type: 'text-end', id: 't1' },
    { type: 'finish', finishReason: { unified: 'stop', raw: undefined }, usage: usage() },
  ]);
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(metadataWorkflow, { workflowID })();
  const original = await handle.getResult();
  assert.equal(original.id, 'resp-split');
  assert.equal(original.modelId, 'mock-model-9');

  // Before the fix the second part clobbered the first and the replayed response lost its id.
  const forked = await DBOS.forkWorkflow<ReturnType<typeof metadataWorkflow>>(workflowID, 1);
  const replayed = (await forked.getResult()) as Awaited<ReturnType<typeof metadataWorkflow>>;
  assert.equal(replayed.id, 'resp-split');
  assert.equal(replayed.modelId, 'mock-model-9');
});

test('a mixed string/bytes image batch is encoded per element, not corrupted', async () => {
  // Spec-violating but defensive: images[0] is bytes, images[1] is already base64.
  mixedImageMock.imageOverrides.push([new Uint8Array([1, 2, 3]), Buffer.from('ABC').toString('base64')]);
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(mixedImageWorkflow, { workflowID })();
  const original = await handle.getResult();
  assert.deepEqual(original[0], [1, 2, 3]);
  assert.deepEqual(original[1], [65, 66, 67]); // 'ABC' — before the fix this was double-encoded garbage
  assert.equal(mixedImageMock.generateCalls, 1); // n=2 within maxImagesPerCall → single call

  const forked = await DBOS.forkWorkflow<ReturnType<typeof mixedImageWorkflow>>(workflowID, 1);
  const replayed = (await forked.getResult()) as Awaited<ReturnType<typeof mixedImageWorkflow>>;
  assert.deepEqual(replayed, original);
});

test('a streaming MCP tool execute checkpoints its final value, not an empty object', async () => {
  streamingToolMock.generateResults.push(toolCallResponse('countdown', '{}'), textResponse('Lift off confirmed.'));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(streamingToolWorkflow, { workflowID })('count down');
  assert.equal(await handle.getResult(), 'Lift off confirmed.');

  // The checkpoint records the last yielded value; before the fix the generator serialized as {}.
  const steps = await DBOS.listWorkflowSteps(workflowID);
  const toolStep = steps!.find((s) => s.name === 'mcp.tool.countdown')!;
  assert.equal(toolStep.output, 'lift off');

  // The follow-up model call saw the final value, not {}.
  const followUp = streamingToolMock.generateOptions.at(-1)!;
  assert.ok(JSON.stringify(followUp.prompt).includes('lift off'));
});

test('a timed-out (abandoned) stream attempt stops emitting and cannot interleave with its retry', async () => {
  let releaseStalled!: () => void;
  const stalled = new Promise<void>((resolve) => (releaseStalled = resolve));
  let releaseRetry!: () => void;
  const retryGate = new Promise<void>((resolve) => (releaseRetry = resolve));

  // Attempt 1 stalls pre-emission (timeout fires, DBOS abandons it and retries); its stream later wakes up while attempt 2 is mid-stream and the consumer is listening.
  timeoutStreamMock.streamPartLists.push(
    [
      () => stalled,
      { type: 'stream-start', warnings: [] },
      { type: 'text-start', id: 'x1' },
      { type: 'text-delta', id: 'x1', delta: 'DUP' },
      { type: 'text-end', id: 'x1' },
      { type: 'finish', finishReason: { unified: 'stop', raw: undefined }, usage: usage() },
    ],
    [
      { type: 'stream-start', warnings: [] },
      { type: 'text-start', id: 't1' },
      { type: 'text-delta', id: 't1', delta: 'Good' },
      () => retryGate,
      { type: 'text-delta', id: 't1', delta: ' answer' },
      { type: 'text-end', id: 't1' },
      { type: 'finish', finishReason: { unified: 'stop', raw: undefined }, usage: usage() },
    ],
  );

  const handle = await DBOS.startWorkflow(timeoutStreamWorkflow, { workflowID: randomUUID() })();
  // Wait until the retry is mid-stream, then wake the abandoned attempt while the stream is still open.
  for (let i = 0; i < 500 && !timeoutLiveDeltas.includes('Good'); i++) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert.ok(timeoutLiveDeltas.includes('Good'), 'retry attempt never streamed');
  releaseStalled();
  await new Promise((resolve) => setTimeout(resolve, 50)); // give the abandoned stream time to (incorrectly) emit
  releaseRetry();

  const result = await handle.getResult();
  // The abandoned attempt contributes nothing; only the retry reaches the consumer and the checkpoint.
  assert.deepEqual(timeoutLiveDeltas, ['Good', ' answer']);
  assert.equal(result.text, 'Good answer');
  assert.equal(timeoutStreamMock.streamCalls, 2);
});

test('aborting mid-stream checkpoints the partial output as a success and replays gracefully', async () => {
  abortStreamMock.streamPartLists.push(textStreamParts(['a', 'b', 'c', 'd', 'e']));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(abortStreamWorkflow, { workflowID })();
  // The AI SDK ends an aborted stream gracefully, so the live workflow completes.
  const original = await handle.getResult();
  assert.ok(original.length >= 2, `expected partial deltas, got ${original.length}`);
  assert.equal(abortStreamMock.streamCalls, 1); // an abort is never retried

  // The step must record the partial output as a success, not the post-abort AbortError.
  const { step, output } = await recordedStreamStep(workflowID);
  assert.equal(step.error, null);
  const recordedText = output.content[0]!.text!;
  assert.ok(recordedText.length >= 2 && recordedText.length < 5, `expected a partial checkpoint, got "${recordedText}"`);
  assert.ok('abcde'.startsWith(recordedText));

  // Fork past the stream step: before the fix the checkpoint was an AbortError and replay threw where the live run completed (the replayed execution's signal isn't aborted, so ai treats it as a hard error).
  const forked = await DBOS.forkWorkflow<ReturnType<typeof abortStreamWorkflow>>(workflowID, 1);
  const replayed = (await forked.getResult()) as Awaited<ReturnType<typeof abortStreamWorkflow>>;
  // Replay delivers the checkpointed partial content as one delta per block; no abort fires.
  assert.equal(replayed.join(''), recordedText);
  assert.equal(abortStreamMock.streamCalls, 1);
});

test('aborting mid-stream still checkpoints the complete model call, so recovery replays the whole answer', async () => {
  abortRecoveryMock.streamPartLists.push([
    { type: 'stream-start', warnings: [] },
    { type: 'text-start', id: 't1' },
    { type: 'text-delta', id: 't1', delta: 'HELLO' },
    () => abortRecoveryGate.promise,
    { type: 'text-delta', id: 't1', delta: ' WORLD' },
    { type: 'text-end', id: 't1' },
    { type: 'finish', finishReason: { unified: 'stop', raw: undefined }, usage: usage() },
  ]);
  abortRecoveryGate = newGate();
  abortRecoveryAborts = true;

  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(abortRecoveryWorkflow, { workflowID })();
  // The AI SDK ends an aborted stream gracefully, so the live consumer completes with the pre-abort text.
  assert.equal(await handle.getResult(), 'HELLO');
  assert.equal(abortRecoveryMock.streamCalls, 1); // an abort is never retried

  // The model call keeps draining after the consumer detaches, so wait for its checkpoint before reading it.
  for (let i = 0; i < 500 && !(await DBOS.listWorkflowSteps(workflowID))?.some((s) => s.name === 'mock.mock-model.stream'); i++) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  // The consumer's abort must not truncate the durable record: a partial success here is permanent and silent.
  const { step, output } = await recordedStreamStep(workflowID);
  assert.equal(step.error, null);
  assert.equal(output.content[0]!.text, 'HELLO WORLD');
  assert.equal(output.finishReason.unified, 'stop');

  // Recovery (fork past the model step) must replay the whole answer, not the truncated live view.
  abortRecoveryGate = newGate();
  abortRecoveryAborts = false;
  const forked = await DBOS.forkWorkflow<ReturnType<typeof abortRecoveryWorkflow>>(workflowID, 1);
  const recovered = (await forked.getResult()) as Awaited<ReturnType<typeof abortRecoveryWorkflow>>;
  assert.equal(recovered, 'HELLO WORLD');
  assert.equal(abortRecoveryMock.streamCalls, 1); // replayed from the checkpoint, not re-generated
});

test('a durable call after aborting a stream is not rejected as concurrent', async () => {
  guardRaceMock.generateResults.push(textResponse('summary'));
  const handle = await DBOS.startWorkflow(guardRaceWorkflow, { workflowID: randomUUID() })();
  // Before the fix the guard stayed held until the aborted stream step settled, so this follow-up threw
  // "Concurrent durable model calls ..." and the workflow rejected.
  assert.equal(await handle.getResult(), 'summary');
});

test('an MCP tool aborted with a DOMException reason checkpoints the real error, not a TypeError', async () => {
  domAbortMock.generateResults.push(toolCallResponse('slowTool', '{}'));
  const started = new Promise<void>((resolve) => (slowToolStarted = resolve));
  const workflowID = randomUUID();
  const executionsBefore = slowToolExecutions;
  const handle = await DBOS.startWorkflow(domAbortWorkflow, { workflowID })();
  await started;
  toolAbort!();
  // The abort surfaces to the caller as an AbortError, not a TypeError from mutating the reason's message.
  assert.equal(await handle.getResult(), 'caught:AbortError');
  assert.equal(slowToolExecutions - executionsBefore, 1); // aborted tool not retried

  const steps = await DBOS.listWorkflowSteps(workflowID);
  const toolStep = steps!.find((s) => s.name === 'mcp.tool.slowTool')!;
  // The real abort reason is recorded — before the fix this was "Cannot set property message ... which has only a getter".
  assert.ok(toolStep.error !== null, 'aborted tool call recorded as a step error');
  assert.match(String(toolStep.error), /abort/i);
  assert.doesNotMatch(String(toolStep.error), /Cannot set property message|only a getter|TypeError/);
});

test('an aborted MCP tool call is not retried even when its error is not named AbortError', async () => {
  genericAbortMock.generateResults.push(toolCallResponse('slowTool', '{}'));
  const started = new Promise<void>((resolve) => (slowToolStarted = resolve));
  const workflowID = randomUUID();
  const executionsBefore = slowToolExecutions;
  const handle = await DBOS.startWorkflow(genericAbortWorkflow, { workflowID })();
  await started;
  toolAbort!();
  assert.equal(await handle.getResult(), 'caught:AbortError');
  // The generic "connection reset" error would be retryable by name; the aborted signal makes it terminal.
  assert.equal(slowToolExecutions - executionsBefore, 1);

  const steps = await DBOS.listWorkflowSteps(workflowID);
  const toolStep = steps!.find((s) => s.name === 'mcp.tool.slowTool')!;
  assert.match(String(toolStep.error), /connection reset by peer/);
});

test('a non-aborted MCP tool failure is still retried', async () => {
  retryToolMock.generateResults.push(toolCallResponse('slowTool', '{}'), textResponse('done'));
  const executionsBefore = slowToolExecutions;
  const handle = await DBOS.startWorkflow(retryToolWorkflow, { workflowID: randomUUID() })();
  await handle.getResult().catch(() => undefined);
  // The retry fix is scoped to aborts: a plain failure still burns all maxAttempts.
  assert.equal(slowToolExecutions - executionsBefore, 3);
});

test('a throwing isRetryable accessor does not replace the step error or disable retries', async () => {
  const makeEvil = () => {
    const error = new Error('real failure');
    Object.defineProperty(error, 'isRetryable', {
      get() {
        throw new Error('getter boom');
      },
    });
    return error;
  };
  evilRetryMock.generateResults.push(makeEvil(), makeEvil());
  const handle = await DBOS.startWorkflow(evilRetryWorkflow, { workflowID: randomUUID() })();
  await assert.rejects(handle.getResult(), (error: Error) => {
    // Before the fix the classifier's own error was recorded as the step outcome.
    assert.ok(!error.message.includes('getter boom'), `classifier error leaked: ${error.message}`);
    assert.match(error.message, /real failure/);
    return true;
  });
  assert.equal(evilRetryMock.generateCalls, 2); // classified retryable → retried to maxAttempts
});

test('a retryable APICallError checkpointed under retriesAllowed:false keeps its AI SDK identity on replay', async () => {
  const apiError = new APICallError({
    message: 'service unavailable',
    url: 'https://mock/api',
    requestBodyValues: {},
    statusCode: 503,
    isRetryable: true,
  });
  // call 1 (live) fails and checkpoints an error; call 2 (live) and call 3 (fork replay) succeed.
  identityMock.generateResults.push(apiError, textResponse('recovered'), textResponse('recovered'));

  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(identityWorkflow, { workflowID })();
  // Live: DBOS records the error at step 0 (no DBOS retry); the AI SDK's own retry re-runs the model call at step 1.
  assert.equal(await handle.getResult(), 'recovered');
  assert.equal(identityMock.generateCalls, 2);

  // Fork past the errored step 0 so the error checkpoint is revived on replay. Before the fix, serialize-error
  // strips the AI SDK Symbol marker → APICallError.isInstance() is false → the AI SDK does not retry → the
  // workflow throws where the live run succeeded. With the marker restored, replay retries into the step-1 success.
  const forked = await DBOS.forkWorkflow<ReturnType<typeof identityWorkflow>>(workflowID, 1);
  assert.equal(await forked.getResult(), 'recovered');
  assert.equal(identityMock.generateCalls, 3);
});

test('a retryable GatewayError checkpointed under retriesAllowed:false keeps its identity on replay', async () => {
  // The gateway provider throws GatewayError (not APICallError) for retryable failures; its markers use a different
  // namespace, so this only passes once restoreAISDKErrorIdentity handles the gateway family too.
  const gatewayError = new GatewayRateLimitError({ message: 'rate limited', statusCode: 429 });
  assert.equal(gatewayError.isRetryable, true);
  gatewayMock.generateResults.push(gatewayError, textResponse('recovered'), textResponse('recovered'));

  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(gatewayWorkflow, { workflowID })();
  assert.equal(await handle.getResult(), 'recovered');
  assert.equal(gatewayMock.generateCalls, 2);

  const forked = await DBOS.forkWorkflow<ReturnType<typeof gatewayWorkflow>>(workflowID, 1);
  assert.equal(await forked.getResult(), 'recovered');
  assert.equal(gatewayMock.generateCalls, 3);
});

test('a retryable APICallError in embed keeps its identity on replay', async () => {
  const apiError = new APICallError({ message: 'unavailable', url: 'https://mock/api', requestBodyValues: {}, statusCode: 503, isRetryable: true });
  embedIdentityMock.errors.push(apiError); // call 1 fails; call 2 (live) and call 3 (fork) pass through to a normal embedding
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(embedIdentityWorkflow, { workflowID })();
  assert.equal(await handle.getResult(), 3); // embedding length
  assert.equal(embedIdentityMock.embedCalls, 2);

  const forked = await DBOS.forkWorkflow<ReturnType<typeof embedIdentityWorkflow>>(workflowID, 1);
  assert.equal(await forked.getResult(), 3);
  assert.equal(embedIdentityMock.embedCalls, 3);
});

test('a retryable APICallError in generateImage keeps its identity on replay', async () => {
  const apiError = new APICallError({ message: 'unavailable', url: 'https://mock/api', requestBodyValues: {}, statusCode: 503, isRetryable: true });
  imageIdentityMock.errors.push(apiError);
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(imageIdentityWorkflow, { workflowID })();
  assert.equal(await handle.getResult(), 1); // one image
  assert.equal(imageIdentityMock.generateCalls, 2);

  const forked = await DBOS.forkWorkflow<ReturnType<typeof imageIdentityWorkflow>>(workflowID, 1);
  assert.equal(await forked.getResult(), 1);
  assert.equal(imageIdentityMock.generateCalls, 3);
});

test('restoreAISDKErrorIdentity does not let a failed marker assignment replace the error', () => {
  // A frozen/non-extensible AI SDK error would throw on the symbol assignment under strict mode; a throwing set trap
  // reproduces that deterministically (tsx runs the CJS source sloppily, where a frozen assignment silently no-ops).
  const assignThrows = new Proxy(Object.assign(new Error('boom'), { name: 'AI_APICallError' }), {
    set() {
      throw new TypeError('read only');
    },
  });
  assert.equal(restoreAISDKErrorIdentity(assignThrows), assignThrows); // returns the error, does not throw

  // A throwing `name` getter must also not escape (the read happens before the symbol assignments).
  const readThrows = new Error('boom');
  Object.defineProperty(readThrows, 'name', {
    get() {
      throw new Error('name getter boom');
    },
  });
  assert.equal(restoreAISDKErrorIdentity(readThrows), readThrows);
});

test('generateText response id/timestamp stay stable across replay when the provider omits them', async () => {
  idGenMock.generateResults.push(textResponseNoMetadata('hi there'));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(idGenWorkflow, { workflowID })();
  const live = await handle.getResult();
  assert.ok(live.id && live.timestamp, 'response id/timestamp were populated');
  assert.equal(idGenMock.generateCalls, 1);

  // Fork past the model step so it replays from the checkpoint. Before the fix, generateText re-ran
  // generateId()/new Date() outside the step, so the replay produced a different id and timestamp.
  const forked = await DBOS.forkWorkflow<ReturnType<typeof idGenWorkflow>>(workflowID, 1);
  const replayed = await forked.getResult();
  assert.equal(replayed.id, live.id);
  assert.equal(replayed.timestamp, live.timestamp);
  assert.equal(idGenMock.generateCalls, 1); // model not re-called on replay
});

test('streamText response id/timestamp stay stable across replay when the provider omits them', async () => {
  idStreamMock.streamPartLists.push(textStreamPartsNoMetadata(['hi', ' there']));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(idStreamWorkflow, { workflowID })();
  const live = await handle.getResult();
  assert.ok(live.id && live.timestamp, 'response id/timestamp were populated');
  assert.equal(idStreamMock.streamCalls, 1);

  const forked = await DBOS.forkWorkflow<ReturnType<typeof idStreamWorkflow>>(workflowID, 1);
  const replayed = await forked.getResult();
  assert.equal(replayed.id, live.id);
  assert.equal(replayed.timestamp, live.timestamp);
  assert.equal(idStreamMock.streamCalls, 1); // model not re-called on replay
});

test('generateText response id/timestamp stay stable across replay when the provider returns null metadata', async () => {
  idGenMock.generateResults.push(textResponseNullMetadata('hi'));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(idGenWorkflow, { workflowID })();
  const live = await handle.getResult();
  assert.ok(live.id && live.timestamp, 'response id/timestamp were populated');
  // Before the `!= null` guard, ensureResponseMetadata skipped a null id/timestamp, so the AI SDK regenerated on replay.
  const forked = await DBOS.forkWorkflow<ReturnType<typeof idGenWorkflow>>(workflowID, 1);
  const replayed = await forked.getResult();
  assert.equal(replayed.id, live.id);
  assert.equal(replayed.timestamp, live.timestamp);
});

// Keep this test last: it shuts down and relaunches DBOS mid-suite.
test('recovered workflows replay model calls and messages from checkpoints', async () => {
  recoveryMock.generateResults.push(textResponse('durable answer'));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(recoveryWorkflow, { workflowID })('hi');
  await DBOS.send(workflowID, 'proceed', 'go');
  const original = await handle.getResult();
  assert.equal(original.text, 'durable answer');
  assert.equal(original.go, 'proceed');
  assert.equal(recoveryMock.generateCalls, 1);

  // Simulate a lost completion: flip the workflow back to PENDING, then relaunch so recovery re-executes it and the model call and recv both replay from checkpoints.
  const client = new PgClient({ connectionString: systemDatabaseUrl });
  await client.connect();
  try {
    await client.query(
      "UPDATE dbos.workflow_status SET status = 'PENDING', recovery_attempts = 0 WHERE workflow_uuid = $1",
      [workflowID],
    );
  } finally {
    await client.end();
  }
  await DBOS.shutdown();
  await DBOS.launch();

  const recovered = await DBOS.retrieveWorkflow<Awaited<ReturnType<typeof recoveryWorkflow>>>(workflowID).getResult();
  assert.equal(recovered.text, 'durable answer');
  assert.equal(recovered.go, 'proceed');
  assert.equal(recoveryMock.generateCalls, 1);
});
