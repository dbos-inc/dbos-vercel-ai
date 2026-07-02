import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { after, before, test } from 'node:test';
import { DBOS } from '@dbos-inc/dbos-sdk';
import { Client as PgClient } from 'pg';
import {
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
import { durableCalls, durableEmbeddingCalls, durableImageCalls, durableMCPTools } from '../src/index.js';
import {
  contentResponse,
  IMAGE_BYTES,
  MockEmbeddingModel,
  MockImageModel,
  MockLanguageModel,
  MockMCPClient,
  textResponse,
  textStreamParts,
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

const embedMock = new MockEmbeddingModel();
const embedModel = wrapEmbeddingModel({ model: embedMock, middleware: durableEmbeddingCalls() });

const embedWorkflow = DBOS.registerWorkflow(
  async (values: string[]) => {
    const result = await embedMany({ model: embedModel, values });
    return { count: result.embeddings.length, first: result.embeddings[0] };
  },
  { name: 'embedWorkflow' },
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

const bufferedMock = new MockLanguageModel();
const bufferedModel = wrapLanguageModel({
  model: bufferedMock,
  middleware: durableCalls({ retriesAllowed: true, maxAttempts: 3, intervalSeconds: 0 }),
});

const bufferedStreamWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const result = streamText({ model: bufferedModel, prompt });
    const deltas: string[] = [];
    for await (const delta of result.textStream) {
      deltas.push(delta);
    }
    return { deltas, text: await result.text };
  },
  { name: 'bufferedStreamWorkflow' },
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
const orderingModel = wrapLanguageModel({
  model: orderingMock,
  // Buffered (retries) mode so the consumer sees parts synthesized from the checkpointed content — asserts the accumulator's order.
  middleware: durableCalls({ retriesAllowed: true, maxAttempts: 3, intervalSeconds: 0 }),
});

const orderingWorkflow = DBOS.registerWorkflow(
  async () => {
    const result = streamText({ model: orderingModel, prompt: 'hi' });
    const types: string[] = [];
    for (const part of await result.content) {
      types.push(part.type);
    }
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

test('embedMany runs as a durable step inside a workflow', async () => {
  const handle = await DBOS.startWorkflow(embedWorkflow, { workflowID: randomUUID() })(['a', 'b']);
  const result = await handle.getResult();

  assert.equal(result.count, 2);
  assert.deepEqual(result.first, [0, 0.5, 0.25]);
  assert.equal(embedMock.embedCalls, 1);
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

test('permanent model failures propagate and fail the workflow', async () => {
  errorMock.generateResults.push(new Error('model exploded'));
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

test('streaming with retries buffers output until an attempt succeeds', async () => {
  bufferedMock.streamPartLists.push(
    [
      { type: 'stream-start', warnings: [] },
      { type: 'text-start', id: 't1' },
      { type: 'text-delta', id: 't1', delta: 'partial garbage from failed attempt' },
      { type: 'error', error: new Error('connection reset') },
    ],
    textStreamParts(['Good', ' answer']),
  );
  const handle = await DBOS.startWorkflow(bufferedStreamWorkflow, { workflowID: randomUUID() })('hi');
  const result = await handle.getResult();
  // Nothing from the failed attempt leaks to the consumer; the successful attempt is flushed after completion as one delta per block.
  assert.deepEqual(result.deltas, ['Good answer']);
  assert.equal(result.text, 'Good answer');
  assert.equal(bufferedMock.streamCalls, 2);
});

test('mid-stream errors fail the model call', async () => {
  failStreamMock.streamPartLists.push([
    { type: 'stream-start', warnings: [] },
    { type: 'text-start', id: 't1' },
    { type: 'text-delta', id: 't1', delta: 'about to fail' },
    { type: 'error', error: new Error('boom') },
  ]);
  const handle = await DBOS.startWorkflow(failStreamWorkflow, { workflowID: randomUUID() })();
  await assert.rejects(handle.getResult(), /boom/);
  assert.equal(failStreamMock.streamCalls, 1);
});

test('cancelling the consumer stream still checkpoints the full model call', async () => {
  cancelMock.streamPartLists.push(textStreamParts(['Hello', ' world']));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(cancelWorkflow, { workflowID })();
  assert.equal(await handle.getResult(), 'cancelled early');
  assert.equal(cancelMock.streamCalls, 1);
  const steps = await DBOS.listWorkflowSteps(workflowID);
  assert.ok(steps !== undefined && steps.some((s) => s.name === 'mock.mock-model.stream'));
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

  // Fork after the stream step: it replays from its checkpoint. Before the fix the step was
  // recorded as an error and the fork threw; now it's a success and replays identically.
  const forked = await DBOS.forkWorkflow<ReturnType<typeof cancelErrorWorkflow>>(workflowID, 1);
  assert.equal(await forked.getResult(), 'cancelled before error');
  assert.equal(cancelErrorMock.streamCalls, 1);
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
  const handle = await DBOS.startWorkflow(orderingWorkflow, { workflowID: randomUUID() })();
  const result = await handle.getResult();
  // The source arrived while the text block was open; content must stay in arrival order (text then source), not [source, text].
  assert.deepEqual(result.types, ['text', 'source']);
  assert.equal(result.text, 'According to the docs');
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
