import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { after, before, test } from 'node:test';
import { DBOS } from '@dbos-inc/dbos-sdk';
import { Client as PgClient } from 'pg';
import { embedMany, generateText, stepCountIs, streamText, tool, wrapEmbeddingModel, wrapLanguageModel } from 'ai';
import { z } from 'zod';
import { durableCalls, durableEmbeddingCalls } from '../src/index.js';
import {
  contentResponse,
  MockEmbeddingModel,
  MockLanguageModel,
  textResponse,
  textStreamParts,
  toolCallResponse,
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

const retryMock = new MockLanguageModel();
const retryModel = wrapLanguageModel({
  model: retryMock,
  middleware: durableCalls({ retriesAllowed: true, maxAttempts: 3, intervalSeconds: 0 }),
});

const retryWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    // maxRetries: 0 disables the AI SDK's own retry layer so the test observes
    // DBOS step retries in isolation.
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
    // cancel() awaits the in-flight step, so the model call is durably checkpointed
    // before the workflow proceeds — no sleep needed to avoid racing the checkpoint.
    await reader.cancel();
    return 'cancelled early';
  },
  { name: 'cancelWorkflow' },
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

// Streams a `source` part while a text block is still open, to check the
// accumulator keeps arrival order ([text, source]) rather than pushing the
// complete part ahead of the not-yet-closed text.
const orderingMock = new MockLanguageModel();
const orderingModel = wrapLanguageModel({
  model: orderingMock,
  // Buffered (retries) mode so the consumer sees the parts synthesized from the
  // accumulated/checkpointed content — i.e. this asserts the accumulator's order.
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

  // Fork after the model-call step: the workflow function re-executes, but the
  // model call is replayed from its checkpoint. No mock responses are queued, so
  // any real model call would throw.
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

  // Fork after step 0 (first model call) and step 1 (tool step), i.e. mid-loop:
  // both replay from checkpoints and only the second model call re-executes.
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
  // Nothing from the failed attempt leaks to the consumer; the successful
  // attempt is flushed after completion as one delta per text block.
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
  // The source arrived while the text block was open; content must stay in arrival
  // order (text then source), not [source, text].
  assert.deepEqual(result.types, ['text', 'source']);
  assert.equal(result.text, 'According to the docs');
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

  // Simulate a crash that lost the completion: flip the workflow back to
  // PENDING (the same technique the DBOS SDK's own recovery tests use), then
  // relaunch. Launch-time recovery re-executes the workflow function; the model
  // call and the recv must both replay from checkpoints (no mock responses are
  // queued and no message is re-sent, so real re-execution would fail).
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
