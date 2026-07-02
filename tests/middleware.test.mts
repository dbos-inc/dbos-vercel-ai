import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { after, before, test } from 'node:test';
import { DBOS } from '@dbos-inc/dbos-sdk';
import { embedMany, generateText, stepCountIs, streamText, tool, wrapEmbeddingModel, wrapLanguageModel } from 'ai';
import { z } from 'zod';
import type { LanguageModelV4StreamPart } from '@ai-sdk/provider';
import { durableCalls, durableEmbeddingCalls } from '../src/index.js';
import {
  MockEmbeddingModel,
  MockLanguageModel,
  textResponse,
  textStreamParts,
  toolCallResponse,
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

const forwardMock = new MockLanguageModel();
const forwardModel = wrapLanguageModel({
  model: forwardMock,
  middleware: durableCalls({ streamKey: 'llm-stream' }),
});

const forwardWorkflow = DBOS.registerWorkflow(
  async (prompt: string) => {
    const result = streamText({ model: forwardModel, prompt });
    return { text: await result.text };
  },
  { name: 'forwardWorkflow' },
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

test('streamKey forwards raw stream parts to a DBOS workflow stream', async () => {
  forwardMock.streamPartLists.push(textStreamParts(['Hi', ' there']));
  const workflowID = randomUUID();
  const handle = await DBOS.startWorkflow(forwardWorkflow, { workflowID })('hi');
  const result = await handle.getResult();
  assert.equal(result.text, 'Hi there');

  const parts: LanguageModelV4StreamPart[] = [];
  for await (const part of DBOS.readStream<LanguageModelV4StreamPart>(workflowID, 'llm-stream')) {
    parts.push(part);
  }
  assert.deepEqual(
    parts.map((p) => p.type),
    ['stream-start', 'response-metadata', 'text-start', 'text-delta', 'text-delta', 'text-end', 'finish'],
  );
  const metadata = parts[1]!;
  assert.ok(metadata.type === 'response-metadata' && metadata.timestamp instanceof Date);
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
