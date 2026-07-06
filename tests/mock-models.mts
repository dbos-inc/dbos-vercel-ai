import type {
  EmbeddingModelV4,
  EmbeddingModelV4CallOptions,
  EmbeddingModelV4Result,
  ImageModelV4,
  ImageModelV4CallOptions,
  ImageModelV4Result,
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamPart,
  LanguageModelV4StreamResult,
  LanguageModelV4Usage,
} from '@ai-sdk/provider';
import { dynamicTool, jsonSchema, tool, type ToolSet } from 'ai';
import { z } from 'zod';

export function usage(inputTokens = 10, outputTokens = 20): LanguageModelV4Usage {
  return {
    inputTokens: { total: inputTokens, noCache: undefined, cacheRead: undefined, cacheWrite: undefined },
    outputTokens: { total: outputTokens, text: undefined, reasoning: undefined },
  };
}

export function finishReason(
  unified: LanguageModelV4FinishReason['unified'] = 'stop',
): LanguageModelV4FinishReason {
  return { unified, raw: undefined };
}

export function contentResponse(
  content: LanguageModelV4Content[],
  finish: LanguageModelV4FinishReason = finishReason(),
): LanguageModelV4GenerateResult {
  return {
    content,
    finishReason: finish,
    usage: usage(),
    warnings: [],
    response: { id: 'resp-1', timestamp: new Date('2026-07-02T12:00:00Z'), modelId: 'mock-model' },
  };
}

export function textResponse(text: string): LanguageModelV4GenerateResult {
  return contentResponse([{ type: 'text', text }]);
}

export function toolCallResponse(toolName: string, input: string): LanguageModelV4GenerateResult {
  return contentResponse(
    [{ type: 'tool-call', toolCallId: 'call-1', toolName, input }],
    finishReason('tool-calls'),
  );
}

// Multiple tool calls in one response → the AI SDK executes them in parallel (Promise.all).
export function toolCallsResponse(calls: { toolName: string; input: string }[]): LanguageModelV4GenerateResult {
  return contentResponse(
    calls.map((c, i) => ({ type: 'tool-call', toolCallId: `call-${i}`, toolName: c.toolName, input: c.input })),
    finishReason('tool-calls'),
  );
}

export function textStreamParts(deltas: string[]): LanguageModelV4StreamPart[] {
  return [
    { type: 'stream-start', warnings: [] },
    { type: 'response-metadata', id: 'resp-1', timestamp: new Date('2026-07-02T12:00:00Z'), modelId: 'mock-model' },
    { type: 'text-start', id: 't1' },
    ...deltas.map((delta): LanguageModelV4StreamPart => ({ type: 'text-delta', id: 't1', delta })),
    { type: 'text-end', id: 't1' },
    { type: 'finish', finishReason: finishReason(), usage: usage() },
  ];
}

export class MockLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = 'v4';
  readonly provider = 'mock';
  readonly modelId = 'mock-model';
  readonly supportedUrls: Record<string, RegExp[]> = {};

  // Queue an Error to fail that doGenerate call; queue an 'error' stream part to fail that doStream partway,
  // or an Error in a part list to fail the stream itself (read() rejects) at that point.
  generateResults: (LanguageModelV4GenerateResult | Error)[] = [];
  streamPartLists: (LanguageModelV4StreamPart | Error)[][] = [];
  generateCalls = 0;
  streamCalls = 0;
  generateOptions: LanguageModelV4CallOptions[] = [];

  async doGenerate(options: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult> {
    this.generateCalls++;
    this.generateOptions.push(options);
    const result = this.generateResults.shift();
    if (result === undefined) {
      throw new Error('MockLanguageModel: no generate responses left');
    }
    if (result instanceof Error) {
      throw result;
    }
    return result;
  }

  // Queue an Error here to make doStream itself reject (after a tick) instead of returning a stream.
  streamCallErrors: Error[] = [];
  // Counts model-stream cancellations (the middleware tearing down the provider connection).
  streamCancellations = 0;

  async doStream(options: LanguageModelV4CallOptions): Promise<LanguageModelV4StreamResult> {
    this.streamCalls++;
    const callError = this.streamCallErrors.shift();
    if (callError) {
      // Reject after a tick so a consumer cancelling right away wins the race.
      await new Promise((resolve) => setImmediate(resolve));
      throw callError;
    }
    const parts = this.streamPartLists.shift();
    if (parts === undefined) {
      throw new Error('MockLanguageModel: no stream responses left');
    }
    return {
      stream: new ReadableStream<LanguageModelV4StreamPart>({
        async start(controller) {
          try {
            for (const part of parts) {
              if (options.abortSignal?.aborted) {
                // Real providers reject reads once the call's abortSignal fires.
                controller.error(Object.assign(new Error('The operation was aborted'), { name: 'AbortError' }));
                return;
              }
              if (part instanceof Error) {
                // Stream-level failure: reads reject, unlike an 'error' part.
                controller.error(part);
                return;
              }
              controller.enqueue(part);
              // Yield to the event loop so parts arrive asynchronously, as from a network.
              await new Promise((resolve) => setImmediate(resolve));
            }
            controller.close();
          } catch {
            // Cancelled mid-emission: enqueue/close throw once the stream is torn down.
          }
        },
        cancel: () => {
          this.streamCancellations++;
        },
      }),
      request: { body: 'mock-request' },
      response: { headers: { 'x-mock': '1' } },
    };
  }
}

export class MockEmbeddingModel implements EmbeddingModelV4 {
  readonly specificationVersion = 'v4';
  readonly provider = 'mock';
  readonly modelId = 'mock-embed';
  readonly maxEmbeddingsPerCall: number | undefined;
  readonly supportsParallelCalls = true;

  embedCalls = 0;

  // A finite maxEmbeddingsPerCall makes embedMany split large inputs into batches (parallel by default).
  constructor(maxEmbeddingsPerCall?: number) {
    this.maxEmbeddingsPerCall = maxEmbeddingsPerCall;
  }

  async doEmbed(options: EmbeddingModelV4CallOptions): Promise<EmbeddingModelV4Result> {
    this.embedCalls++;
    return {
      embeddings: options.values.map((_, i) => [i, i + 0.5, i + 0.25]),
      usage: { tokens: options.values.length * 3 },
      warnings: [],
    };
  }
}

export const IMAGE_BYTES = [137, 80, 78, 71, 13, 10, 26, 10];

export class MockImageModel implements ImageModelV4 {
  readonly specificationVersion = 'v4';
  readonly provider = 'mock';
  readonly modelId = 'mock-image';
  readonly maxImagesPerCall: number;

  generateCalls = 0;
  // Queue an images array to override the default bytes (e.g. a spec-violating mixed string/bytes batch).
  imageOverrides: (string | Uint8Array)[][] = [];

  // The default of 1 forces generateImage to split n>1 into parallel batches.
  constructor(maxImagesPerCall = 1) {
    this.maxImagesPerCall = maxImagesPerCall;
  }

  async doGenerate(options: ImageModelV4CallOptions): Promise<ImageModelV4Result> {
    this.generateCalls++;
    // Tag each generated image with this call's ordinal so a reordering on replay is detectable.
    const images =
      this.imageOverrides.shift() ??
      Array.from({ length: options.n }, () => new Uint8Array([...IMAGE_BYTES, this.generateCalls]));
    return {
      images: images as ImageModelV4Result['images'],
      warnings: [],
      response: { timestamp: new Date('2026-07-02T12:00:00Z'), modelId: 'mock-image', headers: {} },
    };
  }
}

// Mimics an @ai-sdk/mcp client: tools() lists tools over the "wire"; execute runs a tool.
export class MockMCPClient {
  listCalls = 0;
  weatherCalls = 0;
  timeCalls = 0;
  toolsOptionsLog: unknown[] = [];

  get executeCalls(): number {
    return this.weatherCalls + this.timeCalls;
  }

  async tools(options?: { schemas?: Record<string, unknown> }): Promise<ToolSet> {
    this.listCalls++;
    this.toolsOptionsLog.push(options);
    const all: ToolSet = {
      getWeather: tool({
        description: 'Get the weather for a city',
        inputSchema: z.object({ city: z.string() }),
        execute: async ({ city }: { city: string }) => {
          this.weatherCalls++;
          return `sunny in ${city}`;
        },
      }),
      getTime: tool({
        description: 'Get the current time in a city',
        inputSchema: z.object({ city: z.string() }),
        execute: async ({ city }: { city: string }) => {
          this.timeCalls++;
          return `noon in ${city}`;
        },
      }),
    };
    if (!options?.schemas) return all;
    // Schemas mode subsets like @ai-sdk/mcp: only explicitly listed tools are returned.
    return Object.fromEntries(Object.entries(all).filter(([name]) => name in options.schemas!));
  }

  async close(): Promise<void> {}
}

// Mimics @ai-sdk/mcp's rebuilt tools: dynamicTool with title/metadata/toModelOutput plus a spread _meta.
export class RichMockMCPClient {
  screenshotCalls = 0;

  async tools(): Promise<ToolSet> {
    const screenshot = dynamicTool({
      description: 'Take a screenshot',
      title: 'Screenshot',
      metadata: { clientName: 'mock-mcp', toolName: 'screenshot' },
      inputSchema: jsonSchema({ type: 'object', properties: {}, additionalProperties: false }),
      execute: async () => {
        this.screenshotCalls++;
        return {
          content: [
            { type: 'text', text: 'took screenshot' },
            { type: 'image', data: 'QUJD', mimeType: 'image/png' },
          ],
        };
      },
      toModelOutput: ({ output }) => {
        const result = output as { content: { type: string; text?: string; data?: string; mimeType?: string }[] };
        return {
          type: 'content',
          value: result.content.map((part) =>
            part.type === 'image'
              ? { type: 'file' as const, mediaType: part.mimeType!, data: { type: 'data' as const, data: part.data! } }
              : // Uppercase distinguishes this converter from the middleware's built-in conversion (which keeps text as-is).
                { type: 'text' as const, text: part.text!.toUpperCase() },
          ),
        };
      },
    });
    return { screenshot: Object.assign(screenshot, { _meta: { 'mcp/app': { uri: 'ui://screenshot' } } }) };
  }

  async close(): Promise<void> {}
}
