import type {
  EmbeddingModelV4,
  EmbeddingModelV4CallOptions,
  EmbeddingModelV4Result,
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamPart,
  LanguageModelV4StreamResult,
  LanguageModelV4Usage,
} from '@ai-sdk/provider';

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

  // Queue an Error to make that doGenerate call fail; queue a stream part of
  // type 'error' to make that doStream call fail partway through.
  generateResults: (LanguageModelV4GenerateResult | Error)[] = [];
  streamPartLists: LanguageModelV4StreamPart[][] = [];
  generateCalls = 0;
  streamCalls = 0;

  async doGenerate(_options: LanguageModelV4CallOptions): Promise<LanguageModelV4GenerateResult> {
    this.generateCalls++;
    const result = this.generateResults.shift();
    if (result === undefined) {
      throw new Error('MockLanguageModel: no generate responses left');
    }
    if (result instanceof Error) {
      throw result;
    }
    return result;
  }

  async doStream(_options: LanguageModelV4CallOptions): Promise<LanguageModelV4StreamResult> {
    this.streamCalls++;
    const parts = this.streamPartLists.shift();
    if (parts === undefined) {
      throw new Error('MockLanguageModel: no stream responses left');
    }
    return {
      stream: new ReadableStream<LanguageModelV4StreamPart>({
        async start(controller) {
          for (const part of parts) {
            controller.enqueue(part);
            // Yield to the event loop so parts arrive asynchronously, as from a network.
            await new Promise((resolve) => setImmediate(resolve));
          }
          controller.close();
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
  readonly maxEmbeddingsPerCall = undefined;
  readonly supportsParallelCalls = true;

  embedCalls = 0;

  async doEmbed(options: EmbeddingModelV4CallOptions): Promise<EmbeddingModelV4Result> {
    this.embedCalls++;
    return {
      embeddings: options.values.map((_, i) => [i, i + 0.5, i + 0.25]),
      usage: { tokens: options.values.length * 3 },
      warnings: [],
    };
  }
}
