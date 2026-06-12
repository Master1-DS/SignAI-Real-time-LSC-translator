export interface TranslationResult {
  word: string;
  sentence: string;
  confidence: number; // 0 to 1
  timestamp: Date;
}

export interface StreamStatus {
  isStreaming: boolean;
  fps: number;
  latencyMs: number;
}

export interface TranslationSession {
  id: string;
  startedAt: Date;
  results: TranslationResult[];
  language: 'LSC' | 'ASL' | 'BSL';
}
