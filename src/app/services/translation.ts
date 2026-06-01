import { Injectable, signal, computed } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Subject, interval, Subscription } from 'rxjs';
import { TranslationResult, TranslationSession, StreamStatus } from '../models/translation';
import { environment } from '../../environments/environment';

@Injectable({ providedIn: 'root' })
export class TranslationService {
  private frameCapture: Subscription | null = null;
  private videoElement: HTMLVideoElement | null = null;
  private canvas = document.createElement('canvas');

  readonly currentWord = signal<string>('');
  readonly currentSentence = signal<string>('');
  readonly confidence = signal<number>(0);
  readonly isTranslating = signal<boolean>(false);

  readonly streamStatus = signal<StreamStatus>({
    isStreaming: false,
    fps: 0,
    latencyMs: 0,
  });

  readonly sessionHistory = signal<TranslationSession[]>([]);

  private currentSession: TranslationSession | null = null;
  readonly translationUpdate$ = new Subject<TranslationResult>();

  constructor(private http: HttpClient) {}

  setVideoElement(video: HTMLVideoElement): void {
    this.videoElement = video;
  }

  startTranslation(): void {
    if (!this.videoElement) return;

    this.currentSession = {
      id: crypto.randomUUID(),
      startedAt: new Date(),
      results: [],
      language: 'LSF',
    };

    this.isTranslating.set(true);
    this.streamStatus.update(s => ({ ...s, isStreaming: true }));

    // Capture and send a frame every ~100ms (10 fps to backend)
    // this.frameCapture = interval(100).subscribe(() => {
    //   this.captureAndSendFrame();
    // });

    // Simulate FPS/latency update (replace with real WebSocket metrics)
    interval(1000).subscribe(() => {
      this.streamStatus.update(s => ({
        ...s,
        fps: 60,
        latencyMs: Math.floor(Math.random() * 8) + 10,
      }));
    });
  }

  stopTranslation(): void {
    this.frameCapture?.unsubscribe();
    this.frameCapture = null;
    this.isTranslating.set(false);
    this.streamStatus.update(s => ({ ...s, isStreaming: false }));

    if (this.currentSession && this.currentSession.results.length > 0) {
      this.sessionHistory.update(h => [this.currentSession!, ...h]);
    }
    this.currentSession = null;
  }

  private captureAndSendFrame(): void {
    if (!this.videoElement) return;

    const { videoWidth, videoHeight } = this.videoElement;
    if (!videoWidth || !videoHeight) return;

    this.canvas.width = videoWidth;
    this.canvas.height = videoHeight;
    const ctx = this.canvas.getContext('2d')!;
    ctx.drawImage(this.videoElement, 0, 0);

    this.canvas.toBlob(blob => {
      if (!blob) return;
      const formData = new FormData();
      formData.append('frame', blob, 'frame.jpg');

      const t0 = performance.now();
      this.http.post<{ word: string; sentence: string; confidence: number }>(
        `${environment.apiUrl}/predict`,
        formData
      ).subscribe({
        next: (res) => {
          const latency = Math.round(performance.now() - t0);
          const result: TranslationResult = {
            word: res.word,
            sentence: res.sentence,
            confidence: res.confidence,
            timestamp: new Date(),
          };

          this.currentWord.set(res.word);
          this.currentSentence.set(res.sentence);
          this.confidence.set(res.confidence);
          this.streamStatus.update(s => ({ ...s, latencyMs: latency }));
          this.currentSession?.results.push(result);
          this.translationUpdate$.next(result);
        },
        error: (err) => console.error('Prediction error:', err),
      });
    }, 'image/jpeg', 0.85);
  }
}