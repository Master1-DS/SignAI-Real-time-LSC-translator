import { Injectable, signal } from '@angular/core';
import * as ort from 'onnxruntime-web';
import { Subject, Subscription } from 'rxjs';
import { TranslationResult, TranslationSession, StreamStatus } from '../models/translation';
import { LandmarkService } from './landmark';
import { SegmentationService, SignSegment } from './segmentation';
import { LABELS } from '../models/labels';

const NUM_FRAMES = 87;
const DIM_HAND   = 63;
const DIM_POSE   = 132;
const DIM_FRAME  = DIM_HAND * 2 + DIM_POSE;
const L_SHOULDER = 11;
const R_SHOULDER = 12;

const VOTE_WINDOW   = 5;
const VOTE_MAJORITY = 3;
const VOTE_MIN_CONF = 0.55;

export interface RawPrediction {
  word:       string;
  confidence: number;
  validated:  boolean;
  isFinal:    boolean;
  timestamp:  Date;
}

export interface VoteEntry {
  word:       string;
  confidence: number;
}

@Injectable({ providedIn: 'root' })
export class TranslationService {

  readonly currentWord     = signal<string>('');
  readonly currentSentence = signal<string>('');
  readonly confidence      = signal<number>(0);
  readonly isTranslating   = signal<boolean>(false);
  readonly streamStatus    = signal<StreamStatus>({ isStreaming: false, fps: 0, latencyMs: 0 });
  readonly sessionHistory  = signal<TranslationSession[]>([]);
  readonly rawPredictions  = signal<RawPrediction[]>([]);
  readonly votedWord       = signal<string>('');
  readonly votedConf       = signal<number>(0);

  readonly translationUpdate$ = new Subject<TranslationResult>();

  private model: ort.InferenceSession | null = null;
  private currentSession: TranslationSession | null = null;
  private segmentSub: Subscription | null = null;
  private fpsSub: Subscription | null = null;
  private frameCount  = 0;
  private lastFpsTime = performance.now();
  private voteQueue: VoteEntry[] = [];

  // ── File d'attente d'inférence ───────────────────────────────────────────────
  /**
   * Une seule inférence tourne à la fois.
   * Si un segment arrive pendant une inférence en cours :
   *   - segment glissant  → remplace le pending (le plus récent prime)
   *   - segment final     → remplace le pending ET est marqué prioritaire
   * Après l'inférence courante, on traite le pending s'il existe.
   * Après un segment final, le vote est vidé.
   */
  private inferenceRunning = false;
  private pendingSegment: SignSegment | null = null;

  constructor(
    private landmarkService:     LandmarkService,
    private segmentationService: SegmentationService,
  ) {
    this.initModel();
  }

  private async initModel(): Promise<void> {
    try {
      ort.env.wasm.wasmPaths  = '/ort-wasm/';
      ort.env.wasm.numThreads = 1;
      this.model = await ort.InferenceSession.create('/models/best_model.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      console.log('✅ ONNX chargé:', this.model.inputNames);
    } catch (e) {
      console.error('❌ ONNX:', e);
    }
  }

  setVideoElement(_v: HTMLVideoElement): void {}

  startTranslation(): void {
    this.voteQueue        = [];
    this.inferenceRunning = false;
    this.pendingSegment   = null;
    this.frameCount       = 0;
    this.lastFpsTime      = performance.now();
    this.currentWord.set('');
    this.currentSentence.set('');
    this.confidence.set(0);
    this.votedWord.set('');
    this.votedConf.set(0);
    this.rawPredictions.set([]);

    this.currentSession = {
      id: crypto.randomUUID(),
      startedAt: new Date(),
      results: [],
      language: 'LSC',
    };

    this.isTranslating.set(true);
    this.streamStatus.update(s => ({ ...s, isStreaming: true }));

    this.segmentationService.start();

    this.segmentSub = this.segmentationService.segment$.subscribe(seg => {
      this.enqueueSegment(seg);
    });

    this.fpsSub = this.landmarkService.frame$.subscribe(() => {
      this.frameCount++;
      const now = performance.now();
      if (now - this.lastFpsTime >= 1000) {
        this.streamStatus.update(s => ({ ...s, fps: this.frameCount }));
        this.frameCount  = 0;
        this.lastFpsTime = now;
      }
    }) as unknown as Subscription;
  }

  stopTranslation(): void {
    this.segmentSub?.unsubscribe();
    this.segmentSub = null;
    this.fpsSub?.unsubscribe();
    this.fpsSub = null;
    this.segmentationService.stop();
    this.voteQueue        = [];
    this.pendingSegment   = null;
    this.inferenceRunning = false;
    this.isTranslating.set(false);
    this.streamStatus.update(s => ({ ...s, isStreaming: false, fps: 0 }));

    if (this.currentSession?.results.length) {
      this.sessionHistory.update(h => [this.currentSession!, ...h]);
    }
    this.currentSession = null;
  }

  // ── Gestion de la file d'attente ─────────────────────────────────────────────

  private enqueueSegment(seg: SignSegment): void {
    if (!this.inferenceRunning) {
      // Aucune inférence en cours → on démarre immédiatement
      this.runInferencePipeline(seg);
    } else {
      // Une inférence tourne déjà :
      // - Si le nouveau segment est final, il prend la priorité absolue
      //   même si un segment final attendait déjà (ne devrait pas arriver
      //   mais on se protège)
      // - Si c'est un segment glissant, il remplace simplement le pending
      //   car le plus récent est toujours plus pertinent
      if (seg.isFinal || !this.pendingSegment?.isFinal) {
        this.pendingSegment = seg;
      }
      // Si un segment final attend déjà et qu'un glissant arrive,
      // on ignore le glissant — le final est prioritaire
    }
  }

  private async runInferencePipeline(seg: SignSegment): Promise<void> {
    this.inferenceRunning = true;
    try {
      await this.processSegment(seg);
    } finally {
      this.inferenceRunning = false;

      // Traite le segment en attente s'il existe
      if (this.pendingSegment) {
        const next          = this.pendingSegment;
        this.pendingSegment = null;
        this.runInferencePipeline(next);
      }
    }
  }

  // ── Pipeline d'inférence ─────────────────────────────────────────────────────

  private async processSegment(seg: SignSegment): Promise<void> {
    if (!this.model) return;

    const t0         = performance.now();
    const resampled  = this.resampleSegment(seg.frames, NUM_FRAMES);
    const normalized = resampled.map(f => this.normalizeKeypoints(f));

    const flat = new Float32Array(NUM_FRAMES * DIM_FRAME);
    for (let i = 0; i < NUM_FRAMES; i++) flat.set(normalized[i], i * DIM_FRAME);

    const tensor = new ort.Tensor('float32', flat, [1, NUM_FRAMES, DIM_FRAME]);

    try {
      const results = await this.model.run({ [this.model.inputNames[0]]: tensor });
      const latency = Math.round(performance.now() - t0);
      this.streamStatus.update(s => ({ ...s, latencyMs: latency }));
      this.handlePrediction(results, seg.isFinal);
    } catch (e) {
      console.error('❌ Inférence:', e);
    }
  }

  // ── Vote majoritaire ─────────────────────────────────────────────────────────

  private handlePrediction(
    results: Record<string, ort.Tensor>,
    isFinal: boolean,
  ): void {
    const logits = results[this.model!.outputNames[0]].data as Float32Array;
    const probs  = this.softmax(logits);
    const idx    = this.argmax(probs);
    const conf   = probs[idx];
    const word   = LABELS[idx] ?? `Signe_${idx}`;

    this.rawPredictions.update(list => [
      { word, confidence: conf, validated: false, isFinal, timestamp: new Date() },
      ...list.slice(0, 29),
    ]);

    if (conf == 1.0) {
      this.validateWord(word, conf);
      this.voteQueue = [];
      return;
    }
    if (conf >= VOTE_MIN_CONF) {
      this.voteQueue.push({ word, confidence: conf });
      if (this.voteQueue.length > VOTE_WINDOW) this.voteQueue.shift();

      const winner = this.getMajorityVote();
      if (winner) {
        this.validateWord(winner.word, winner.confidence);
        this.voteQueue = [];
      }
    }

    // Après la dernière inférence d'un segment final :
    // vide le vote pour ne pas polluer le mot suivant
    // avec des prédictions du signe qui vient de se terminer
    if (isFinal) {
      this.voteQueue = [];
    }
  }

  private getMajorityVote(): VoteEntry | null {
    if (this.voteQueue.length < VOTE_MAJORITY) return null;

    const counts = new Map<string, { count: number; totalConf: number }>();
    for (const v of this.voteQueue) {
      const entry = counts.get(v.word) ?? { count: 0, totalConf: 0 };
      counts.set(v.word, {
        count:     entry.count + 1,
        totalConf: entry.totalConf + v.confidence,
      });
    }

    for (const [word, { count, totalConf }] of counts) {
      if (count >= VOTE_MAJORITY) {
        return { word, confidence: totalConf / count };
      }
    }
    return null;
  }

  private validateWord(word: string, conf: number): void {
    this.rawPredictions.update(list => {
      const idx = list.findIndex(p => p.word === word && !p.validated);
      if (idx === -1) return list;
      const updated = [...list];
      updated[idx]  = { ...updated[idx], validated: true };
      return updated;
    });

    this.votedWord.set(word);
    this.votedConf.set(conf);
    this.currentWord.set(word);
    this.confidence.set(conf);

    const sentence = [this.currentSentence(), word].filter(Boolean).join(' ');
    this.currentSentence.set(sentence);

    const result: TranslationResult = {
      word, sentence, confidence: conf, timestamp: new Date(),
    };
    this.currentSession?.results.push(result);
    this.translationUpdate$.next(result);
  }

  // ── Utilitaires ──────────────────────────────────────────────────────────────

  private resampleSegment(frames: Float32Array[], target: number): Float32Array[] {
    const T = frames.length;
    if (T === target) return frames;
    return Array.from({ length: target }, (_, i) => {
      const pos = (i / (target - 1)) * (T - 1);
      const lo  = Math.floor(pos), hi = Math.min(lo + 1, T - 1);
      const t   = pos - lo;
      if (t === 0 || lo === hi) return frames[lo].slice();
      const out = new Float32Array(DIM_FRAME);
      for (let j = 0; j < DIM_FRAME; j++)
        out[j] = frames[lo][j] * (1 - t) + frames[hi][j] * t;
      return out;
    });
  }

  private normalizeKeypoints(kp: Float32Array): Float32Array {
    const norm = kp.slice(), ps = DIM_HAND * 2;
    const lSx  = ps + L_SHOULDER * 4, rSx = ps + R_SHOULDER * 4;
    const dx   = kp[rSx] - kp[lSx], dy = kp[rSx+1] - kp[lSx+1];
    const dist = Math.max(Math.sqrt(dx*dx + dy*dy), 1e-6);
    const cx   = (kp[lSx] + kp[rSx]) / 2, cy = (kp[lSx+1] + kp[rSx+1]) / 2;
    for (const off of [0, DIM_HAND] as const)
      for (let lm = 0; lm < 21; lm++) {
        const ix = off + lm * 3;
        norm[ix] = (kp[ix] - cx) / dist;
        norm[ix+1] = (kp[ix+1] - cy) / dist;
      }
    for (let lm = 0; lm < 33; lm++) {
      const ix = ps + lm * 4;
      norm[ix] = (kp[ix] - cx) / dist;
      norm[ix+1] = (kp[ix+1] - cy) / dist;
    }
    return norm;
  }

  private argmax(arr: Float32Array): number {
    let idx = 0;
    for (let i = 1; i < arr.length; i++) if (arr[i] > arr[idx]) idx = i;
    return idx;
  }

  private softmax(arr: Float32Array): Float32Array {
    const max  = Math.max(...arr);
    const exps = Float32Array.from(arr, v => Math.exp(v - max));
    const sum  = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum) as unknown as Float32Array;
  }
}