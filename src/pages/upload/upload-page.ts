import {
  Component, ElementRef, ViewChild, signal, computed, OnDestroy
} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  HandLandmarker, PoseLandmarker, FilesetResolver,
  HandLandmarkerResult, PoseLandmarkerResult,
} from '@mediapipe/tasks-vision';
import * as ort from 'onnxruntime-web';
import { LABELS } from '../../app/models/labels';

const NUM_FRAMES = 87;
const DIM_HAND   = 63;
const DIM_POSE   = 132;
const DIM_FRAME  = DIM_HAND * 2 + DIM_POSE;
const L_SHOULDER = 11;
const R_SHOULDER = 12;

type PageState = 'idle' | 'loading-models' | 'ready' | 'processing' | 'done' | 'error';

export interface PredictionResult {
    word: string;
    confidence: number;
    allScores: { label: string; score: number }[];
}

@Component({
    selector: 'app-upload-page',
    standalone: true,
    imports: [CommonModule],
    styles: [`
        @keyframes fade-up {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .result-card { animation: fade-up 0.25s ease forwards; }
        .progress-fill { transition: width 0.15s ease; }
        .conf-fill { transition: width 0.5s cubic-bezier(0.34, 1.56, 0.64, 1); }
    `],
    templateUrl: './upload-page.html',
})
export class UploadPageComponent implements OnDestroy {

    @ViewChild('videoPlayer', { static: false }) videoRef?: ElementRef<HTMLVideoElement>;
    @ViewChild('fileInput',   { static: true  }) fileInputRef!: ElementRef<HTMLInputElement>;

    readonly NUM_FRAMES = NUM_FRAMES;

    readonly state              = signal<PageState>('idle');
    readonly fileName           = signal<string>('');
    readonly videoDuration      = signal<number>(0);
    readonly videoUrl           = signal<string | null>(null);
    readonly videoProgress      = signal<number>(0);
    readonly isPlaying          = signal<boolean>(false);
    readonly isMuted            = signal<boolean>(false);
    readonly currentTimeLabel   = signal<string>('0:00');
    readonly durationLabel      = signal<string>('0:00');
    readonly processingProgress = signal<number>(0);
    readonly currentFrame       = signal<number>(0);
    readonly totalFrames        = signal<number>(0);
    readonly result             = signal<PredictionResult | null>(null);
    readonly errorMessage       = signal<string>('');

    private handLandmarker: HandLandmarker | null = null;
    private poseLandmarker: PoseLandmarker | null = null;
    private model: ort.InferenceSession | null = null;
    private objectUrl: string | null = null;
    private offscreenCanvas = document.createElement('canvas');

    onFileSelected(event: Event): void {
        const file = (event.target as HTMLInputElement).files?.[0];
        if (!file) return;
        if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
        this.objectUrl = URL.createObjectURL(file);
        this.fileName.set(file.name);
        this.videoUrl.set(this.objectUrl);
        this.result.set(null);
        this.processingProgress.set(0);
        this.currentFrame.set(0);
        this.state.set('ready');
        (event.target as HTMLInputElement).value = '';
    }

    onVideoLoaded(): void {
        const v = this.videoRef?.nativeElement;
        if (!v) return;
        this.videoDuration.set(v.duration);
        this.durationLabel.set(this.formatTime(v.duration));
    }

    onTimeUpdate(): void {
        const v = this.videoRef?.nativeElement;
        if (!v || !v.duration) return;
        this.videoProgress.set((v.currentTime / v.duration) * 100);
        this.currentTimeLabel.set(this.formatTime(v.currentTime));
    }

    togglePlay(): void {
        const v = this.videoRef?.nativeElement;
        if (!v) return;
        if (v.paused) { v.play(); this.isPlaying.set(true); }
        else          { v.pause(); this.isPlaying.set(false); }
    }

    toggleMute(): void {
        const v = this.videoRef?.nativeElement;
        if (!v) return;
        v.muted = !v.muted;
        this.isMuted.set(v.muted);
    }

    skip(s: number): void {
        const v = this.videoRef?.nativeElement;
        if (!v) return;
        v.currentTime = Math.max(0, Math.min(v.duration, v.currentTime + s));
    }

    seekVideo(e: MouseEvent): void {
        const v = this.videoRef?.nativeElement;
        if (!v) return;
        v.currentTime = (e.offsetX / (e.currentTarget as HTMLElement).offsetWidth) * v.duration;
    }

    async startAnalysis(): Promise<void> {
        if (!this.objectUrl || this.videoDuration() <= 0) {
            this.errorMessage.set('Vidéo non chargée correctement.');
            this.state.set('error');
            return;
        }

        this.state.set('loading-models');
        try {
            await this.ensureModelsLoaded();
        } catch (e: any) {
            this.errorMessage.set('Chargement modèles : ' + (e?.message ?? String(e)));
            this.state.set('error');
            return;
        }

        this.state.set('processing');
        this.result.set(null);
        this.processingProgress.set(0);

        try {
            const allFeatures = await this.extractAllFrames();
            const resampled = this.resampleSegment(allFeatures, NUM_FRAMES);
            const normalized = resampled.map(f => this.normalizeKeypoints(f));
            const prediction = await this.runInference(normalized);
            this.result.set(prediction);
            this.state.set('done');
        } catch (e: any) {
            this.errorMessage.set(e?.message ?? 'Erreur inconnue pendant l\'analyse.');
            this.state.set('error');
        }
    }

    private async ensureModelsLoaded(): Promise<void> {
        if (!this.handLandmarker || !this.poseLandmarker) {
            const vision = await FilesetResolver.forVisionTasks('/mediapipe-wasm');
            [this.handLandmarker, this.poseLandmarker] = await Promise.all([
                HandLandmarker.createFromOptions(vision, {
                    baseOptions: { modelAssetPath: '/models/hand_landmarker.task', delegate: 'GPU' },
                    runningMode: 'IMAGE', numHands: 2,
                    minHandDetectionConfidence: 0.5,
                    minHandPresenceConfidence: 0.5,
                    minTrackingConfidence: 0.4,
                }),
                PoseLandmarker.createFromOptions(vision, {
                    baseOptions: { modelAssetPath: '/models/pose_landmarker_lite.task', delegate: 'GPU' },
                    runningMode: 'IMAGE', numPoses: 1,
                    minPoseDetectionConfidence: 0.4,
                    minPosePresenceConfidence: 0.4,
                    minTrackingConfidence: 0.4,
                }),
            ]);
        }

        if (!this.model) {
            ort.env.wasm.wasmPaths  = '/ort-wasm/';
            ort.env.wasm.numThreads = 1;
            this.model = await ort.InferenceSession.create('/models/best_model.onnx', {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
            });
            console.log('ONNX chargé:', this.model.inputNames);
        }
    }

    private async extractAllFrames(): Promise<Float32Array[]> {
        const offVideo       = document.createElement('video');
        offVideo.src         = this.objectUrl!;
        offVideo.muted       = true;
        offVideo.playsInline = true;

        await new Promise<void>((res, rej) => {
            offVideo.onloadedmetadata = () => res();
            offVideo.onerror = () => rej(new Error('Vidéo hors-écran introuvable.'));
        });

        const fps   = 30;
        const total = Math.floor(offVideo.duration * fps);
        this.totalFrames.set(total);

        const canvas  = this.offscreenCanvas;
        canvas.width  = 640;
        canvas.height = 360;
        const ctx     = canvas.getContext('2d')!;

        const features: Float32Array[] = [];

        for (let f = 0; f < total; f++) {
            offVideo.currentTime = f / fps;
            await new Promise<void>((res, rej) => {
                offVideo.onseeked = () => res();
                offVideo.onerror  = () => rej(new Error(`Seek impossible frame ${f}`));
            });

            ctx.drawImage(offVideo, 0, 0, canvas.width, canvas.height);
            const hand = this.handLandmarker!.detect(canvas as any);
            const pose = this.poseLandmarker!.detect(canvas as any);
            features.push(this.extractKeypoints(hand, pose));

            this.currentFrame.set(f + 1);
            this.processingProgress.set(Math.round(((f + 1) / total) * 100));

            if (f % 5 === 0) await new Promise(res => setTimeout(res, 0));
        }

        return features;
    }

    private async runInference(normalized: Float32Array[]): Promise<PredictionResult> {
        if (!this.model) throw new Error('Modèle ONNX non chargé.');

        const flat = new Float32Array(NUM_FRAMES * DIM_FRAME);
        for (let i = 0; i < NUM_FRAMES; i++) flat.set(normalized[i], i * DIM_FRAME);

        const tensor = new ort.Tensor('float32', flat, [1, NUM_FRAMES, DIM_FRAME]);
        const out    = await this.model.run({ [this.model.inputNames[0]]: tensor });
        const logits = out[this.model.outputNames[0]].data as Float32Array;
        const probs  = this.softmax(logits);

        const indexed = Array.from(probs)
            .map((score, i) => ({ label: LABELS[i] ?? `Signe_${i}`, score }))
            .sort((a, b) => b.score - a.score);

        return {
            word:       indexed[0].label,
            confidence: indexed[0].score,
            allScores:  indexed,
        };
    }

    private extractKeypoints(h: HandLandmarkerResult, p: PoseLandmarkerResult): Float32Array {
        const lh = new Float32Array(DIM_HAND), rh = new Float32Array(DIM_HAND);
        const pose = new Float32Array(DIM_POSE);

        for (let i = 0; i < h.landmarks.length; i++) {
            const lm   = h.landmarks[i];
            const side = h.handedness?.[i]?.[0]?.categoryName ?? '';
            const vec  = new Float32Array(DIM_HAND);
            for (let j = 0; j < 21; j++) {
                vec[j*3] = lm[j].x; vec[j*3+1] = lm[j].y; vec[j*3+2] = lm[j].z;
            }
            if (side === 'Left') lh.set(vec); else if (side === 'Right') rh.set(vec);
        }

        if (p.landmarks.length > 0) {
            const lm = p.landmarks[0];
            for (let j = 0; j < 33; j++) {
                pose[j*4] = lm[j].x; pose[j*4+1] = lm[j].y;
                pose[j*4+2] = lm[j].z; pose[j*4+3] = (lm[j] as any).visibility ?? 0;
            }
        }

        const out = new Float32Array(DIM_FRAME);
        out.set(lh, 0); out.set(rh, DIM_HAND); out.set(pose, DIM_HAND * 2);
        return out;
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

    formatTime(s: number): string {
        if (!s || isNaN(s)) return '0:00';
        const m = Math.floor(s / 60), sec = Math.floor(s % 60);
        return `${m}:${sec.toString().padStart(2, '0')}`;
    }

    reset(): void {
        if (this.objectUrl) { URL.revokeObjectURL(this.objectUrl); this.objectUrl = null; }
        this.videoUrl.set(null);
        this.fileName.set('');
        this.videoDuration.set(0);
        this.result.set(null);
        this.processingProgress.set(0);
        this.currentFrame.set(0);
        this.totalFrames.set(0);
        this.isPlaying.set(false);
        this.state.set('idle');
    }

    ngOnDestroy(): void {
        if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
        this.handLandmarker?.close();
        this.poseLandmarker?.close();
    }
}