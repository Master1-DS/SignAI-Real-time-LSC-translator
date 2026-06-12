import { Injectable, signal } from '@angular/core';
import {
  HandLandmarker,
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
  HandLandmarkerResult,
  PoseLandmarkerResult,
} from '@mediapipe/tasks-vision';
import { Subject } from 'rxjs';

export interface FrameLandmarks {
  handResult: HandLandmarkerResult;
  poseResult: PoseLandmarkerResult;
}

@Injectable({ providedIn: 'root' })
export class LandmarkService {
  private handLandmarker: HandLandmarker | null = null;
  private poseLandmarker: PoseLandmarker | null = null;
  private drawingUtils: DrawingUtils | null = null;
  private animFrameId: number | null = null;
  private lastVideoTime = -1;

  readonly isReady       = signal(false);
  readonly handsDetected = signal(0);

  readonly frame$ = new Subject<FrameLandmarks>();

  async init(): Promise<void> {
    if (this.isReady()) return;

    const vision = await FilesetResolver.forVisionTasks('/mediapipe-wasm');

    [this.handLandmarker, this.poseLandmarker] = await Promise.all([
      HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: '/models/hand_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.4,
        minHandPresenceConfidence: 0.4,
        minTrackingConfidence: 0.4,
      }),
      PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: '/models/pose_landmarker_lite.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.4,
        minPosePresenceConfidence: 0.4,
        minTrackingConfidence: 0.4,
      }),
    ]);

    this.isReady.set(true);
  }

  startDetection(video: HTMLVideoElement, canvas: HTMLCanvasElement): void {
    if (!this.isReady()) return;

    const ctx = canvas.getContext('2d')!;
    this.drawingUtils = new DrawingUtils(ctx);

    const loop = () => {
      this.animFrameId = requestAnimationFrame(loop);

      // Synchronise les dimensions canvas <-> vidéo
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      if (video.currentTime === this.lastVideoTime) return;
      this.lastVideoTime = video.currentTime;

      const ts = performance.now();
      const handResult = this.handLandmarker!.detectForVideo(video, ts);
      const poseResult = this.poseLandmarker!.detectForVideo(video, ts);

      // Dessin sur le canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      this.drawHands(handResult);
      this.drawPose(poseResult);
      this.handsDetected.set(handResult.landmarks.length);

      // Émission des landmarks bruts -> TranslationService
      this.frame$.next({ handResult, poseResult });
    };

    loop();
  }

  stopDetection(): void {
    if (this.animFrameId !== null) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
  }

  private drawHands(result: HandLandmarkerResult): void {
    if (!this.drawingUtils) return;
    for (const lm of result.landmarks) {
      this.drawingUtils.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS,
        { color: 'rgba(173,198,255,0.7)', lineWidth: 2 });
      this.drawingUtils.drawLandmarks(lm,
        { color: '#f70303', fillColor: 'rgb(255, 7, 7)', lineWidth: 1, radius: 4 });
    }
  }

  private drawPose(result: PoseLandmarkerResult): void {
    if (!this.drawingUtils) return;
    for (const lm of result.landmarks) {
      this.drawingUtils.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS,
        { color: 'rgba(192,193,255,0.5)', lineWidth: 2 });
      this.drawingUtils.drawLandmarks(lm,
        { color: '#0409ff', fillColor: 'rgba(5, 9, 255, 0.8)', lineWidth: 1, radius: 3 });
    }
  }

  async destroy(): Promise<void> {
    this.stopDetection();
    this.handLandmarker?.close();
    this.poseLandmarker?.close();
    this.handLandmarker = null;
    this.poseLandmarker = null;
    this.isReady.set(false);
  }
}