import { Injectable, signal } from '@angular/core';
import { Subject, Subscription } from 'rxjs';
import { FrameLandmarks, LandmarkService } from './landmark';
import { HandLandmarkerResult } from '@mediapipe/tasks-vision';

// ─── Paramètres ───────────────────────────────────────────────────────────────

/** Vélocité minimale pour considérer qu'une main est en mouvement */
const VELOCITY_THRESHOLD = 0.015;

/** Rayon d'immobilité : distance max du poignet sur la fenêtre de stabilité */
const STILLNESS_RADIUS = 0.02;

/** Nombre de frames consécutives sous le rayon pour confirmer l'immobilité (~1s à 30fps) */
const STILLNESS_FRAMES = 30;

/** Nombre minimal de frames actives pour démarrer l'accumulation */
const MIN_ONSET_FRAMES = 5;

/** Taille minimale d'un segment valide */
const MIN_SEGMENT_FRAMES = 15;

/** Taille maximale d'un segment avant forçage */
const MAX_SEGMENT_FRAMES = 180;

/** Frames de contexte conservées après validation d'un mot */
const CONTEXT_FRAMES = 15;

/** Fréquence de glissement : une inférence toutes les K frames accumulées */
export const SLIDE_EVERY_K = 15;

export type SegmenterState = 'IDLE' | 'SIGNING';

export interface SignSegment {
  frames: Float32Array[];
  rawFrameCount: number;
  /** true = segment forcé par condition d'arrêt, false = segment glissant en cours de signe */
  isFinal: boolean;
}

interface WristPositions {
  lx: number; ly: number;
  rx: number; ry: number;
  valid: boolean;
}

@Injectable({ providedIn: 'root' })
export class SegmentationService {

  readonly state          = signal<SegmenterState>('IDLE');
  readonly framesCaptured = signal<number>(0);
  readonly segment$       = new Subject<SignSegment>();

  private accumulator: Float32Array[] = [];
  private onsetCounter  = 0;
  private prevWrist: WristPositions | null = null;

  private wristHistory: WristPositions[] = [];
  private framesSinceLastSlide = 0;

  private frameSub: Subscription | null = null;

  /** Dimensions de la vidéo — détermine si on est en portrait */
  private videoWidth  = 0;
  private videoHeight = 0;

  constructor(private landmarkService: LandmarkService) {}

  /** Appelé depuis camera-page après que la vidéo est prête */
  setVideoDimensions(width: number, height: number): void {
    this.videoWidth  = width;
    this.videoHeight = height;
  }

  private get isPortrait(): boolean {
    return this.videoHeight > this.videoWidth && this.videoWidth > 0;
  }

  /**
   * Transforme les coordonnées d'un landmark du repère portrait
   * vers le repère paysage attendu par le modèle.
   *
   * Rotation -90° (antihoraire) :
   *   x_landscape = y_portrait
   *   y_landscape = 1 - x_portrait
   *   z inchangé
   */
  private tolandscape(x: number, y: number, z: number): [number, number, number] {
    if (!this.isPortrait) return [x, y, z];
    return [y, 1 - x, z];
  }

  start(): void {
    this.reset();
    this.frameSub = this.landmarkService.frame$.subscribe(f => this.onFrame(f));
  }

  stop(): void {
    // Flush du buffer restant si assez de frames
    if (this.accumulator.length >= MIN_SEGMENT_FRAMES) {
      this.emitSegment(true);
    }
    this.frameSub?.unsubscribe();
    this.frameSub = null;
    this.reset();
  }

  private reset(): void {
    this.accumulator         = [];
    this.onsetCounter        = 0;
    this.prevWrist           = null;
    this.wristHistory        = [];
    this.framesSinceLastSlide = 0;
    this.state.set('IDLE');
    this.framesCaptured.set(0);
  }

  private onFrame(frame: FrameLandmarks): void {
    const features = this.extractFeatures(frame);
    const wrist    = this.getWristPositions(frame.handResult);
    const velocity = this.computeVelocity(wrist);
    const isActive = velocity > VELOCITY_THRESHOLD;
    const handsVisible = wrist.valid;

    // Met à jour l'historique des positions pour la détection d'immobilité
    if (wrist.valid) {
      this.wristHistory.push(wrist);
      if (this.wristHistory.length > STILLNESS_FRAMES) {
        this.wristHistory.shift();
      }
    }

    if (wrist.valid) this.prevWrist = wrist;

    switch (this.state()) {

      case 'IDLE': {
        if (isActive && handsVisible) {
          this.onsetCounter++;
          if (this.onsetCounter >= MIN_ONSET_FRAMES) {
            this.state.set('SIGNING');
            this.accumulator         = [];
            this.framesSinceLastSlide = 0;
            this.wristHistory        = [];
          }
        } else {
          this.onsetCounter = 0;
        }
        break;
      }

      case 'SIGNING': {
        this.accumulator.push(features);
        this.framesCaptured.set(this.accumulator.length);
        this.framesSinceLastSlide++;

        // ── Condition d'arrêt 1 : mains disparues ──────────────────────────
        if (!handsVisible) {
          this.emitSegment(true);
          break;
        }

        // ── Condition d'arrêt 2 : mains immobiles pendant STILLNESS_FRAMES ─
        if (this.isHandStill()) {
          this.emitSegment(true);
          break;
        }

        // ── Segment trop long : forçage ────────────────────────────────────
        if (this.accumulator.length >= MAX_SEGMENT_FRAMES) {
          this.emitSegment(true);
          break;
        }

        // ── Inférence glissante toutes les K frames ────────────────────────
        if (
          this.framesSinceLastSlide >= SLIDE_EVERY_K &&
          this.accumulator.length >= 87
        ) {
          this.emitSegment(false);
          this.framesSinceLastSlide = 0;
          // Garde les CONTEXT_FRAMES dernières pour le mot suivant
          // (le buffer n'est PAS vidé pour un segment glissant)
        }

        break;
      }
    }
  }

  // ── Détecte si la main est immobile sur la fenêtre STILLNESS_FRAMES ──────────
  private isHandStill(): boolean {
    if (this.wristHistory.length < STILLNESS_FRAMES) return false;

    const first = this.wristHistory[0];
    for (const w of this.wristHistory) {
      const dl = Math.sqrt((w.lx - first.lx) ** 2 + (w.ly - first.ly) ** 2);
      const dr = Math.sqrt((w.rx - first.rx) ** 2 + (w.ry - first.ry) ** 2);
      if (Math.max(dl, dr) > STILLNESS_RADIUS) return false;
    }
    return true;
  }

  // ── Émet un segment et gère le contexte de transition ────────────────────────
  private emitSegment(isFinal: boolean): void {
    const raw     = [...this.accumulator];
    const trimmed = isFinal ? this.trimInactiveFrames(raw) : raw;

    if (trimmed.length >= MIN_SEGMENT_FRAMES) {
      this.segment$.next({
        frames:        trimmed,
        rawFrameCount: raw.length,
        isFinal,
      });
    }

    if (isFinal) {
      // Garde les CONTEXT_FRAMES dernières comme contexte pour le prochain mot
      const context = this.accumulator.slice(-CONTEXT_FRAMES);
      this.accumulator         = context;
      this.framesSinceLastSlide = 0;
      this.wristHistory        = [];
      this.onsetCounter        = 0;
      this.state.set('IDLE');
      this.framesCaptured.set(0);
    }
    // Pour un segment glissant (isFinal=false), on ne vide pas le buffer
  }

  // ─── Trim des extrémités inactives ────────────────────────────────────────────
  private trimInactiveFrames(frames: Float32Array[]): Float32Array[] {
    if (frames.length === 0) return frames;
    const threshold = VELOCITY_THRESHOLD * 0.5;

    const velocities = frames.map((f, i) =>
      i === 0 ? 0 : this.featureVelocity(frames[i - 1], f)
    );

    let start = 0;
    for (let i = 0; i < velocities.length; i++) {
      if (velocities[i] > threshold) { start = Math.max(0, i - 1); break; }
    }
    let end = frames.length - 1;
    for (let i = velocities.length - 1; i >= 0; i--) {
      if (velocities[i] > threshold) { end = Math.min(frames.length - 1, i + 1); break; }
    }
    return frames.slice(start, end + 1);
  }

  // ─── Extraction features ──────────────────────────────────────────────────────
  private extractFeatures(frame: FrameLandmarks): Float32Array {
    const DIM_HAND  = 63, DIM_POSE = 132;
    const lh   = new Float32Array(DIM_HAND);
    const rh   = new Float32Array(DIM_HAND);
    const pose = new Float32Array(DIM_POSE);

    for (let i = 0; i < frame.handResult.landmarks.length; i++) {
      const lm   = frame.handResult.landmarks[i];
      const side = frame.handResult.handedness?.[i]?.[0]?.categoryName ?? '';
      const vec  = new Float32Array(DIM_HAND);
      for (let j = 0; j < 21; j++) {
        const [x, y, z] = this.tolandscape(lm[j].x, lm[j].y, lm[j].z);
        vec[j*3] = x; vec[j*3+1] = y; vec[j*3+2] = z;
      }
      if (side === 'Left') lh.set(vec); else if (side === 'Right') rh.set(vec);
    }

    if (frame.poseResult.landmarks.length > 0) {
      const lm = frame.poseResult.landmarks[0];
      for (let j = 0; j < 33; j++) {
        const [x, y, z] = this.tolandscape(lm[j].x, lm[j].y, lm[j].z);
        pose[j*4]=x; pose[j*4+1]=y;
        pose[j*4+2]=z; pose[j*4+3]=(lm[j] as any).visibility ?? 0;
      }
    }

    const out = new Float32Array(DIM_HAND * 2 + DIM_POSE);
    out.set(lh, 0); out.set(rh, DIM_HAND); out.set(pose, DIM_HAND * 2);
    return out;
  }

  private getWristPositions(handResult: HandLandmarkerResult): WristPositions {
    let lx = 0, ly = 0, rx = 0, ry = 0;
    let hasLeft = false, hasRight = false;
    for (let i = 0; i < handResult.landmarks.length; i++) {
      const side  = handResult.handedness?.[i]?.[0]?.categoryName ?? '';
      const wrist = handResult.landmarks[i][0];
      // On applique la même transformation que pour les features
      // pour que la vélocité soit cohérente avec le repère paysage
      const [wx, wy] = this.tolandscape(wrist.x, wrist.y, 0);
      if (side === 'Left')       { lx = wx; ly = wy; hasLeft  = true; }
      else if (side === 'Right') { rx = wx; ry = wy; hasRight = true; }
    }
    if (hasLeft && !hasRight)  { rx = lx; ry = ly; }
    if (hasRight && !hasLeft)  { lx = rx; ly = ry; }
    return { lx, ly, rx, ry, valid: hasLeft || hasRight };
  }

  private computeVelocity(current: WristPositions): number {
    if (!current.valid || !this.prevWrist) return 0;
    const dlx = current.lx - this.prevWrist.lx, dly = current.ly - this.prevWrist.ly;
    const drx = current.rx - this.prevWrist.rx, dry = current.ry - this.prevWrist.ry;
    return Math.max(
      Math.sqrt(dlx*dlx + dly*dly),
      Math.sqrt(drx*drx + dry*dry),
    );
  }

  private featureVelocity(a: Float32Array, b: Float32Array): number {
    const dlx = b[0]-a[0], dly = b[1]-a[1];
    const drx = b[63]-a[63], dry = b[64]-a[64];
    return Math.max(Math.sqrt(dlx*dlx+dly*dly), Math.sqrt(drx*drx+dry*dry));
  }
}