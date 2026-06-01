import { Injectable, signal } from '@angular/core';

export type CameraFacing = 'user' | 'environment';
export type FlashlightState = 'on' | 'off';

@Injectable({ providedIn: 'root' })
export class CameraService {
  private stream: MediaStream | null = null;
  private videoTrack: MediaStreamTrack | null = null;

  readonly isCameraActive = signal(false);
  readonly facing = signal<CameraFacing>('environment');
  readonly flashlightOn = signal(false);
  readonly hasFlashlight = signal(false);

  async startCamera(facingMode: CameraFacing = 'environment'): Promise<MediaStream> {
    await this.stopCamera();

    const constraints: MediaStreamConstraints = {
      video: {
        facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 60 },
      },
      audio: false,
    };

    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.videoTrack = this.stream.getVideoTracks()[0];

    await this.detectFlashlight();
    this.isCameraActive.set(true);
    this.facing.set(facingMode);

    return this.stream;
  }

  async stopCamera(): Promise<void> {
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
      this.videoTrack = null;
    }
    this.isCameraActive.set(false);
    this.flashlightOn.set(false);
  }

  async flipCamera(): Promise<MediaStream> {
    const newFacing: CameraFacing = this.facing() === 'environment' ? 'user' : 'environment';
    return this.startCamera(newFacing);
  }

  async toggleFlashlight(): Promise<void> {
    if (!this.videoTrack || !this.hasFlashlight()) return;
    const newState = !this.flashlightOn();
    try {
      await (this.videoTrack as any).applyConstraints({
        advanced: [{ torch: newState }],
      });
      this.flashlightOn.set(newState);
    } catch {
      console.warn('Torch not supported on this device.');
    }
  }

  getStream(): MediaStream | null {
    return this.stream;
  }

  private async detectFlashlight(): Promise<void> {
    if (!this.videoTrack) return;
    try {
      const capabilities = (this.videoTrack as any).getCapabilities?.();
      this.hasFlashlight.set(!!capabilities?.torch);
    } catch {
      this.hasFlashlight.set(false);
    }
  }
}