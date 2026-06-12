import {
  Component, ElementRef, OnDestroy, ViewChild, signal, computed,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { CameraService } from '../../app/services/camera';
import { TranslationService } from '../../app/services/translation';
import { LandmarkService } from '../../app/services/landmark';
import { SegmentationService } from '../../app/services/segmentation';
import { TrackingOverlayComponent } from './components/tracking-overlay';
import { TranslationPanelComponent } from './components/translation-panel';
import { CameraControlsComponent } from './components/camera-controls';

@Component({
  selector: 'app-camera-page',
  standalone: true,
  imports: [
    CommonModule,
    TrackingOverlayComponent,
    CameraControlsComponent,
    TranslationPanelComponent,
  ],
  templateUrl: './camera-page.html',
})
export class CameraPageComponent implements OnDestroy {
  @ViewChild('videoEl',  { static: true }) videoRef!:  ElementRef<HTMLVideoElement>;
  @ViewChild('canvasEl', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;

  readonly isPlaying   = signal(false);
  readonly isLoadingML = signal(false);

  // Exposé au template pour la status bar
  readonly segmentationState = computed(() =>
    this.segmentationService.state()
  );

  constructor(
    public cameraService:        CameraService,
    public translationService:   TranslationService,
    public landmarkService:      LandmarkService,
    private segmentationService: SegmentationService,
  ) {}

  async startCamera(): Promise<void> {
    try {
      const stream = await this.cameraService.startCamera('environment');
      const video  = this.videoRef.nativeElement;
      video.srcObject = stream;
      await video.play();

      this.isLoadingML.set(true);
      this.translationService.setVideoElement(video);
      await this.landmarkService.init();
      this.isLoadingML.set(false);

      // Communique les dimensions réelles de la vidéo au service de segmentation
      // pour que la transformation portrait→paysage soit correctement appliquée
      this.segmentationService.setVideoDimensions(
        video.videoWidth,
        video.videoHeight,
      );

      this.landmarkService.startDetection(video, this.canvasRef.nativeElement);
      this.translationService.startTranslation();
      this.isPlaying.set(true);
    } catch (err) {
      this.isLoadingML.set(false);
      console.error('Erreur démarrage caméra / MediaPipe', err);
    }
  }

  async ngOnDestroy(): Promise<void> {
    this.landmarkService.stopDetection();
    this.translationService.stopTranslation();
    await this.cameraService.stopCamera();
  }

  async togglePlayPause(): Promise<void> {
    const playing = !this.isPlaying();
    this.isPlaying.set(playing);
    if (playing) {
      this.landmarkService.startDetection(
        this.videoRef.nativeElement,
        this.canvasRef.nativeElement
      );
      this.translationService.startTranslation();
    } else {
      this.landmarkService.stopDetection();
      this.translationService.stopTranslation();
    }
  }

  async onFlipCamera(): Promise<void> {
    const stream = await this.cameraService.flipCamera();
    const video  = this.videoRef.nativeElement;
    video.srcObject = stream;
    await video.play();
    // Met à jour les dimensions après le flip (la caméra frontale peut avoir
    // une résolution différente de la caméra arrière)
    this.segmentationService.setVideoDimensions(
      video.videoWidth,
      video.videoHeight,
    );
    this.landmarkService.startDetection(video, this.canvasRef.nativeElement);
  }

  async onToggleFlash(): Promise<void> {
    await this.cameraService.toggleFlashlight();
  }
}