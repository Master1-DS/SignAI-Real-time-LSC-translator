import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-camera-controls',
  standalone: true,
  imports: [CommonModule],
  styles: [`
    /* Icônes remplies par défaut via CSS */
    .icon-filled {
      font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    }
    .icon-outline {
      font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    }
  `],
  template: `
    <div class="flex items-center justify-between gap-4">

      <!-- Flashlight -->
      <button
        (click)="flashToggled.emit()"
        [disabled]="!hasFlashlight"
        class="w-14 h-14 rounded-full bg-surface-container-high/80 glass-panel border border-white/10
               flex items-center justify-center text-on-surface
               hover:bg-surface-container-highest transition-colors active:scale-90 duration-150
               disabled:opacity-30 disabled:cursor-not-allowed">
        <span class="material-symbols-outlined"
              [class.icon-filled]="flashlightOn"
              [class.icon-outline]="!flashlightOn">
          flashlight_on
        </span>
      </button>

      <!-- Center: Mode label + Play/Pause -->
      <div class="flex-1 flex items-center justify-center
                  bg-surface-container-high/80 glass-panel border border-white/10
                  rounded-full p-1.5 px-4 gap-4">
        <div class="flex items-center gap-2">
          <span class="font-label-md text-label-md text-on-surface-variant">LSF</span>
          <span class="material-symbols-outlined icon-outline" style="font-size:16px; color:#8c909f">
            arrow_forward
          </span>
          <span class="font-label-md text-label-md text-primary font-bold">TEXTE</span>
        </div>
        <div class="h-6 w-[1px] bg-white/10"></div>
        <button
          (click)="playPauseToggled.emit()"
          class="w-12 h-12 rounded-full flex items-center justify-center
                 hover:opacity-90 transition-all active:scale-90 duration-150 shadow-lg"
          [class.bg-primary]="isPlaying"
          [class.bg-error-container]="!isPlaying">
          <span class="material-symbols-outlined icon-filled"
                [class.text-on-primary]="isPlaying"
                [class.text-on-error-container]="!isPlaying">
            {{ isPlaying ? 'pause' : 'play_arrow' }}
          </span>
        </button>
      </div>

      <!-- Flip Camera -->
      <button
        (click)="flipRequested.emit()"
        class="w-14 h-14 rounded-full bg-surface-container-high/80 glass-panel border border-white/10
               flex items-center justify-center text-on-surface
               hover:bg-surface-container-highest transition-colors active:scale-90 duration-150">
        <span class="material-symbols-outlined icon-outline">flip_camera_android</span>
      </button>

    </div>
  `,
})
export class CameraControlsComponent {
  @Input() isPlaying = true;
  @Input() flashlightOn = false;
  @Input() hasFlashlight = false;

  @Output() playPauseToggled = new EventEmitter<void>();
  @Output() flipRequested    = new EventEmitter<void>();
  @Output() flashToggled     = new EventEmitter<void>();
}