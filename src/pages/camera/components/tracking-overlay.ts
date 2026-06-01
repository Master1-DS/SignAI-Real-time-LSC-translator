import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-tracking-overlay',
  standalone: true,
  imports: [CommonModule],
  styles: [`
    .confidence-bar-fill {
      transition: width 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
  `],
  template: `
    <div class="flex items-center justify-center w-full h-full">
      <div class="relative w-[300px] h-[300px] rounded-3xl transition-all duration-300"
           >
           <!-- [class.border-2]="isActive"
           [class.border-primary/40]="isActive"
           [class.ai-pulse]="isActive"> -->

        <!-- Corner accents -->
        <!-- <div class="absolute -top-1 -left-1  w-8 h-8 border-t-4 border-l-4 border-primary rounded-tl-lg"></div>
        <div class="absolute -top-1 -right-1 w-8 h-8 border-t-4 border-r-4 border-primary rounded-tr-lg"></div>
        <div class="absolute -bottom-1 -left-1  w-8 h-8 border-b-4 border-l-4 border-primary rounded-bl-lg"></div>
        <div class="absolute -bottom-1 -right-1 w-8 h-8 border-b-4 border-r-4 border-primary rounded-br-lg"></div> -->

        <!-- Word bubble + confidence bar -->
        @if (word) {
          <div class="absolute -top-24 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 w-max">

            <!-- Mot détecté -->
            <div class="bg-surface/80 glass-panel border border-white/10
                        px-4 py-2 rounded-full flex items-center gap-2">
              <span class="font-headline-md text-headline-md text-primary">{{ word }}</span>
            </div>

            <!-- Barre de confiance -->
            <div class="w-40 flex flex-col gap-1">
              <!-- Piste -->
              <div class="w-full h-2 rounded-full bg-white/10 overflow-hidden">
                <!-- Remplissage coloré selon le niveau -->
                <div class="h-full rounded-full confidence-bar-fill"
                     [style.width.%]="confidencePct"
                     [class.bg-error]="confidencePct < 50"
                     [class.bg-tertiary]="confidencePct >= 50 && confidencePct < 80"
                     [class.bg-primary]="confidencePct >= 80">
                </div>
              </div>
              <!-- Label -->
              <div class="flex justify-between items-center">
                <span class="text-[10px] font-semibold uppercase tracking-widest text-on-surface-variant/70">
                  Confiance
                </span>
                <span class="text-[11px] font-bold"
                      [class.text-error]="confidencePct < 50"
                      [class.text-tertiary]="confidencePct >= 50 && confidencePct < 80"
                      [class.text-primary]="confidencePct >= 80">
                  {{ confidencePct }}%
                </span>
              </div>
            </div>

          </div>
        }
      </div>
    </div>
  `,
})
export class TrackingOverlayComponent {
  @Input() word = '';
  @Input() confidence = 80;
  @Input() isActive = false;

  get confidencePct(): number {
    return Math.round(this.confidence * 100);
  }
}