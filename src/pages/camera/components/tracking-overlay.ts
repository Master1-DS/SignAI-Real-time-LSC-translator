import { Component, Input, OnChanges, SimpleChanges, signal } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-tracking-overlay',
  standalone: true,
  imports: [CommonModule],
  styles: [`
    .confidence-bar-fill {
      transition: width 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    .word-bubble {
      transition: opacity 0.4s ease, transform 0.4s ease;
    }
    .word-bubble.visible {
      opacity: 1;
      transform: translateY(0);
    }
    .word-bubble.hidden {
      opacity: 0;
      transform: translateY(-8px);
      pointer-events: none;
    }
  `],
  template: `
    <div class="flex items-center justify-center w-full h-full">
      <div class="relative w-[300px] h-[300px] rounded-3xl transition-all duration-300">

        @if (displayedWord()) {
          <div class="word-bubble absolute -top-24 left-1/2 -translate-x-1/2
                      flex flex-col items-center gap-2 w-max"
               [class.visible]="visible()"
               [class.hidden]="!visible()">

            <!-- Mot détecté -->
            <div class="bg-surface/80 glass-panel border border-white/10
                        px-4 py-2 rounded-full flex items-center gap-2">
              <span class="font-headline-md text-headline-md text-primary">
                {{ displayedWord() }}
              </span>
            </div>

            <!-- Barre de confiance -->
            <div class="w-40 flex flex-col gap-1">
              <div class="w-full h-2 rounded-full bg-white/10 overflow-hidden">
                <div class="h-full rounded-full confidence-bar-fill"
                     [style.width.%]="displayedConfidencePct()"
                     [class.bg-error]="displayedConfidencePct() < 50"
                     [class.bg-tertiary]="displayedConfidencePct() >= 50 && displayedConfidencePct() < 80"
                     [class.bg-primary]="displayedConfidencePct() >= 80">
                </div>
              </div>
              <div class="flex justify-between items-center">
                <span class="text-[10px] font-semibold uppercase tracking-widest
                             text-on-surface-variant/70">
                  Confiance
                </span>
                <span class="text-[11px] font-bold"
                      [class.text-error]="displayedConfidencePct() < 50"
                      [class.text-tertiary]="displayedConfidencePct() >= 50 && displayedConfidencePct() < 80"
                      [class.text-primary]="displayedConfidencePct() >= 80">
                  {{ displayedConfidencePct() }}%
                </span>
              </div>
            </div>

          </div>
        }
      </div>
    </div>
  `,
})
export class TrackingOverlayComponent implements OnChanges {
  @Input() word       = '';
  @Input() confidence = 0;
  @Input() isActive   = false;

  readonly visible              = signal(false);
  readonly displayedWord        = signal('');
  readonly displayedConfidencePct = signal(0);

  private hideTimeout: ReturnType<typeof setTimeout> | null = null;

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['word'] && this.word && this.word !== changes['word'].previousValue) {
      this.showWord();
    }
  }

  private showWord(): void {
    if (this.hideTimeout) {
      clearTimeout(this.hideTimeout);
      this.hideTimeout = null;
    }

    this.displayedWord.set(this.word);
    this.displayedConfidencePct.set(Math.round(this.confidence * 100));
    this.visible.set(true);

    this.hideTimeout = setTimeout(() => {
      this.visible.set(false);
      // Efface le mot après la transition CSS (400ms)
      setTimeout(() => this.displayedWord.set(''), 400);
      this.hideTimeout = null;
    }, 10_000);
  }
}