import { Component, Input, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-translation-panel',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="bg-surface/40 glass-panel border border-white/10 rounded-3xl p-4 shadow-2xl">
      <div class="flex flex-col gap-2">
        <span class="text-primary font-label-sm text-label-sm uppercase tracking-widest opacity-70">
          Traduction en direct
        </span>
        <p class="font-headline-md text-headline-md text-on-surface leading-snug transition-opacity duration-300"
           [class.opacity-0]="fading"
           [class.opacity-100]="!fading">
          {{ displayedSentence || '...' }}
        </p>
      </div>
    </div>
  `,
})
export class TranslationPanelComponent implements OnChanges {
  @Input() sentence = '';

  displayedSentence = '';
  fading = false;

  ngOnChanges(): void {
    if (this.sentence === this.displayedSentence) return;
    this.fading = true;
    setTimeout(() => {
      this.displayedSentence = this.sentence;
      this.fading = false;
    }, 300);
  }
}
