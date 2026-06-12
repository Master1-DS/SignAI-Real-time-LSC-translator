import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { StreamStatus } from '../../../app/models/translation';

@Component({
  selector: 'app-status-bar',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="bg-surface-container-low/60 glass-panel border border-white/5 rounded-2xl p-2 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-3 h-3 rounded-full ai-pulse"
             [class.bg-primary]="status.isStreaming"
             [class.bg-outline]="!status.isStreaming">
        </div>
        <span class="font-label-md text-label-md text-on-surface-variant">
          {{ status.isStreaming ? 'Active' : 'En pause' }}
        </span>
      </div>
      <div class="flex items-center gap-3">
        @if (translationService.isTranslating()) {
          <span class="text-label-sm px-2 py-0.5"
                [class.text-primary]="segmentationState() === 'SIGNING'"
                [class.text-outline]="segmentationState() === 'IDLE'">
            <!-- {{ segmentationState() }} -->
            @if (segmentationState() === 'SIGNING') {
              · {{ translationService['segmentationService'].framesCaptured() }}f
            }
          </span>
        }
      </div>
      <div class="flex items-center gap-2">
        <span class="font-label-sm text-label-sm text-outline">FPS: {{ status.fps }}</span>
        <span class="w-1 h-1 rounded-full bg-outline"></span>
        <span class="font-label-sm text-label-sm text-outline">LATENCE: {{ status.latencyMs }}ms</span>
      </div>
    </div>
  `,
})
export class StatusBarComponent {
  @Input() status!: StreamStatus;
  @Input() segmentationState!: () => 'IDLE' | 'SIGNING';
  @Input() translationService!: { isTranslating: () => boolean; segmentationService: { framesCaptured: () => number } };
}
