import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { LiquidGlassNavComponent } from '../components/liquid-glass-nav.component';
import { StatusBarComponent } from '../pages/camera/components/status-bar';
import { TranslationService } from './services/translation';
import { SegmentationService } from './services/segmentation';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    RouterOutlet,
    CommonModule,
    LiquidGlassNavComponent,
    StatusBarComponent],
  styles: [`
    :host {
      display: block;
      min-height: 100dvh;
    }
    .page-wrapper {
      /* pt-16 = header fixe (64px) */
      /* pb = navbar (72px) + safe-area + 12px de marge */
      padding-top: 64px;
      padding-bottom: calc(72px + env(safe-area-inset-bottom, 0px) + 12px);
      min-height: 100dvh;
      box-sizing: border-box;
    }
  `],
  template: `
    <!-- Top App Bar -->
    <header class="fixed top-0 w-full z-50 flex justify-between items-center px-2 py-4
                   bg-surface/80 backdrop-blur-xl border-b border-white/10 gap-4">
      <h1 class="font-bold text-2xl text-primary-fixed-dim tracking-tight">Signai</h1>
      <!-- <button class="hover:opacity-80 active:scale-95 transition-all duration-200">
        <span class="material-symbols-outlined text-primary-fixed-dim">settings</span>
      </button> -->
      <app-status-bar
        class="w-[calc(100%-40px)] max-w-xl z-20"
        [status]="translationService.streamStatus()"
        [segmentationState]="segmentationService.state"
        [translationService]="{
          isTranslating: translationService.isTranslating,
          segmentationService: { framesCaptured: translationService['segmentationService'].framesCaptured }
        }"
      />
    </header>

    <!-- Routed page content -->
    <div class="pt-16 pb-28 max-h-[100dvh] overflow-scroll scrollbar-thin scrollbar-thumb-surface/50 scrollbar-track-transparent">
      <router-outlet />
    </div>

    <!-- Liquid Glass Navbar -->
    <app-liquid-glass-nav />
  `,
})
export class App {
  constructor(
    public translationService: TranslationService,
    public segmentationService: SegmentationService
  ) {}
}
