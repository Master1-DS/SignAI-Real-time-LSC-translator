import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { LiquidGlassNavComponent } from '../components/liquid-glass-nav.component';
import { StatusBarComponent } from '../pages/camera/components/status-bar';
import { TranslationService } from './services/translation';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    RouterOutlet,
    CommonModule,
    LiquidGlassNavComponent,
    StatusBarComponent],
  template: `
    <!-- Top App Bar -->
    <header class="fixed top-0 w-full z-50 flex justify-between items-center gap-5 px-3 py-4
                   bg-surface/80 backdrop-blur-xl border-b border-white/10">
      <h1 class="font-bold text-2xl text-primary-fixed-dim tracking-tight">Signai</h1>
      <!-- <button class="hover:opacity-80 active:scale-95 transition-all duration-200">
        <span class="material-symbols-outlined text-primary-fixed-dim">settings</span>
      </button> -->
      <app-status-bar
        class="w-[calc(100%-40px)] max-w-xl z-20"
        [status]="translationService.streamStatus()"
      />
    </header>

    <!-- Routed page content -->
    <div class="pt-16 pb-28">
      <router-outlet />
    </div>

    <!-- Liquid Glass Navbar -->
    <app-liquid-glass-nav />
  `,
  styleUrl: './app.css'
})
export class App {
  constructor(public translationService: TranslationService) {}
}
