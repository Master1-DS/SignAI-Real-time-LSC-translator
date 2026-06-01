import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, RouterLinkActive } from '@angular/router';

export interface NavItem {
    label: string;
    icon: string;
    route: string;
}

@Component({
    selector: 'app-liquid-glass-nav',
    standalone: true,
    imports: [CommonModule, RouterModule, RouterLinkActive],
    styles: [`
        .liquid-glass-bar {
            position: fixed;
            bottom: 7px;
            left: 50%;
            transform: translateX(-50%);
            width: min(100%, 540px);
            margin: 0 auto;
            background: rgba(18, 18, 24, 0.55);
            backdrop-filter: saturate(180%) blur(28px);
            -webkit-backdrop-filter: saturate(180%) blur(28px);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 28px;
            box-shadow:
                0 20px 60px rgba(0,0,0,0.2),
                0 -1px 0 0 rgba(255,255,255,0.06) inset;
        }
        .nav-pill {
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 3px;
            padding: 7px 12px;
            border-radius: 999px;
            transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
            cursor: pointer;
            -webkit-tap-highlight-color: transparent;
            text-decoration: none;
        }
        .nav-pill.active {
            background: rgba(173, 198, 255, 0.14);
            box-shadow:
                0 0 0 1px rgba(173, 198, 255, 0.2),
                0 2px 12px rgba(77, 142, 255, 0.25),
                inset 0 1px 0 rgba(255,255,255,0.12);
        }
        .nav-pill.active::before {
        content: '';
            position: absolute;
            top: 0; left: 20%; right: 20%;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent);
            border-radius: 999px;
        }
        .nav-pill:active { transform: scale(0.88); }
        .nav-icon {
            font-size: 24px;
            transition: color 0.2s ease, filter 0.2s ease;
            font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        }
        .nav-pill.active .nav-icon {
            color: #adc6ff;
            filter: drop-shadow(0 0 6px rgba(77,142,255,0.7));
        }
        .nav-label {
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.01em;
            transition: color 0.2s ease;
            color: #c2c6d6;
        }
        .nav-pill.active .nav-label { color: #adc6ff; }
        .safe-area-pb {
            padding-bottom: max(12px, env(safe-area-inset-bottom));
        }
    `],
    template: `
        <nav class="liquid-glass-bar z-50 safe-area-pb">
            <div class="flex justify-around items-center px-2 pt-3">
                @for (item of navItems; track item.route) {
                    <a
                        [routerLink]="item.route"
                        routerLinkActive="active"
                        class="nav-pill">
                        <span class="material-symbols-outlined nav-icon">{{ item.icon }}</span>
                        <span class="nav-label">{{ item.label }}</span>
                    </a>
                }
            </div>
        </nav>
    `,
})
export class LiquidGlassNavComponent {
    @Input() navItems: NavItem[] = [
        { label: 'Settings',    icon: 'settings',     route: '/setting' },
        { label: 'Camera',     icon: 'videocam', route: '/camera' },
        { label: 'History', icon: 'history',  route: '/history' },
    ];
}