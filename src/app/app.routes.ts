import { Routes } from '@angular/router';

export const routes: Routes = [
    { path: '', redirectTo: 'camera', pathMatch: 'full' },
    // {
    //     path: 'home',
    //     loadComponent: () =>
    //     import('./features/home/pages/home-page.component').then(m => m.HomePageComponent),
    // },
    {
        path: 'camera',
        loadComponent: () =>
        import('../pages/camera/camera-page').then(m => m.CameraPageComponent),
    },
    // {
    //     path: 'history',
    //     loadComponent: () =>
    //     import('./features/history/pages/history-page.component').then(m => m.HistoryPageComponent),
    // },
    { path: '**', redirectTo: 'camera' },
];
