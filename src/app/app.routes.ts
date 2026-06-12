import { Routes } from '@angular/router';

export const routes: Routes = [
    { path: '', redirectTo: 'camera', pathMatch: 'full' },
    {
        path: 'upload',
        loadComponent: () =>
        import('../pages/upload/upload-page').then(m => m.UploadPageComponent),
    },
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
