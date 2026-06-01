import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.signai.app',
  appName: 'SignAI',
  webDir: 'dist/signai/browser',
  server: {
    androidScheme: 'https',
  },
  android: {
    allowMixedContent: true,
  },
};

export default config;
