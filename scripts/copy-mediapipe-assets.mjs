import { cpSync, mkdirSync, existsSync } from 'fs';
import { join, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const ROOT = resolve(__dirname, '..');

const SRC_WASM  = join(ROOT, 'node_modules/@mediapipe/tasks-vision/wasm');
const DEST_WASM = join(ROOT, 'public/mediapipe-wasm');

if (!existsSync(SRC_WASM)) {
  console.error('  @mediapipe/tasks-vision introuvable dans node_modules.');
  console.error('    Lance : npm install @mediapipe/tasks-vision');
  process.exit(1);
}

mkdirSync(DEST_WASM, { recursive: true });
mkdirSync(join(ROOT, 'public/models'), { recursive: true });

cpSync(SRC_WASM, DEST_WASM, { recursive: true });
console.log('  Fichiers WASM copiés  ->  public/mediapipe-wasm/');
console.log('');
console.log('  Télécharge maintenant les modèles avec :');
console.log('');
console.log('    # Hand Landmarker');
console.log('    curl -L -o public/models/hand_landmarker.task \\');
console.log('      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task');
console.log('');
console.log('    # Pose Landmarker Lite');
console.log('    curl -L -o public/models/pose_landmarker_lite.task \\');
console.log('      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task');
