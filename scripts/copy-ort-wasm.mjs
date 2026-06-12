import { cpSync, mkdirSync, readdirSync, existsSync } from 'fs';
import { join, resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const ROOT = resolve(__dirname, '..');

const candidates = [
  join(ROOT, 'node_modules/onnxruntime-web/dist'),
  join(ROOT, 'node_modules/onnxruntime-web'),
];

const SRC = candidates.find(p => existsSync(p) &&
  readdirSync(p).some(f => f.endsWith('.wasm')));

if (!SRC) {
  console.error('  Fichiers WASM onnxruntime-web introuvables.');
  console.error('    Lance : npm install onnxruntime-web');
  process.exit(1);
}

const DEST = join(ROOT, 'public/ort-wasm');
mkdirSync(DEST, { recursive: true });

const files = readdirSync(SRC).filter(f =>
  f.endsWith('.wasm') || f.match(/ort-wasm.*\.(js|mjs)$/)
);

for (const f of files) {
  cpSync(join(SRC, f), join(DEST, f));
}

console.log(`  ${files.length} fichiers ORT copiés → public/ort-wasm/`);
files.forEach(f => console.log(`    ${f}`));
