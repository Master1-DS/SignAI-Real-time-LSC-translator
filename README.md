# Signai

Application de reconnaissance et segmentation de gestes en temps réel, alimentée par des modèles de machine learning (ONNX) et la vision par ordinateur avec MediaPipe.

**Stack:** Angular 21.2.13 · Vite · TypeScript · MediaPipe · ONNX Runtime · Capacitor

---

## ⚠️ Prérequis Obligatoires

Avant de commencer, assurez-vous que les ressources suivantes sont présentes :

### Modèle ONNX Entraîné (CRITIQUE)

Placez le fichier du modèle pré-entraîné dans le dossier approprié :

```
public/models/best_model.onnx
```

Ce fichier contient le modèle ONNX pré-entraîné utilisé pour la reconnaissance et la classification des gestes. **Sans ce fichier, l'application ne fonctionnera pas correctement.**

### Autres Ressources (Incluses)

Les ressources suivantes sont déjà incluses :

- `public/models/hand_landmarker.task` - Modèle MediaPipe pour la détection des points de repère des mains
- `public/models/pose_landmarker_lite.task` - Modèle MediaPipe pour la détection de pose
- Assets MediaPipe WASM dans `public/mediapipe-wasm/`
- Runtime ONNX WASM dans `public/ort-wasm/`

---

## Installation

```bash
npm install
```

Les scripts de copie (dans `scripts/`) copieront automatiquement les assets MediaPipe et ONNX WASM vers le dossier public lors de l'installation.

---

## Serveur de Développement

Démarrez le serveur de développement local :

```bash
npm start
```

ou

```bash
ng serve
```

Ouvrez votre navigateur et accédez à `http://localhost:4200/`. L'application se rechargera automatiquement lors de modifications des fichiers source.

---

## Structure du Projet

```
src/
├── app/
│   ├── services/              # Services métier
│   │   ├── camera.ts         # Capture vidéo depuis la caméra
│   │   ├── landmark.ts       # Détection des landmarks (MediaPipe)
│   │   ├── segmentation.ts   # Segmentation avec modèle ONNX
│   │   └── translation.ts    # Service de traduction
│   ├── pages/
│   │   ├── camera/           # Page capture en direct
│   │   └── upload/           # Page d'upload d'images
│   ├── models/               # Modèles de données et labels
│   └── app.ts                # Composant racine
├── components/
│   └── liquid-glass-nav.ts   # Navigation
└── styles.css                # Styles globaux
```

---

## Génération de Composants

Pour générer un nouveau composant :

```bash
ng generate component component-name
```

Pour d'autres schematics :

```bash
ng generate --help
```

---

## Tests

### Tests Unitaires

Exécutez les tests unitaires avec [Vitest](https://vitest.dev/) :

```bash
npm test
```

ou

```bash
ng test
```

### Tests End-to-End

```bash
ng e2e
```

---

## Build pour Production

Compilez le projet pour la production :

```bash
npm run build
```

ou

```bash
ng build
```

Les artefacts de build seront stockés dans le dossier `dist/`. Le build est optimisé pour la performance et la vitesse.

---

## Build Mobile (Android)

Le projet utilise [Capacitor](https://capacitorjs.com/) pour les builds mobiles :

```bash
npm run build
npx cap add android
npx cap sync android
npx cap open android
```

Le dossier `android/` contient la configuration Android.

---

## Architecture

- **MediaPipe** : Détection des points de repère des mains et pose en temps réel
- **ONNX Runtime** : Exécution du modèle ML (best_model.onnx) côté client
- **Angular** : Framework UI avec standalone components
- **Vite** : Bundler rapide
- **Capacitor** : Wrapper natif pour iOS/Android

---

## Ressources Utiles

- [Documentation Angular](https://angular.dev)
- [Documentation MediaPipe](https://developers.google.com/mediapipe)
- [Documentation ONNX Runtime](https://onnxruntime.ai/)
- [Documentation Capacitor](https://capacitorjs.com/docs)
- [Vite Guide](https://vitejs.dev/guide/)
