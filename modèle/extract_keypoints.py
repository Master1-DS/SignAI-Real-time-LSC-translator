#!/usr/bin/env python3
"""
Script 1 — extract_keypoints.py
Extraction des keypoints MediaPipe (Tasks API) pour toutes les vidéos CASL-W60.

Compatible mediapipe >= 0.10.20
Entrée  : CASL/train|val|test/*.mp4  +  glossary.csv
Sortie  : features/train|val|test/*.npy  +  labels_*.json  +  metadata.json
"""

import re
import csv
import cv2
import json
import logging
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from pathlib import Path
from tqdm import tqdm
import sys

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
CASL_DIR          = Path("CASL")
FEATURES_DIR      = Path("features")
GLOSSARY_CSV      = Path("glossary.csv")
ERROR_LOG         = "extraction_errors.log"
HAND_MODEL_PATH   = "hand_landmarker.task"
POSE_MODEL_PATH   = "pose_landmarker.task"
SPLITS            = ["train", "val", "test"]

# Dimensions des vecteurs
DIM_HAND  = 21 * 3    # 63  (x, y, z) par main
DIM_POSE  = 33 * 4    # 132 (x, y, z, visibility) pour le corps
DIM_TOTAL = DIM_HAND * 2 + DIM_POSE  # 258
# ─────────────────────────────────────────────────────────────────────────────


def check_inputs():
    """Vérifie que tous les fichiers d'entrée nécessaires existent."""
    ok = True

    if not GLOSSARY_CSV.exists():
        print(f"❌ Fichier manquant : {GLOSSARY_CSV}")
        print("   Lance d'abord : python build_glossary.py")
        ok = False

    if not Path(HAND_MODEL_PATH).exists():
        print(f"❌ Modèle manquant : {HAND_MODEL_PATH}")
        print("   Télécharge-le avec : python download_models.py")
        ok = False

    if not Path(POSE_MODEL_PATH).exists():
        print(f"❌ Modèle manquant : {POSE_MODEL_PATH}")
        print("   Télécharge-le avec : python download_models.py")
        ok = False

    for split in SPLITS:
        split_dir = CASL_DIR / split
        if not split_dir.exists():
            print(f"❌ Dossier manquant : {split_dir}")
            ok = False

    if not ok:
        sys.exit(1)

    print("✓ Tous les fichiers d'entrée sont présents")


def download_models():
    """
    Télécharge les modèles MediaPipe Tasks si absents.
    Appelé automatiquement si les fichiers .task sont manquants.
    """
    import urllib.request

    models = {
        HAND_MODEL_PATH: (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        ),
        POSE_MODEL_PATH: (
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/1/"
            "pose_landmarker_lite.task"
        ),
    }

    for filename, url in models.items():
        if not Path(filename).exists():
            print(f"📥 Téléchargement de {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"   ✓ {filename} téléchargé")
            except Exception as e:
                print(f"   ❌ Échec du téléchargement : {e}")
                sys.exit(1)
        else:
            print(f"   ✓ {filename} déjà présent")


def load_glossary(csv_path: Path) -> dict:
    """
    Charge glossary.csv → {geste_id: label}.
    Ex : {"G001": "BONJOUR", "G002": "MERCI"}
    """
    glossary = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            glossary[row["geste_id"].strip()] = row["label"].strip()

    unknowns = [k for k, v in glossary.items() if v.startswith("INCONNU")]
    if unknowns:
        print(f"⚠️  {len(unknowns)} label(s) INCONNU dans glossary.csv")
        sys.exit(1)

    print(f"✓ glossary.csv chargé : {len(glossary)} signes")
    return glossary


def parse_filename(filename: str) -> tuple:
    """
    Parse un nom de fichier CASL-W60.
    "G001S02T01.mp4" → ("G001", "S02", "T01")
    Retourne (None, None, None) si format invalide.
    """
    pattern = re.compile(r'^(G\d{3})(S\d{2,3})(T\d{2,3})', re.IGNORECASE)
    stem    = Path(filename).stem
    match   = pattern.match(stem)
    if match:
        return (
            match.group(1).upper(),
            match.group(2).upper(),
            match.group(3).upper(),
        )
    return None, None, None


def build_hand_options() -> HandLandmarkerOptions:
    """
    Construit les options du HandLandmarker (Tasks API).

    - num_hands=2       : détecter les deux mains simultanément
    - running_mode=IMAGE: traitement frame par frame (pas de vidéo en streaming)
    """
    base_options = mp_python.BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    )
    return HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def build_pose_options() -> PoseLandmarkerOptions:
    """
    Construit les options du PoseLandmarker (Tasks API).

    - running_mode=IMAGE : traitement frame par frame
    """
    base_options = mp_python.BaseOptions(
        model_asset_path=POSE_MODEL_PATH
    )
    return PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_keypoints_from_frame(
    hand_result,
    pose_result,
) -> np.ndarray:
    """
    Construit le vecteur 258-dim depuis les résultats Tasks API.

    Ordre :
      [0:63]    → main gauche  (21 × 3)
      [63:126]  → main droite  (21 × 3)
      [126:258] → pose         (33 × 4)

    La Tasks API retourne les mains avec un label "Left"/"Right"
    dans hand_result.handedness. On trie explicitement.
    """
    # ── Mains : initialiser à zéro ──
    lh = np.zeros(DIM_HAND, dtype=np.float32)
    rh = np.zeros(DIM_HAND, dtype=np.float32)

    if hand_result and hand_result.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
            # Récupérer le label de la main (Left ou Right)
            handedness_label = (
                hand_result.handedness[i][0].category_name
                if hand_result.handedness
                else None
            )

            vec = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                dtype=np.float32,
            ).flatten()  # (63,)

            # ⚠️ MediaPipe renvoie "Left"/"Right" du point de vue
            #    de la caméra (image miroir) → Left caméra = main droite réelle
            #    On garde la convention MediaPipe pour la cohérence.
            if handedness_label == "Left":
                lh = vec
            elif handedness_label == "Right":
                rh = vec

    # ── Pose ──
    pose = np.zeros(DIM_POSE, dtype=np.float32)
    if pose_result and pose_result.pose_landmarks:
        pose = np.array(
            [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in pose_result.pose_landmarks[0]
            ],
            dtype=np.float32,
        ).flatten()  # (132,)

    return np.concatenate([lh, rh, pose])  # (258,)


def process_video(
    video_path: Path,
    hand_landmarker: HandLandmarker,
    pose_landmarker: PoseLandmarker,
) -> tuple:
    """
    Extrait les keypoints de toutes les frames d'une vidéo.

    Returns:
        (keypoints_array, n_frames) — shape (T, 258)
        (None, 0) en cas d'erreur.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0

    keypoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir BGR → RGB pour MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Créer l'objet mp.Image requis par la Tasks API
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        # Inférence
        hand_result = hand_landmarker.detect(mp_image)
        pose_result = pose_landmarker.detect(mp_image)

        kp = extract_keypoints_from_frame(hand_result, pose_result)
        keypoints.append(kp)

    cap.release()

    if not keypoints:
        return None, 0

    return np.array(keypoints, dtype=np.float32), len(keypoints)


def process_split(
    split: str,
    glossary: dict,
    hand_landmarker: HandLandmarker,
    pose_landmarker: PoseLandmarker,
) -> tuple:
    """
    Traite tous les fichiers .mp4 d'un split.

    Returns:
        (labels_dict, metadata_list, n_success, n_errors)
    """
    split_dir    = CASL_DIR / split
    features_dir = FEATURES_DIR / split
    features_dir.mkdir(parents=True, exist_ok=True)

    # Chercher .mp4 puis .avi si vide
    video_files = sorted(split_dir.glob("*.mp4"))
    if not video_files:
        video_files = sorted(split_dir.glob("*.avi"))

    if not video_files:
        print(f"   ⚠️  Aucune vidéo trouvée dans {split_dir}")
        return {}, [], 0, 0

    labels_dict   = {}
    metadata_list = []
    n_success     = 0
    n_errors      = 0

    for video_path in tqdm(video_files, desc=f"  {split:5s}", unit="vidéo"):
        stem          = video_path.stem
        gid, sid, tid = parse_filename(video_path.name)

        # Vérifier le format du nom
        if gid is None:
            logging.warning(f"[{split}] Format invalide : {video_path.name}")
            n_errors += 1
            continue

        # Vérifier que le geste est dans le glossaire
        if gid not in glossary:
            logging.warning(f"[{split}] {gid} absent du glossary.csv")
            n_errors += 1
            continue

        label = glossary[gid]

        # Extraction
        keypoints, n_frames = process_video(
            video_path, hand_landmarker, pose_landmarker
        )

        if keypoints is None or n_frames == 0:
            logging.error(f"[{split}] {video_path.name} — aucun keypoint extrait")
            n_errors += 1
            continue

        # Vérifier qu'au moins une frame a une détection non nulle
        non_zero = np.any(keypoints != 0, axis=1).sum()
        if non_zero == 0:
            logging.warning(
                f"[{split}] {video_path.name} — toutes les frames nulles"
            )
            n_errors += 1
            continue

        # Sauvegarder le .npy
        npy_path = features_dir / f"{stem}.npy"
        np.save(str(npy_path), keypoints)

        labels_dict[stem] = label
        metadata_list.append({
            "file"    : stem,
            "geste_id": gid,
            "sujet_id": sid,
            "take_id" : tid,
            "split"   : split,
            "label"   : label,
            "n_frames": n_frames,
        })
        n_success += 1

    return labels_dict, metadata_list, n_success, n_errors


def main():
    print("=" * 55)
    print("  Script 1 — extract_keypoints.py")
    print("  MediaPipe Tasks API — mediapipe 0.10.33")
    print("=" * 55)

    # ── 1. Télécharger les modèles si nécessaire ──
    download_models()

    # ── 2. Vérifications des fichiers d'entrée ──
    check_inputs()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── 3. Logger d'erreurs ──
    logging.basicConfig(
        filename=ERROR_LOG,
        filemode="w",
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── 4. Charger le glossaire ──
    glossary = load_glossary(GLOSSARY_CSV)

    # ── 5. Initialiser les landmarkers ──
    print("\n🔧 Initialisation des landmarkers MediaPipe Tasks...")
    hand_options = build_hand_options()
    pose_options = build_pose_options()

    # ── 6. Traitement de chaque split ──
    all_metadata = []
    summary      = {}

    print("\n📹 Extraction des keypoints...\n")

    with HandLandmarker.create_from_options(hand_options) as hand_lm, \
         PoseLandmarker.create_from_options(pose_options) as pose_lm:

        for split in SPLITS:
            labels_dict, metadata_list, n_ok, n_err = process_split(
                split, glossary, hand_lm, pose_lm
            )

            # Sauvegarder labels_{split}.json
            labels_path = FEATURES_DIR / f"labels_{split}.json"
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(labels_dict, f, ensure_ascii=False, indent=2)

            all_metadata.extend(metadata_list)
            summary[split] = {"success": n_ok, "errors": n_err}

    # ── 7. Sauvegarder metadata.json ──
    with open(FEATURES_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    # ── 8. Résumé final ──
    print(f"\n{'─'*55}")
    print("  RÉSUMÉ D'EXTRACTION")
    print(f"{'─'*55}")
    total_ok  = 0
    total_err = 0
    for split, counts in summary.items():
        ok  = counts["success"]
        err = counts["errors"]
        total_ok  += ok
        total_err += err
        status = "✓" if err == 0 else "⚠️ "
        print(
            f"  {status} Split {split:5s} : {ok:4d} vidéos traitées"
            + (f", {err} erreur(s)" if err else "")
        )
    print(f"{'─'*55}")
    print(f"  TOTAL : {total_ok} vidéos traitées, {total_err} erreur(s)")
    print(f"  Vecteur par frame : {DIM_TOTAL} dimensions")

    if total_err > 0:
        print(f"\n  📄 Détail des erreurs : {ERROR_LOG}")

    # ── 9. Statistiques sur les longueurs ──
    if all_metadata:
        lengths = [m["n_frames"] for m in all_metadata]
        print(f"\n  📊 Longueurs de séquences :")
        print(f"     Min    : {min(lengths)}")
        print(f"     Max    : {max(lengths)}")
        print(f"     Médiane: {int(np.median(lengths))}")
        print(f"     Moyenne: {np.mean(lengths):.1f}")

    print("\n✅ Extraction terminée — lance ensuite : python augment.py")


if __name__ == "__main__":
    main()
