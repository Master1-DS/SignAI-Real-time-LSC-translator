#!/usr/bin/env python3
"""
extract_validated_segments.py
=============================
Extraction MediaPipe sur les segments VALIDÉS par l'humain
dans diagnostic/segments.csv.

Ne traite QUE les segments dont action == "keep".
Utilise label_validated (corrigé manuellement si besoin).

Utilisation :
    python3 extract_validated_segments.py --video Glossaire_LSC.mp4

Prérequis :
    diagnostic/segments.csv  (généré + édité par diagnostic_video_a.py)
    features/train/          (généré par extract_keypoints.py)
    hand_landmarker.task
    pose_landmarker.task
"""

import re
import csv
import cv2
import json
import argparse
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
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
DIAGNOSTIC_DIR  = Path("diagnostic")
SEGMENTS_CSV    = DIAGNOSTIC_DIR / "segments.csv"

FEATURES_DIR    = Path("features")
TRAIN_DIR       = FEATURES_DIR / "train"
LABELS_TRAIN    = FEATURES_DIR / "labels_train.json"
METADATA_FILE   = FEATURES_DIR / "metadata.json"

WIERZBICKI_GLOSSARY = Path("wierzbicki_glossary.csv")
CASL_GLOSSARY       = Path("glossary.csv")

HAND_MODEL_PATH = "hand_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker.task"

WIERZBICKI_PREFIX = "WA"

DIM_HAND  = 21 * 3
DIM_POSE  = 33 * 4
DIM_TOTAL = DIM_HAND * 2 + DIM_POSE
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Extraction MediaPipe sur segments validés."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Chemin vers la Vidéo A",
    )
    parser.add_argument(
        "--separate_variants",
        action="store_true",
        help="Ajoute le suffixe _WA aux labels (sépare des classes CASL)",
    )
    return parser.parse_args()


def check_inputs(video_path: Path) -> None:
    """Vérifie l'existence de tous les fichiers requis."""
    errors = []
    if not video_path.exists():
        errors.append(f"❌ Vidéo manquante : {video_path}")
    if not SEGMENTS_CSV.exists():
        errors.append(
            f"❌ Fichier manquant : {SEGMENTS_CSV}\n"
            "   Lance d'abord : python diagnostic_video_a.py"
        )
    if not TRAIN_DIR.exists():
        errors.append(
            f"❌ Dossier manquant : {TRAIN_DIR}\n"
            "   Lance d'abord : python extract_keypoints.py"
        )
    if not LABELS_TRAIN.exists():
        errors.append(f"❌ Fichier manquant : {LABELS_TRAIN}")
    if not Path(HAND_MODEL_PATH).exists():
        errors.append(f"❌ Modèle manquant : {HAND_MODEL_PATH}")
    if not Path(POSE_MODEL_PATH).exists():
        errors.append(f"❌ Modèle manquant : {POSE_MODEL_PATH}")

    if errors:
        for err in errors:
            print(err)
        sys.exit(1)


def load_validated_segments(csv_path: Path) -> list[dict]:
    """
    Charge les segments à extraire depuis segments.csv.

    Filtre :
      - action == "keep"
      - label_validated non vide

    Returns
    -------
    list[dict]
        Segments à extraire avec leurs labels validés.
    """
    segments = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            action = row.get("action", "keep").strip().lower()
            label  = row.get("label_validated", "").strip().upper()

            if action != "keep":
                continue
            if not label:
                continue

            segments.append({
                "id":          int(row["id"]),
                "label":       label,
                "frame_start": int(row["frame_start"]),
                "frame_end":   int(row["frame_end"]),
                "n_frames":    int(row["n_frames"]),
            })
    return segments


def load_casl_labels(csv_path: Path) -> set:
    """Charge les labels CASL pour distinguer connus / nouveaux."""
    labels = set()
    if not csv_path.exists():
        return labels
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.add(row["label"].strip().upper())
    return labels


def build_landmarkers() -> tuple:
    """Initialise HandLandmarker + PoseLandmarker."""
    hand_options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_options = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return (
        HandLandmarker.create_from_options(hand_options),
        PoseLandmarker.create_from_options(pose_options),
    )


def extract_keypoints_from_frame(hand_result, pose_result) -> np.ndarray:
    """Construit le vecteur 258-dim depuis les résultats Tasks API."""
    lh = np.zeros(DIM_HAND, dtype=np.float32)
    rh = np.zeros(DIM_HAND, dtype=np.float32)

    if hand_result and hand_result.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
            side = (
                hand_result.handedness[i][0].category_name
                if hand_result.handedness else None
            )
            vec = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                dtype=np.float32,
            ).flatten()
            if side == "Left":
                lh = vec
            elif side == "Right":
                rh = vec

    pose = np.zeros(DIM_POSE, dtype=np.float32)
    if pose_result and pose_result.pose_landmarks:
        pose = np.array(
            [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in pose_result.pose_landmarks[0]
            ],
            dtype=np.float32,
        ).flatten()

    return np.concatenate([lh, rh, pose])


def extract_segment(
    video_path: Path,
    frame_start: int,
    frame_end: int,
    hand_lm: HandLandmarker,
    pose_lm: PoseLandmarker,
) -> np.ndarray | None:
    """Extrait les keypoints d'un segment [frame_start, frame_end]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    keypoints = []
    frame_idx = frame_start

    while frame_idx <= frame_end:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        hand_result = hand_lm.detect(mp_image)
        pose_result = pose_lm.detect(mp_image)

        kp = extract_keypoints_from_frame(hand_result, pose_result)
        keypoints.append(kp)

        frame_idx += 1

    cap.release()

    if not keypoints:
        return None

    return np.array(keypoints, dtype=np.float32)


def make_stem(label: str, index: int) -> str:
    """Génère un nom de fichier unique (ex: WA_BONJOUR_001)."""
    safe_label = re.sub(r'[^A-Z0-9]', '_', label.upper())
    return f"{WIERZBICKI_PREFIX}_{safe_label}_{index:03d}"


def apply_variant_suffix(label: str, separate: bool) -> str:
    """Ajoute le suffixe _WA si --separate_variants est activé."""
    if separate and not label.endswith("_WA"):
        return f"{label}_WA"
    return label


def save_json(data: object, path: Path) -> None:
    """Sauvegarde JSON indenté."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_wierzbicki_glossary(
    label_counter: dict,
    casl_labels: set,
    path: Path,
) -> None:
    """Sauvegarde wierzbicki_glossary.csv."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "count", "is_new"])
        for label in sorted(label_counter.keys()):
            count       = label_counter[label]
            base        = label.replace("_WA", "")
            is_new      = base not in casl_labels
            writer.writerow([label, count, is_new])


def main() -> None:
    """Point d'entrée principal."""
    args       = parse_args()
    video_path = Path(args.video)

    print("=" * 60)
    print("  extract_validated_segments.py")
    print("  Extraction MediaPipe sur segments validés")
    print("=" * 60)

    # ── 0. Vérifications ──
    check_inputs(video_path)

    # ── 1. Charger les segments validés ──
    segments    = load_validated_segments(SEGMENTS_CSV)
    casl_labels = load_casl_labels(CASL_GLOSSARY)

    print(f"\n✓ {len(segments)} segments validés à extraire")

    if not segments:
        print("❌ Aucun segment validé (action='keep')")
        sys.exit(1)

    # ── 2. Charger les fichiers existants ──
    with open(LABELS_TRAIN, "r", encoding="utf-8") as f:
        labels_train = json.load(f)

    metadata = []
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    # ── 3. Initialiser MediaPipe ──
    print("\n🔧 Initialisation MediaPipe...")
    hand_lm, pose_lm = build_landmarkers()

    # ── 4. Extraction ──
    print(f"\n📹 Extraction des keypoints (LENT)...\n")

    label_counter: dict[str, int] = {}
    n_saved   = 0
    n_skipped = 0

    for seg in tqdm(segments, desc="  Extraction", unit="seg"):
        label = apply_variant_suffix(seg["label"], args.separate_variants)

        keypoints = extract_segment(
            video_path,
            seg["frame_start"],
            seg["frame_end"],
            hand_lm,
            pose_lm,
        )

        if keypoints is None or len(keypoints) == 0:
            print(f"\n   ⚠️  Segment #{seg['id']} [{label}] : aucun keypoint")
            n_skipped += 1
            continue

        non_zero = np.any(keypoints != 0, axis=1).sum()
        if non_zero == 0:
            print(f"\n   ⚠️  Segment #{seg['id']} [{label}] : aucune détection")
            n_skipped += 1
            continue

        # Nom unique
        label_counter[label] = label_counter.get(label, 0) + 1
        stem = make_stem(label, label_counter[label])

        # Sauvegarder le .npy
        npy_path = TRAIN_DIR / f"{stem}.npy"
        np.save(str(npy_path), keypoints)

        # Mettre à jour les dictionnaires
        labels_train[stem] = label
        metadata.append({
            "file":         stem,
            "geste_id":     f"WA_{label_counter[label]:03d}",
            "sujet_id":     "WA",
            "take_id":      f"T{label_counter[label]:02d}",
            "split":        "train",
            "label":        label,
            "n_frames":     len(keypoints),
            "frame_start":  seg["frame_start"],
            "frame_end":    seg["frame_end"],
            "source":       "wierzbicki_a",
            "diagnostic_id": seg["id"],
        })

        n_saved += 1

    hand_lm.close()
    pose_lm.close()

    # ── 5. Sauvegardes ──
    print(f"\n💾 Sauvegarde des fichiers")
    save_json(labels_train, LABELS_TRAIN)
    save_json(metadata, METADATA_FILE)
    save_wierzbicki_glossary(label_counter, casl_labels, WIERZBICKI_GLOSSARY)

    print(f"   ✓ labels_train.json mis à jour ({len(labels_train)} entrées)")
    print(f"   ✓ metadata.json mis à jour ({len(metadata)} entrées)")
    print(f"   ✓ wierzbicki_glossary.csv ({len(label_counter)} labels)")

    # ── 6. Résumé ──
    print(f"\n{'═'*60}")
    print(f"  RÉSUMÉ EXTRACTION")
    print(f"{'═'*60}")
    print(f"  Segments validés     : {len(segments)}")
    print(f"  Segments sauvegardés : {n_saved}    ✅")
    print(f"  Segments ignorés     : {n_skipped}  ⚠️")

    n_known = sum(1 for lbl in label_counter
                  if lbl.replace("_WA", "") in casl_labels)
    n_new   = len(label_counter) - n_known
    print(f"  Labels connus (CASL) : {n_known}")
    print(f"  Nouveaux labels      : {n_new}  🆕")

    print(f"\n  Distribution :")
    for label, count in sorted(label_counter.items()):
        base   = label.replace("_WA", "")
        marker = "🆕" if base not in casl_labels else "✓ "
        print(f"     {marker} {label:<25} : {count} segment(s)")

    print(f"{'═'*60}")
    print(f"\n✅ Terminé — lance ensuite : python augment.py")


if __name__ == "__main__":
    main()
