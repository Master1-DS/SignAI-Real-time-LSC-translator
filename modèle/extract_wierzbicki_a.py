#!/usr/bin/env python3
"""
extract_wierzbicki_a.py
=======================
Extrait les keypoints MediaPipe de la Vidéo A Wierzbicki (lexique illustré)
et les ajoute dans features/train/ pour enrichir le classifieur.

Stratégie validée par check_brightness.py :
  - Écran noir (luminosité < 20) = séparateur entre deux signes
  - Frame normale (luminosité > 20) = signe en cours
  - Texte blanc en bas = label OCR (lu pendant le signe)

Algorithme :
  1. Lire la vidéo frame par frame
  2. Calculer la luminosité de chaque frame
  3. Si écran noir → clore le segment en cours
  4. Sinon → OCR sur la zone basse + extraction MediaPipe
  5. Voter pour le label le plus fréquent dans le segment
  6. Garder TOUS les labels (même hors glossaire CASL)
  7. Sauvegarder le .npy dans features/train/
  8. Mettre à jour labels_train.json + metadata.json
  9. Générer wierzbicki_glossary.csv (tous les labels détectés)

Utilisation :
    python3 extract_wierzbicki_a.py --video Glossaire_LSC.mp4


Exécuter APRÈS :
    extract_keypoints.py   (features/train/ doit exister)
    check_brightness.py    (pour calibrer --black_threshold)
"""

import re
import csv
import cv2
import json
import logging
import argparse
import numpy as np
import pytesseract
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import sys


# ─── CONFIGURATION ───────────────────────────────────────────────────────────
FEATURES_DIR        = Path("features")
TRAIN_DIR           = FEATURES_DIR / "train"
LABELS_TRAIN        = FEATURES_DIR / "labels_train.json"
METADATA_FILE       = FEATURES_DIR / "metadata.json"

CASL_GLOSSARY       = Path("glossary.csv")               # lecture (généré par build_glossary.py)
WIERZBICKI_GLOSSARY = Path("wierzbicki_glossary.csv")    # généré par ce script

HAND_MODEL_PATH     = "hand_landmarker.task"
POSE_MODEL_PATH     = "pose_landmarker.task"
ERROR_LOG           = "extraction_errors_wierzbicki_a.log"

# Préfixe pour identifier les fichiers issus de la Vidéo A
WIERZBICKI_PREFIX = "WA"

# Dimensions des vecteurs MediaPipe (identiques à extract_keypoints.py)
DIM_HAND  = 21 * 3    # 63
DIM_POSE  = 33 * 4    # 132
DIM_TOTAL = DIM_HAND * 2 + DIM_POSE  # 258
# ─────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────
# Arguments CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Extraction keypoints Vidéo A Wierzbicki → features/train/"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Chemin vers la Vidéo A Wierzbicki (ex : video_a.mp4)",
    )
    parser.add_argument(
        "--black_threshold",
        type=float,
        default=20.0,
        help="Seuil de luminosité pour détecter l'écran noir "
             "(défaut : 20.0, calibré par check_brightness.py)",
    )
    parser.add_argument(
        "--bottom_ratio",
        type=float,
        default=0.15,
        help="Proportion de la hauteur à analyser en bas pour l'OCR "
             "(défaut : 0.15)",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=8,
        help="Nombre minimum de frames par segment pour le garder "
             "(défaut : 8)",
    )
    parser.add_argument(
        "--min_dark_frames",
        type=int,
        default=3,
        help="Nombre de frames noires consécutives pour valider "
             "un séparateur (défaut : 3)",
    )
    parser.add_argument(
        "--ocr_step",
        type=int,
        default=3,
        help="Lire l'OCR 1 frame sur N pendant un signe (défaut : 3)",
    )
    parser.add_argument(
        "--separate_variants",
        action="store_true",
        help="Si activé, sépare les variantes Wierzbicki des variantes CASL "
             "en ajoutant le suffixe _WA aux labels "
             "(ex : BONJOUR_WA au lieu de BONJOUR). "
             "Par défaut, fusionne avec les classes CASL.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────
# Vérifications des fichiers d'entrée
# ──────────────────────────────────────────────
def check_inputs(video_path: Path) -> None:
    """
    Vérifie l'existence de tous les fichiers requis avant traitement.

    Parameters
    ----------
    video_path : Path
        Chemin vers la Vidéo A.

    Raises
    ------
    SystemExit
        Si un fichier requis est manquant.
    """
    errors = []

    if not video_path.exists():
        errors.append(f"❌ Vidéo manquante : {video_path}")

    if not TRAIN_DIR.exists():
        errors.append(
            f"❌ Dossier manquant : {TRAIN_DIR}\n"
            "   Lance d'abord : python extract_keypoints.py"
        )

    if not LABELS_TRAIN.exists():
        errors.append(
            f"❌ Fichier manquant : {LABELS_TRAIN}\n"
            "   Lance d'abord : python extract_keypoints.py"
        )

    if not Path(HAND_MODEL_PATH).exists():
        errors.append(
            f"❌ Modèle manquant : {HAND_MODEL_PATH}\n"
            "   Lance d'abord : python extract_keypoints.py"
        )

    if not Path(POSE_MODEL_PATH).exists():
        errors.append(
            f"❌ Modèle manquant : {POSE_MODEL_PATH}\n"
            "   Lance d'abord : python extract_keypoints.py"
        )

    # Vérifier que pytesseract fonctionne
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        errors.append(
            "❌ Tesseract OCR non installé\n"
            "   Installer : sudo apt install tesseract-ocr tesseract-ocr-fra\n"
            "   Puis     : pip install pytesseract"
        )

    if errors:
        for err in errors:
            print(err)
        sys.exit(1)

    print("✓ Tous les fichiers d'entrée sont présents")


# ──────────────────────────────────────────────
# Chargement du glossaire CASL (référence)
# ──────────────────────────────────────────────
def load_casl_labels(csv_path: Path) -> set:
    """
    Charge glossary.csv (CASL) et retourne l'ensemble des labels valides.

    Sert UNIQUEMENT comme référence pour aider l'OCR (pas comme filtre strict).
    Les labels HORS glossaire CASL seront aussi gardés.

    Parameters
    ----------
    csv_path : Path
        Chemin vers glossary.csv.

    Returns
    -------
    set
        Ensemble des labels français en majuscules.
    """
    labels = set()

    if not csv_path.exists():
        print(f"⚠️  glossary.csv absent — mode OCR libre (sans référence)")
        return labels

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.add(row["label"].strip().upper())

    print(f"✓ glossary.csv (CASL) chargé : {len(labels)} labels de référence")
    return labels


# ──────────────────────────────────────────────
# Calcul de la luminosité
# ──────────────────────────────────────────────
def get_brightness(frame: np.ndarray) -> float:
    """
    Calcule la luminosité moyenne d'une frame.

    Utilise le canal V (Value) de HSV, plus représentatif
    que la moyenne RGB.

    Parameters
    ----------
    frame : np.ndarray
        Frame BGR.

    Returns
    -------
    float
        Luminosité entre 0 (noir) et 255 (blanc).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


# ──────────────────────────────────────────────
# OCR : lecture du label sur la zone basse
# ──────────────────────────────────────────────
def read_label_from_frame(
    frame: np.ndarray,
    bottom_ratio: float,
) -> str:
    """
    Extrait le texte affiché en bas de la frame par OCR.

    Le texte de la Vidéo A est BLANC sur fond sombre.
    Pipeline de prétraitement :
      1. Découper la zone basse
      2. Convertir en niveaux de gris
      3. Agrandir ×2 pour améliorer Tesseract
      4. Seuillage Otsu (auto-détecte le meilleur seuil)
      5. OCR Tesseract (PSM 7 = ligne unique, langue française)

    Parameters
    ----------
    frame : np.ndarray
        Frame BGR.
    bottom_ratio : float
        Proportion de la hauteur à analyser en bas.

    Returns
    -------
    str
        Texte détecté en majuscules, nettoyé.
        Chaîne vide si rien n'est détecté.
    """
    h, w = frame.shape[:2]

    # ── Découper la zone basse ──
    y_start = int(h * (1 - bottom_ratio))
    roi     = frame[y_start:h, 0:w]

    # ── Niveaux de gris ──
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ── Agrandir ×2 pour aider Tesseract ──
    scale    = 2
    enlarged = cv2.resize(
        gray,
        (w * scale, roi.shape[0] * scale),
        interpolation=cv2.INTER_LINEAR,
    )

    # ── Seuillage Otsu : isole le texte blanc automatiquement ──
    _, thresh = cv2.threshold(
        enlarged, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # ── OCR Tesseract ──
    # PSM 7 : traiter comme une seule ligne de texte
    # OEM 3 : moteur LSTM (le plus précis)
    config = "--psm 7 --oem 3 -l fra"
    raw    = pytesseract.image_to_string(thresh, config=config)

    # ── Nettoyage ──
    text = raw.upper().strip()
    # Garder lettres + accents + espaces uniquement
    text = re.sub(r'[^A-ZÀÂÄÉÈÊËÎÏÔÙÛÜÇ ]', '', text)
    # Supprimer espaces multiples
    text = ' '.join(text.split())
    # Garder uniquement les mots d'au moins 2 lettres
    words = [w for w in text.split() if len(w) >= 2]

    return ' '.join(words)


def correct_label_via_glossary(ocr_text: str, casl_labels: set) -> str:
    """
    Tente de corriger un texte OCR en utilisant le glossaire CASL comme référence.

    Stratégie :
      1. Match exact dans CASL → label CASL
      2. Match partiel : un label CASL est inclus dans le texte OCR → label CASL
      3. Sinon → garder le texte OCR original (nouveau label)

    Cela permet de :
      - Corriger les erreurs Tesseract sur les signes connus
      - Préserver les nouveaux signes (non rejetés)

    Parameters
    ----------
    ocr_text : str
        Texte brut issu de l'OCR.
    casl_labels : set
        Ensemble des labels du glossaire CASL.

    Returns
    -------
    str
        Label corrigé ou texte OCR original.
        Chaîne vide si OCR vide.
    """
    if not ocr_text:
        return ""

    if not casl_labels:
        return ocr_text  # mode OCR libre

    # Match exact
    if ocr_text in casl_labels:
        return ocr_text

    # Match partiel : chercher si un label CASL est contenu dans le texte
    for label in casl_labels:
        if label in ocr_text:
            return label

    # Aucun match → on garde le texte OCR (nouveau signe)
    return ocr_text


# ──────────────────────────────────────────────
# Initialisation des landmarkers MediaPipe
# ──────────────────────────────────────────────
def build_landmarkers() -> tuple:
    """
    Initialise HandLandmarker et PoseLandmarker (MediaPipe Tasks API).

    Returns
    -------
    tuple
        (hand_landmarker, pose_landmarker)
    """
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
    hand_lm = HandLandmarker.create_from_options(hand_options)
    pose_lm = PoseLandmarker.create_from_options(pose_options)
    return hand_lm, pose_lm


# ──────────────────────────────────────────────
# Extraction keypoints d'une frame
# ──────────────────────────────────────────────
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

    Parameters
    ----------
    hand_result : HandLandmarkerResult
    pose_result : PoseLandmarkerResult

    Returns
    -------
    np.ndarray
        Vecteur de shape (258,).
    """
    # ── Mains ──
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

    # ── Pose ──
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


# ──────────────────────────────────────────────
# Traitement principal de la vidéo
# ──────────────────────────────────────────────
def process_video(
    video_path: Path,
    black_threshold: float,
    bottom_ratio: float,
    min_dark_frames: int,
    min_frames: int,
    ocr_step: int,
    casl_labels: set,
    hand_lm: HandLandmarker,
    pose_lm: PoseLandmarker,
) -> list[dict]:
    """
    Traite la vidéo entière et retourne les segments extraits.

    Algorithme à état :
      ÉTAT "SIGN"  : on accumule les keypoints + on lit l'OCR
      ÉTAT "BLACK" : transition entre deux signes (ignorée)

      Transition SIGN → BLACK : après `min_dark_frames` frames noires
      Transition BLACK → SIGN : dès qu'une frame normale apparaît

    Parameters
    ----------
    video_path : Path
    black_threshold : float
    bottom_ratio : float
    min_dark_frames : int
    min_frames : int
    ocr_step : int
    casl_labels : set
        Glossaire CASL (référence pour correction OCR).
    hand_lm : HandLandmarker
    pose_lm : PoseLandmarker

    Returns
    -------
    list[dict]
        Liste de segments :
        [{"keypoints": np.ndarray (T, 258),
          "label": "BONJOUR",
          "frame_start": 50,
          "frame_end": 137,
          "n_frames": 88,
          "is_new": False}, ...]
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir : {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    print(f"   Durée : {total_frames/fps:.1f}s | "
          f"{total_frames} frames | {fps:.1f} FPS")

    segments: list[dict] = []

    # ── État courant ──
    state             = "BLACK"   # commence par un écran noir
    current_keypoints = []        # keypoints accumulés
    current_labels    = []        # votes OCR
    segment_start     = 0
    dark_streak       = 0         # nb de frames noires consécutives

    frame_idx = 0

    with tqdm(total=total_frames, desc="  Traitement", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            brightness = get_brightness(frame)
            is_dark    = brightness < black_threshold

            # ──────────────────────────────────────────────
            # Cas 1 : frame noire
            # ──────────────────────────────────────────────
            if is_dark:
                dark_streak += 1

                # Confirmation d'un séparateur après min_dark_frames
                if state == "SIGN" and dark_streak >= min_dark_frames:
                    # ── Clore le segment en cours ──
                    n = len(current_keypoints)
                    if n >= min_frames and current_labels:
                        # Label = vote majoritaire
                        label    = Counter(current_labels).most_common(1)[0][0]
                        is_new   = label not in casl_labels

                        kp_array = np.array(current_keypoints, dtype=np.float32)

                        segments.append({
                            "keypoints":   kp_array,
                            "label":       label,
                            "frame_start": segment_start,
                            "frame_end":   frame_idx - dark_streak,
                            "n_frames":    n,
                            "is_new":      is_new,
                        })

                        marker = "🆕" if is_new else "✓ "
                        print(
                            f"\n   {marker} Segment {len(segments):3d} : "
                            f"[{label:<20}] "
                            f"frames {segment_start:5d}-{frame_idx-dark_streak:5d} "
                            f"({n} frames)"
                        )

                    # Réinitialiser
                    current_keypoints = []
                    current_labels    = []
                    state             = "BLACK"

            # ──────────────────────────────────────────────
            # Cas 2 : frame normale
            # ──────────────────────────────────────────────
            else:
                # Transition BLACK → SIGN
                if state == "BLACK":
                    state         = "SIGN"
                    segment_start = frame_idx
                    dark_streak   = 0

                # ── OCR (1 frame sur N pour accélérer) ──
                if frame_idx % ocr_step == 0:
                    raw_text = read_label_from_frame(frame, bottom_ratio)
                    label    = correct_label_via_glossary(raw_text, casl_labels)
                    if label:
                        current_labels.append(label)

                # ── Extraction MediaPipe ──
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image  = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )
                hand_result = hand_lm.detect(mp_image)
                pose_result = pose_lm.detect(mp_image)

                kp = extract_keypoints_from_frame(hand_result, pose_result)
                current_keypoints.append(kp)

                dark_streak = 0

            frame_idx += 1
            pbar.update(1)

    cap.release()

    # ── Clore le dernier segment si la vidéo se termine en SIGN ──
    if state == "SIGN" and len(current_keypoints) >= min_frames and current_labels:
        label    = Counter(current_labels).most_common(1)[0][0]
        is_new   = label not in casl_labels
        kp_array = np.array(current_keypoints, dtype=np.float32)
        segments.append({
            "keypoints":   kp_array,
            "label":       label,
            "frame_start": segment_start,
            "frame_end":   frame_idx - 1,
            "n_frames":    len(current_keypoints),
            "is_new":      is_new,
        })

    return segments


# ──────────────────────────────────────────────
# Génération du nom de fichier unique
# ──────────────────────────────────────────────
def make_stem(label: str, index: int) -> str:
    """
    Génère un nom de fichier unique pour un segment Wierzbicki A.

    Format : WA_<LABEL>_<INDEX>
    Ex    : WA_BONJOUR_001

    Le préfixe WA_ permet de distinguer ces fichiers
    des fichiers CASL-W60 (Gxxx) dans features/train/.

    Parameters
    ----------
    label : str
        Label français du signe.
    index : int
        Index du segment.

    Returns
    -------
    str
        Nom de fichier sans extension.
    """
    safe_label = re.sub(r'[^A-Z0-9]', '_', label.upper())
    return f"{WIERZBICKI_PREFIX}_{safe_label}_{index:03d}"


def apply_variant_suffix(label: str, separate: bool) -> str:
    """
    Applique le suffixe _WA si le mode --separate_variants est activé.

    Parameters
    ----------
    label : str
        Label original (ex : "BONJOUR").
    separate : bool
        Si True, ajoute "_WA" au label.

    Returns
    -------
    str
        Label modifié ou inchangé.
    """
    if separate and not label.endswith("_WA"):
        return f"{label}_WA"
    return label


# ──────────────────────────────────────────────
# Mise à jour des fichiers JSON existants
# ──────────────────────────────────────────────
def load_labels_train() -> dict:
    """Charge labels_train.json existant."""
    with open(LABELS_TRAIN, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata() -> list:
    """Charge metadata.json existant ou retourne une liste vide."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_json(data: object, path: Path) -> None:
    """Sauvegarde un objet en JSON indenté."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_wierzbicki_glossary(
    labels_count: dict,
    casl_labels: set,
    path: Path,
) -> None:
    """
    Sauvegarde wierzbicki_glossary.csv avec tous les labels détectés.

    Format :
        label,count,is_new
        BONJOUR,3,False
        ECOLE,2,True
        ...

    Parameters
    ----------
    labels_count : dict
        {label: nombre_de_segments}
    casl_labels : set
        Glossaire CASL (pour marquer les nouveaux labels).
    path : Path
        Chemin du fichier CSV.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "count", "is_new"])
        for label in sorted(labels_count.keys()):
            count  = labels_count[label]
            is_new = label not in casl_labels
            writer.writerow([label, count, is_new])

    print(f"   ✓ wierzbicki_glossary.csv créé ({len(labels_count)} labels)")


# ──────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────
def main() -> None:
    """Point d'entrée principal."""
    args       = parse_args()
    video_path = Path(args.video)

    print("=" * 60)
    print("  extract_wierzbicki_a.py")
    print("  Extraction Vidéo A → features/train/")
    print("=" * 60)

    # ── 0. Vérifications ──
    check_inputs(video_path)

    # ── 1. Logger d'erreurs ──
    logging.basicConfig(
        filename=ERROR_LOG,
        filemode="w",
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # ── 2. Charger le glossaire CASL (référence pour OCR) ──
    casl_labels = load_casl_labels(CASL_GLOSSARY)

    # ── 3. Initialiser les landmarkers ──
    print("\n🔧 Initialisation des landmarkers MediaPipe...")
    hand_lm, pose_lm = build_landmarkers()

    # ── 4. Traiter la vidéo (segmentation + OCR + MediaPipe) ──
    print(f"\n📹 Traitement de la vidéo")
    print(f"   black_threshold   : {args.black_threshold}")
    print(f"   bottom_ratio      : {args.bottom_ratio}")
    print(f"   min_dark_frames   : {args.min_dark_frames}")
    print(f"   min_frames        : {args.min_frames}")
    print(f"   ocr_step          : 1 frame / {args.ocr_step}")
    print(f"   separate_variants : {args.separate_variants}\n")

    segments = process_video(
        video_path=video_path,
        black_threshold=args.black_threshold,
        bottom_ratio=args.bottom_ratio,
        min_dark_frames=args.min_dark_frames,
        min_frames=args.min_frames,
        ocr_step=args.ocr_step,
        casl_labels=casl_labels,
        hand_lm=hand_lm,
        pose_lm=pose_lm,
    )

    # Fermer les landmarkers
    hand_lm.close()
    pose_lm.close()

    print(f"\n   → {len(segments)} segment(s) extrait(s)")

    if not segments:
        print("\n❌ Aucun segment extrait")
        print("   Causes possibles :")
        print("   - Seuil noir mal calibré → relance check_brightness.py")
        print("   - OCR ne lit pas les labels → ajuste --bottom_ratio")
        sys.exit(1)

    # ── 5. Sauvegarder les segments dans features/train/ ──
    print(f"\n💾 Sauvegarde dans {TRAIN_DIR}/")

    labels_train = load_labels_train()
    metadata     = load_metadata()

    label_counter: dict[str, int] = {}
    n_saved   = 0
    n_skipped = 0

    for seg in segments:
        # Appliquer le suffixe _WA si demandé
        label     = apply_variant_suffix(seg["label"], args.separate_variants)
        keypoints = seg["keypoints"]

        # Vérifier qu'au moins une frame a une détection non nulle
        non_zero = np.any(keypoints != 0, axis=1).sum()
        if non_zero == 0:
            logging.warning(
                f"Segment [{label}] ignoré — aucune détection MediaPipe"
            )
            n_skipped += 1
            continue

        # ── Nom unique ──
        label_counter[label] = label_counter.get(label, 0) + 1
        stem = make_stem(label, label_counter[label])

        # ── Sauvegarder le .npy ──
        npy_path = TRAIN_DIR / f"{stem}.npy"
        np.save(str(npy_path), keypoints)

        # ── Mettre à jour les dictionnaires ──
        labels_train[stem] = label
        metadata.append({
            "file":        stem,
            "geste_id":    f"WA_{label_counter[label]:03d}",
            "sujet_id":    "WA",
            "take_id":     f"T{label_counter[label]:02d}",
            "split":       "train",
            "label":       label,
            "n_frames":    seg["n_frames"],
            "frame_start": seg["frame_start"],
            "frame_end":   seg["frame_end"],
            "source":      "wierzbicki_a",
            "is_new_class": seg["is_new"],
        })

        n_saved += 1

    # ── 6. Sauvegarder les fichiers JSON mis à jour ──
    save_json(labels_train, LABELS_TRAIN)
    save_json(metadata, METADATA_FILE)
    print(f"   ✓ labels_train.json mis à jour ({len(labels_train)} entrées)")
    print(f"   ✓ metadata.json mis à jour ({len(metadata)} entrées)")

    # ── 7. Générer wierzbicki_glossary.csv ──
    save_wierzbicki_glossary(label_counter, casl_labels, WIERZBICKI_GLOSSARY)

    # ── 8. Résumé final ──
    print(f"\n{'═'*60}")
    print(f"  RÉSUMÉ EXTRACTION VIDÉO A")
    print(f"{'═'*60}")
    print(f"  Segments détectés       : {len(segments)}")
    print(f"  Segments sauvegardés    : {n_saved}    ✅")
    print(f"  Segments ignorés        : {n_skipped}  ⚠️  (sans détection)")
    print(f"  Labels uniques détectés : {len(label_counter)}")

    # Distribution des labels
    n_known = sum(1 for lbl in label_counter
                  if lbl.replace("_WA", "") in casl_labels)
    n_new   = len(label_counter) - n_known

    print(f"  Labels présents dans CASL : {n_known}")
    print(f"  Nouveaux labels           : {n_new}  🆕")

    print(f"\n  Distribution des labels :")
    for label, count in sorted(label_counter.items()):
        base_label = label.replace("_WA", "")
        marker     = "🆕" if base_label not in casl_labels else "✓ "
        print(f"     {marker} {label:<25} : {count} segment(s)")

    if n_skipped > 0:
        print(f"\n  📄 Détail des ignorés : {ERROR_LOG}")

    print(f"{'═'*60}")
    print(f"\n✅ Terminé — lance ensuite : python augment.py")


if __name__ == "__main__":
    main()
