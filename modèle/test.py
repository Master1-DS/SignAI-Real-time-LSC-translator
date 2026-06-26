#!/usr/bin/env python3
"""
fix_split_v4.py – Recrée les splits train/val/test sans chevauchement de sujets,
avec les fichiers Wierzbicki (WA_) forcés dans le train.
"""

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

SEED = 42
random.seed(SEED)

FEATURES_ORIG = Path("features")
FEATURES_CLEAN = Path("features_clean")
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

def main():
    if FEATURES_CLEAN.exists():
        shutil.rmtree(FEATURES_CLEAN)
    FEATURES_CLEAN.mkdir()

    # Charger le metadata original
    with open(FEATURES_ORIG / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Séparer les fichiers Wierzbicki (préfixe WA_) des fichiers CASL
    wierz_entries = [e for e in metadata if e["file"].startswith("WA_")]
    casl_entries  = [e for e in metadata if not e["file"].startswith("WA_")]

    # 1. Tous les Wierzbicki vont dans le train
    train_entries = wierz_entries.copy()
    train_count = len(wierz_entries)

    # 2. Répartir les fichiers CASL par sujet
    subjects = defaultdict(list)
    for entry in casl_entries:
        sujet = entry["sujet_id"]
        subjects[sujet].append(entry)

    total_casl = len(casl_entries)
    target_train_casl = int(total_casl * TRAIN_RATIO)
    target_val_casl   = int(total_casl * VAL_RATIO)
    target_test_casl  = total_casl - target_train_casl - target_val_casl

    # Trier les sujets par taille décroissante
    sorted_sujets = sorted(subjects.items(), key=lambda x: len(x[1]), reverse=True)

    train_sujets = set()
    val_sujets = set()
    test_sujets = set()
    train_casl_count = 0
    val_casl_count = 0
    test_casl_count = 0

    for sujet, entries in sorted_sujets:
        # Déficits
        deficit_train = target_train_casl - train_casl_count
        deficit_val   = target_val_casl - val_casl_count
        deficit_test  = target_test_casl - test_casl_count

        if deficit_train >= deficit_val and deficit_train >= deficit_test:
            train_sujets.add(sujet)
            train_casl_count += len(entries)
        elif deficit_val >= deficit_train and deficit_val >= deficit_test:
            val_sujets.add(sujet)
            val_casl_count += len(entries)
        else:
            test_sujets.add(sujet)
            test_casl_count += len(entries)

    # Ajouter les CASL aux ensembles
    for entry in casl_entries:
        sujet = entry["sujet_id"]
        if sujet in train_sujets:
            train_entries.append(entry)
        elif sujet in val_sujets:
            val_entries = [entry] if 'val_entries' not in locals() else val_entries + [entry]
        else:
            test_entries = [entry] if 'test_entries' not in locals() else test_entries + [entry]

    # Initialiser les listes si elles n'existent pas
    if 'val_entries' not in locals():
        val_entries = []
    if 'test_entries' not in locals():
        test_entries = []

    # Vérifier les proportions
    print(f"Répartition voulue (CASL) : train={TRAIN_RATIO:.0%}, val={VAL_RATIO:.0%}, test={TEST_RATIO:.0%}")
    print(f"Répartition obtenue (CASL) : train={train_casl_count}/{total_casl} ({train_casl_count/total_casl:.1%}), "
          f"val={val_casl_count}/{total_casl} ({val_casl_count/total_casl:.1%}), "
          f"test={test_casl_count}/{total_casl} ({test_casl_count/total_casl:.1%})")
    print(f"Total Wierzbicki dans train : {len(wierz_entries)}")
    print(f"Train total : {len(train_entries)}, Val total : {len(val_entries)}, Test total : {len(test_entries)}")

    # Copier les fichiers .npy et générer les labels
    new_labels = {split: {} for split in ["train", "val", "test"]}
    new_metadata = []

    def process_entries(entries, split):
        for entry in entries:
            stem = entry["file"]
            src = FEATURES_ORIG / entry["split"] / f"{stem}.npy"
            if not src.exists():
                print(f"⚠️ Fichier source manquant : {src}")
                continue
            dst_dir = FEATURES_CLEAN / split
            dst_dir.mkdir(exist_ok=True)
            shutil.copy2(src, dst_dir / f"{stem}.npy")
            new_labels[split][stem] = entry["label"]
            entry["split"] = split
            new_metadata.append(entry)

    process_entries(train_entries, "train")
    process_entries(val_entries, "val")
    process_entries(test_entries, "test")

    # Sauvegarder les labels
    for split in ["train", "val", "test"]:
        with open(FEATURES_CLEAN / f"labels_{split}.json", "w") as f:
            json.dump(new_labels[split], f, indent=2)

    # Sauvegarder le metadata_clean.json
    with open(FEATURES_CLEAN / "metadata_clean.json", "w") as f:
        json.dump(new_metadata, f, indent=2)

    print("\n✅ features_clean généré avec splits propres (Wierzbicki dans train)")

if __name__ == "__main__":
    main()
