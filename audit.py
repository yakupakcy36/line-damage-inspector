"""Remote audit of the fixed detector on the pinned DeepPCB data."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import GroupKFold

from evaluate import evaluate_rows
from learned import predict, train_rows


def paths(row):
    return tuple(Path(p) for p in row.split())


def group(row):
    return paths(row)[0].parts[0]


def fingerprint(root, rows, suffix):
    hashes, thumbnails = [], []
    for row in rows:
        path = (root / paths(row)[0]).with_suffix("")
        img = cv2.imread(str(path) + suffix, cv2.IMREAD_GRAYSCALE)
        hashes.append(hashlib.sha256(img.tobytes()).hexdigest())
        thumbnails.append(cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA))
    return set(hashes), np.asarray(thumbnails, dtype=np.int16)


def image_overlap(root, training, official, suffix):
    train_hash, train_thumb = fingerprint(root, training, suffix)
    test_hash, test_thumb = fingerprint(root, official, suffix)
    similar = 0
    for idx, thumb in enumerate(test_thumb):
        # Mean absolute error <= 3 at full resolution implies near-identical
        # block means. Allow one extra level for integer rounding.
        neighbors = np.flatnonzero(np.abs(train_thumb-thumb).mean(axis=(1, 2)) <= 4)
        test_path = (root / paths(official[idx])[0]).with_suffix("")
        for j in neighbors:
            train_path = (root / paths(training[j])[0]).with_suffix("")
            a = cv2.imread(str(train_path) + suffix, 0)
            b = cv2.imread(str(test_path) + suffix, 0)
            if cv2.absdiff(a, b).mean() <= 3:
                similar += 1
                break
    return dict(shared_distinct_hashes=len(train_hash & test_hash),
                test_images_with_aligned_MAE_at_most_3=similar)


def counts(root, rows, model):
    report = evaluate_rows(root, rows, lambda a, b: predict(model, a, b))
    return {k: report[k] for k in ("localized", "predicted_boxes", "annotations")}


def run(root):
    training = (root / "trainval.txt").read_text().splitlines()
    official = (root / "test.txt").read_text().splitlines()
    groups = np.array([group(row) for row in training])
    report = dict(train_groups=sorted(set(groups)), test_groups=sorted({group(r) for r in official}))
    for i, key in enumerate(("sample", "annotation")):
        left = [paths(r)[i] for r in training]
        right = [paths(r)[i] for r in official]
        report[key] = dict(shared=len(set(left) & set(right)),
                           repeated_train=len(left)-len(set(left)), repeated_test=len(right)-len(set(right)))
    for suffix in ("_temp.jpg", "_test.jpg"):
        report[suffix] = image_overlap(root, training, official, suffix)
    print("SPLIT", json.dumps(report), flush=True)
    total = dict(localized=0, predicted_boxes=0, annotations=0)
    folds = []
    for fit, holdout in GroupKFold(n_splits=5).split(training, groups=groups):
        fit_rows = [training[i] for i in fit]
        val_rows = [training[i] for i in holdout]
        model = train_rows(root, fit_rows)
        validation = counts(root, val_rows, model)
        folds.append(dict(groups=sorted(set(groups[holdout])), image_pairs=len(val_rows),
                          fit=counts(root, fit_rows, model), validation=validation))
        for key in total:
            total[key] += validation[key]
    precision = total["localized"] / max(1,total["predicted_boxes"])
    recall = total["localized"] / max(1,total["annotations"])
    report.update(folds=folds, grouped_precision=precision, grouped_recall=recall)
    Path("group-validation.json").write_text(json.dumps(report,indent=2))
    if precision < .895 or recall < .91:
        raise RuntimeError(f"Grouped regression: precision={precision:.4f}, recall={recall:.4f}")


if __name__ == "__main__":
    cv2.setNumThreads(1)
    run(Path("dataset/PCBData"))
