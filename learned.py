"""Train a candidate filter and box refiner on paired DeepPCB training images."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from inspection import inspect


def features(reference, image, result):
    dx, dy = result["shift_px"]
    h, w = image.shape
    aligned = cv2.warpAffine(reference, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))
    residual = cv2.absdiff(aligned, image)
    rows = []
    for x, y, r, b in result["boxes"]:
        cx, cy = (x+r)//2, (y+b)//2
        row = [r-x, b-y]
        for radius in (4, 8, 16):
            region = np.s_[max(0,cy-radius):min(h,cy+radius), max(0,cx-radius):min(w,cx+radius)]
            for plane in (residual, aligned, image):
                patch = plane[region]
                row.extend(cv2.resize(patch, (4, 4), interpolation=cv2.INTER_AREA).ravel())
            row.append(np.mean(residual[region] > 80))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def read_pair(root, row):
    sample, annotation = row.split()
    base = (root / sample).with_suffix("")
    ref, image = (cv2.imread(str(base) + suffix, 0) for suffix in ("_temp.jpg", "_test.jpg"))
    truth = np.loadtxt(root / annotation, dtype=int, ndmin=2)[:, :4]
    truth[:, 2:] += 1
    return ref, image, truth


def candidates(reference, image):
    result = inspect(reference, image)
    if not result["boxes"]:
        return result, []
    return result, features(reference, image, result)


def train(root):
    xs, ys, offsets = [], [], []
    rows = (root / "trainval.txt").read_text().splitlines()
    test_ids = {line.split()[0] for line in (root / "test.txt").read_text().splitlines()}
    if len(rows) != 1000 or any(row.split()[0] in test_ids for row in rows):
        raise ValueError("expected disjoint official splits")
    for row in rows:
        reference, image, truth = read_pair(root, row)
        result, matrix = candidates(reference, image)
        for box, vector in zip(result["boxes"], matrix, strict=True):
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            delta = truth - [cx, cy, cx, cy]
            widths = truth[:, 2:] - truth[:, :2]
            distances = np.max(np.abs(delta[:, :2] + widths/2) / np.maximum(widths/2, 1), axis=1)
            nearest = np.argmin(distances)
            xs.append(vector)
            ys.append(distances[nearest] <= 0.85)
            offsets.append(delta[nearest])
    xs, ys = np.asarray(xs), np.asarray(ys)
    common = dict(n_estimators=48, min_samples_leaf=3, random_state=19, n_jobs=2)
    classifier = ExtraTreesClassifier(**common, class_weight="balanced").fit(xs, ys)
    regressor = ExtraTreesRegressor(**common, max_features=0.6).fit(xs[ys], np.asarray(offsets)[ys])
    return classifier, regressor


def predict(models, reference, image):
    result, matrix = candidates(reference, image)
    if not len(matrix):
        return result
    classifier, regressor = models
    scores = classifier.predict_proba(matrix)[:, 1]
    selected = np.flatnonzero(scores >= 0.5)
    refined = []
    if len(selected):
        for idx, offset in zip(selected, regressor.predict(matrix[selected]), strict=True):
            x,y,r,b = result["boxes"][idx]
            box = np.rint(offset + [(x+r)/2,(y+b)/2]*2).astype(int)
            box = np.clip(box, 0, [image.shape[1], image.shape[0]]*2)
            refined.append([int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])])
    keep = cv2.dnn.NMSBoxes(refined, scores[selected].tolist(), 0.49, 0.3) if refined else []
    boxes = [[x,y,x+w,y+h] for x,y,w,h in (refined[int(i)] for i in np.asarray(keep).ravel())]
    return {**result, "status": "suspect" if boxes else "clear", "boxes": boxes}


if __name__ == "__main__":
    import json
    from evaluate import evaluate

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    models = train(args.dataset)
    print(json.dumps(evaluate(args.dataset, detector=lambda r, i: predict(models, r, i)), indent=2))
