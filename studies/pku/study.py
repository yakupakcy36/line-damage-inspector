"""Paired defect localization on the unrotated PKU-Market-PCB images."""
import argparse
import csv
import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import skops.io
from sklearn.ensemble import ExtraTreesClassifier


def inventory(root):
    images, templates, annotations = {}, {}, {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parent_names = {p.lower() for p in path.relative_to(root).parts[:-1]}
        if path.suffix.lower() == ".jpg" and "pcb_used" in parent_names:
            templates[int(path.stem)] = path
        elif path.suffix.lower() == ".jpg" and "images" in parent_names:
            images[path.stem.lower()] = path
        elif path.suffix.lower() == ".xml" and "annotations" in parent_names:
            annotations[path.stem.lower()] = path
    if not images or images.keys() != annotations.keys():
        raise ValueError(f"Image/XML mismatch: {len(images)} images, {len(annotations)} labels")
    groups = sorted({int(i.split("_")[0]) for i in images})
    if len(groups) < 6 or set(groups) - templates.keys():
        raise ValueError(f"Missing board groups or references: {groups}, {list(templates)}")
    random.Random(19).shuffle(groups)
    held = max(2, round(len(groups)*.2))
    subsets = {g: "test" if i < held else "validation" if i < held*2 else "train"
               for i, g in enumerate(groups)}
    return [(key, subsets[int(key.split("_")[0])], templates[int(key.split("_")[0])],
             images[key], annotations[key]) for key in sorted(images)]


def overlap(box, target):
    x, y, r, b = box
    xx, yy, rr, bb = target
    inter = max(0, min(r, rr)-max(x, xx))*max(0, min(b, bb)-max(y, yy))
    return inter / ((r-x)*(b-y)+(rr-xx)*(bb-yy)-inter)


def proposals(template, image):
    height, width = image.shape
    scale = min(1., 1200/max(width, height))
    shape = (round(width*scale), round(height*scale))
    ref = cv2.resize(template, shape, interpolation=cv2.INTER_AREA)
    query = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
    ref = cv2.GaussianBlur(ref, (3, 3), 0)
    query = cv2.GaussianBlur(query, (3, 3), 0)
    kernel = np.ones((3, 3), np.uint8)
    diff = np.maximum(cv2.subtract(query, cv2.dilate(ref, kernel)),
                      cv2.subtract(cv2.erode(ref, kernel), query))
    mask = cv2.morphologyEx(np.uint8(diff >= 12), cv2.MORPH_CLOSE, kernel)
    stats = cv2.connectedComponentsWithStats(mask)[2]
    boxes, vectors = [], []
    for x, y, w, h, area in stats[1:]:
        if area < 3 or max(w, h) > max(shape)*.2:
            continue
        patch = diff[y:y+h, x:x+w]
        vectors.append([w, h, area, area/(w*h), patch.mean(), patch.max(), patch.std(),
                        ref[y:y+h, x:x+w].mean(), query[y:y+h, x:x+w].mean()])
        boxes.append(tuple(round(int(v)/scale) for v in
                           (max(0,x-2), max(0,y-2), min(shape[0],x+w+2), min(shape[1],y+h+2))))
    return boxes, vectors


def metrics(records, threshold):
    hits = predicted = labels = 0
    for boxes, scores, truth in records:
        kept = [b for b, s in zip(boxes, scores, strict=True) if s >= threshold]
        pairs = sorted(((overlap(b, t), i, j) for i, b in enumerate(kept)
                        for j, t in enumerate(truth)), reverse=True)
        used, matched = set(), set()
        for iou, i, j in pairs:
            if iou > .33 and i not in used and j not in matched:
                used.add(i)
                matched.add(j)
        hits += len(used)
        predicted += len(kept)
        labels += len(truth)
    return dict(images=len(records), hits=hits, predictions=predicted, labels=labels,
                precision=hits/max(1,predicted), recall=hits/max(1,labels),
                f1=2*hits/max(1,predicted+labels))


def run(root, output):
    rows = inventory(root)
    output.mkdir(exist_ok=True, parents=True)
    data = {s: [] for s in ("train","validation","test")}
    seen = {}
    def check_image(image, subset):
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if seen.setdefault(digest, subset) != subset:
            raise ValueError("Identical decoded image across splits")
    with (output/"manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id","split","template","image","annotation"])
        for key, subset, ref_path, image_path, xml_path in rows:
            writer.writerow([key, subset, *(p.relative_to(root) for p in (ref_path,image_path,xml_path))])
            ref = cv2.imread(str(ref_path), 0)
            image = cv2.imread(str(image_path), 0)
            check_image(ref, subset)
            check_image(image, subset)
            boxes, features = proposals(ref, image)
            data[subset].append((boxes, features, xml_path))

    def labels(path):
        return [tuple(int(float(b.findtext(k)))-(k in ("xmin","ymin"))
                      for k in ("xmin","ymin","xmax","ymax"))
                for b in ET.parse(path).findall(".//object/bndbox")]

    samples, targets = [], []
    for boxes, vectors, xml_path in data["train"]:
        truth = labels(xml_path)
        samples.extend(vectors)
        targets.extend(any(overlap(box,t) > .33 for t in truth) for box in boxes)
    if len(set(targets)) != 2:
        raise ValueError("Training candidates need positives and negatives")
    model = ExtraTreesClassifier(n_estimators=64, min_samples_leaf=4, class_weight="balanced",
                                 random_state=19, n_jobs=2).fit(samples,targets)
    def score(subset):
        return [(boxes, model.predict_proba(vectors)[:,1] if vectors else [], labels(xml_path))
                for boxes,vectors,xml_path in data[subset]]

    validation = score("validation")
    threshold = max((.3,.5,.7,.9), key=lambda t: metrics(validation,t)["f1"])
    skops.io.dump(model, output/"pku.skops")
    report = dict(seed=19, threshold=threshold, groups={
        s: sorted({key.split("_")[0] for key, split, *_ in rows if split == s}) for s in data},
        train=metrics(score("train"),threshold), validation=metrics(validation,threshold),
        test=metrics(score("test"),threshold))
    (output/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    run(args.dataset,args.output)
