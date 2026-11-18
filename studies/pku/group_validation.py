"""Compare two fixed box geometries by leaving each development template out."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from coverage import objects, overlaps
from study import inventory, metrics, proposals
from quality import rejection_reason


def expanded(boxes, width, height):
    half = (boxes[:, 2:] - boxes[:, :2]) / 2
    return np.column_stack((np.maximum(0, np.floor(boxes[:, :2]-half)),
                            np.minimum((width,height), np.ceil(boxes[:, 2:]+half)))).astype(int)


def prepare(root):
    groups = {}
    for key, split, ref, image, xml in inventory(root):
        if split == "test":
            continue
        query = cv2.imread(str(image), 0)
        template = cv2.imread(str(ref), 0)
        boxes, features = ([], []) if rejection_reason(template,query) else proposals(template,query)
        boxes = np.asarray(boxes).reshape(-1,4)
        truth = [target for _,target in objects(xml)]
        variants = {}
        for name, regions in (("original",boxes), ("expanded",expanded(boxes,*query.shape[::-1]))):
            positive = np.zeros(len(regions), dtype=bool)
            for target in truth:
                ious, _ = overlaps(regions,target)
                positive |= ious > .33
            variants[name] = (regions, positive)
        groups.setdefault(key.split("_")[0], []).append((np.asarray(features,dtype=np.float32).reshape(-1,9),
                                                        truth, variants))
    if set(groups) != {"01","04","05","06","07","09","10","11"}:
        raise ValueError("Unexpected development templates")
    return groups


def fit(rows, name):
    x = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[2][name][1] for r in rows])
    return ExtraTreesClassifier(n_estimators=64, min_samples_leaf=4, class_weight="balanced",
                                random_state=19, n_jobs=2).fit(x,y)


def validate(groups, variants):
    reports = []
    for name in variants:
        for holdout, held_rows in sorted(groups.items()):
            fitting = [r for g,rows in groups.items() if g != holdout for r in rows]
            model = fit(fitting,name)
            records = [(variants[name][0], model.predict_proba(features)[:,1] if len(features) else [], truth)
                       for features,truth,variants in held_rows]
            reports.append(dict(variant=name, held=holdout, threshold=.5, **metrics(records,.5)))
    return reports


if __name__ == "__main__":
    cv2.setNumThreads(1)
    for record in validate(prepare(Path(sys.argv[1])), ("original","expanded")):
        print(json.dumps(record))
