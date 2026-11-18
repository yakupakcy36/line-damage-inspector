"""Class-agnostic proposal coverage before classification on development images."""
import json
import sys
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from study import inventory, proposals


def objects(path):
    for obj in ET.parse(path).findall(".//object"):
        b = obj.find("bndbox")
        yield obj.findtext("name"), tuple(int(float(b.findtext(k)))-(k in ("xmin","ymin"))
                                          for k in ("xmin","ymin","xmax","ymax"))


def overlaps(boxes, target):
    extents = np.maximum(0, np.minimum(boxes[:, 2:], target[2:]) - np.maximum(boxes[:, :2], target[:2]))
    intersection = extents.prod(axis=1)
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    target_area = (target[2]-target[0])*(target[3]-target[1])
    return intersection/(areas+target_area-intersection), areas/target_area


def run(root):
    details = defaultdict(list)
    for key, split, ref, image, xml in inventory(root):
        if split == "test":
            continue
        boxes, _ = proposals(cv2.imread(str(ref),0), cv2.imread(str(image),0))
        boxes = np.array(boxes).reshape(-1,4)
        for category, target in objects(xml):
            ious, ratios = overlaps(boxes,target)
            best = ious.max(initial=0)
            ratio = ratios[ious.argmax()] if best else 0
            for name in ("all", key.split("_")[0], category.lower()):
                details[split+"/"+name].append((best,ratio))
    for key, rows in sorted(details.items()):
        best, ratios = np.asarray(rows).T
        missed = (best > 0) & (best <= .33)
        print(json.dumps(dict(key=key, labels=len(rows), covered=int((best>.33).sum()),
                             no_overlap=int((best==0).sum()), weak_overlap=int(missed.sum()),
                             median_missed_area_ratio=float(np.median(ratios[missed])) if missed.any() else None)))


if __name__ == "__main__":
    cv2.setNumThreads(1)
    run(Path(sys.argv[1]))
