"""Class-agnostic localization on the official DeepPCB test split."""
import json
import sys
from pathlib import Path

import cv2

from inspection import inspect


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return intersection / (area(a) + area(b) - intersection)


def evaluate(root, detector=inspect):
    tp = predictions = labels = rejected = missed_boards = 0
    rows = (root / "test.txt").read_text().splitlines()
    if len(rows) != 500:
        raise ValueError("expected official 500-board split")
    for row in rows:
        sample, annotation = row.split()
        base = (root / sample).with_suffix("")
        reference = cv2.imread(str(base) + "_temp.jpg", 0)
        image = cv2.imread(str(base) + "_test.jpg", 0)
        result = detector(reference, image)
        boxes = result["boxes"]
        truth = []
        for line in (root / annotation).read_text().splitlines():
            x, y, right, bottom, _ = map(int, line.replace(",", " ").split())
            truth.append([x, y, right + 1, bottom + 1])
        pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(boxes) for j, b in enumerate(truth)), reverse=True)
        used, matched = set(), set()
        for overlap, i, j in pairs:
            if overlap > .33 and i not in used and j not in matched:
                used.add(i)
                matched.add(j)
        tp += len(used)
        predictions += len(boxes)
        labels += len(truth)
        rejected += result["status"] == "uninspectable"
        missed_boards += bool(truth) and result["status"] == "clear"
    return dict(boards=len(rows), localized=tp, annotations=labels, predicted_boxes=predictions,
                precision=tp/max(1, predictions), recall=tp/max(1, labels), uninspectable=rejected,
                defective_boards_called_clear=missed_boards)


if __name__ == "__main__":
    print(json.dumps(evaluate(Path(sys.argv[1])), indent=2))
