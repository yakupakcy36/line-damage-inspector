"""Localize surface changes relative to a matching, intact PCB template."""
import argparse
import json

import cv2
import numpy as np


def inspect(reference, image, *, register=True):
    if (reference is None or image is None or reference.ndim != 2
            or reference.shape != image.shape or reference.dtype != np.uint8
            or image.dtype != np.uint8 or min(image.shape) < 32):
        raise ValueError("expected equally sized grayscale uint8 images, at least 32x32")
    h, w = image.shape
    shift, score = (cv2.phaseCorrelate(reference.astype(np.float32), image.astype(np.float32))
                    if register else ((0, 0), 1))
    if not np.isfinite([*shift, score]).all() or score < 0.25 or max(map(abs, shift)) > 8:
        return {"status": "uninspectable", "boxes": []}
    dx, dy = map(round, shift)
    aligned = cv2.warpAffine(reference, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))
    # Tolerate one pixel of residual edge misregistration in either direction.
    kernel = np.ones((3, 3), np.uint8)
    _, a = cv2.threshold(aligned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, b = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (a & ~cv2.dilate(b, kernel)) | (b & ~cv2.dilate(a, kernel))
    margin = max(abs(dx), abs(dy)) + 2
    mask[:margin] = mask[-margin:] = 0
    mask[:, :margin] = mask[:, -margin:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    stats = cv2.connectedComponentsWithStats(mask, connectivity=8)[2]
    boxes = [[int(x), int(y), int(x + width), int(y + height)]
             for x, y, width, height, area in stats[1:] if area >= 9]
    return {"status": "suspect" if boxes else "clear", "boxes": boxes,
            "shift_px": [dx, dy], "excluded_border_px": margin}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("image")
    args = parser.parse_args()
    print(json.dumps(inspect(cv2.imread(args.reference, 0), cv2.imread(args.image, 0))))
