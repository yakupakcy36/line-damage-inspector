"""Run the fixed, expanded-box PKU detector on an aligned PCB image pair."""
import argparse
import json

import cv2
import numpy as np
import skops.io

from group_validation import expanded
from study import proposals
from quality import rejection_reason

THRESHOLD = .5


def predict(model, reference_path, image_path):
    ref = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if ref is None or image is None:
        raise ValueError("Could not read the reference or inspected image")
    reason = rejection_reason(ref,image)
    if reason:
        return {"status": "uninspectable", "reason": reason, "boxes": [], "scores": []}
    boxes, features = proposals(ref,image)
    if not boxes:
        return {"status": "clear", "boxes": [], "scores": []}
    regions = expanded(np.asarray(boxes), *image.shape[::-1])
    scores = model.predict_proba(np.asarray(features,dtype=np.float32))[:,1]
    keep = scores >= THRESHOLD
    return {"status": "suspect" if keep.any() else "clear",
            "boxes": regions[keep].tolist(), "scores": scores[keep].tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("reference")
    parser.add_argument("image")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    try:
        result = predict(skops.io.load(args.model,trusted=[]), args.reference,args.image)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result))
