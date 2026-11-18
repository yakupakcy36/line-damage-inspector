"""Train on all development boards, verify the CLI, then score the historical split."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import skops.io

from coverage import objects
from detector import THRESHOLD, predict
from group_validation import fit, prepare, validate
from study import inventory, metrics


def run(root, output):
    output.mkdir(exist_ok=True, parents=True)
    groups = prepare(root)
    folds = validate(groups, ("expanded",))
    hits = sum(f["hits"] for f in folds)
    grouped_f1 = 2*hits / sum(f["predictions"]+f["labels"] for f in folds)
    if grouped_f1 < .76 or min(f["f1"] for f in folds) < .49:
        raise RuntimeError(f"Grouped PKU regression: {grouped_f1=:.4f}, folds={folds}")
    model = fit([r for rows in groups.values() for r in rows], "expanded")
    model_path = output/"pku-expanded.skops"
    skops.io.dump(model,model_path)
    rows = inventory(root)
    for _, split, ref, image, _ in rows:
        if split == "test":
            continue
        expected = predict(model,ref,image)
        if expected["boxes"]:
            cmd = [sys.executable,str(Path(__file__).with_name("detector.py")),str(model_path),str(ref),str(image)]
            actual = json.loads(subprocess.check_output(cmd))
            if actual["boxes"] != expected["boxes"] or not np.allclose(
                    actual["scores"], expected["scores"], rtol=0, atol=1e-12):
                raise RuntimeError("Saved model CLI predictions differ")
            break
    else:
        raise RuntimeError("No positive development prediction for CLI verification")

    loaded = skops.io.load(model_path,trusted=[])
    by_template = {}
    rejected = 0
    for key, split, ref, image, xml in rows:
        if split == "test":
            result = predict(loaded,ref,image)
            rejected += result["status"] == "uninspectable"
            record = (result["boxes"],result["scores"],[box for _,box in objects(xml)])
            by_template.setdefault(key.split("_")[0],[]).append(record)
    report = dict(development_images=sum(map(len,groups.values())), development_groups=sorted(groups),
                  decision_threshold=THRESHOLD, cli_parity=True, historical_uninspectable=rejected,
                  grouped_f1=grouped_f1, development_folds=folds,
                  historical_test=metrics([r for records in by_template.values() for r in records],THRESHOLD),
                  by_template={k:metrics(records,THRESHOLD) for k,records in by_template.items()})
    (output/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    cv2.setNumThreads(1)
    run(Path(sys.argv[1]),Path(sys.argv[2]))
