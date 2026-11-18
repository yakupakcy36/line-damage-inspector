"""Invoke the delivered CLI in a clean working directory using synthetic pairs."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def check(root, a, b, expected):
    ref, query = root/"ref.png", root/"query.png"
    cv2.imwrite(str(ref), a)
    cv2.imwrite(str(query), b)
    raw = subprocess.check_output([sys.executable, "detector.py", "pku-expanded.skops", str(ref), str(query)])
    result = json.loads(raw)
    assert result["status"] == expected and result["boxes"] == [] and result["scores"] == [], result


with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    pattern = np.random.default_rng(19).integers(0,256,(160,160),dtype=np.uint8)
    check(root, pattern, pattern.copy(), "clear")
    check(root, pattern, pattern[:-1], "uninspectable")
    check(root, np.zeros_like(pattern), np.full_like(pattern,255), "uninspectable")
    check(root, pattern, np.roll(pattern,15,axis=0), "uninspectable")
