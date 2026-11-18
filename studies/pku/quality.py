"""Input checks for aligned, grayscale, unrotated template/query pairs."""
import cv2
import numpy as np


def rejection_reason(reference, image):
    if reference.shape != image.shape:
        return "different_dimensions"
    if min(image.shape) < 32:
        return "insufficient_resolution"
    height, width = image.shape
    scale = min(1., 768 / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    a, b = (cv2.resize(im, size, interpolation=cv2.INTER_AREA).astype(np.float32)
            for im in (reference, image))
    if min(a.std(), b.std()) < 3:
        return "insufficient_contrast"
    similarity = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED).item()
    if similarity < .80:
        return "template_mismatch"
    window = cv2.createHanningWindow(size, cv2.CV_32F) if min(size) > 1 else None
    shift, response = cv2.phaseCorrelate(a, b, window)
    if not np.isfinite([*shift,response]).all() or response < .2 or max(map(abs,shift)) > max(1.,4*scale):
        return "poor_alignment"
    return None
