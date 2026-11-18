import unittest

import cv2
import numpy as np

from inspection import inspect


class InspectionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.ref = cv2.resize(rng.integers(0, 2, (32, 32), dtype=np.uint8) * 255,
                              (256, 256), interpolation=cv2.INTER_NEAREST)

    def test_translated_intact_board(self):
        image = cv2.warpAffine(self.ref, np.float32([[1, 0, 4], [0, 1, -3]]), (256, 256))
        self.assertEqual(inspect(self.ref, image)["status"], "clear")

    def test_localizes_missing_material(self):
        self.ref[80:120, 80:120] = 255
        image = self.ref.copy()
        image[90:105, 92:110] = 0
        result = inspect(self.ref, image)
        self.assertEqual(result["boxes"], [[93, 91, 109, 104]])

    def test_blank_is_uninspectable(self):
        self.assertEqual(inspect(self.ref, self.ref * 0)["status"], "uninspectable")
