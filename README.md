# Line Damage Inspector

Locate changes on a printed circuit board from a fixed overhead camera and a matching intact template. The inspector estimates translation, checks registration quality and compares binary copper patterns while tolerating one pixel of edge error. It returns candidate defect boxes in the inspected image's coordinates (right and bottom exclusive). A low registration score or displacement above eight pixels yields `uninspectable`.

Run `python inspection.py reference.jpg inspected.jpg` with NumPy 2.2.6 and headless OpenCV 4.12.0.88. No trained weights or API credentials are needed. Read `excluded_border_px` before interpreting a clear result: the border has no reliable template overlap. Rotation, different PCB designs and changing camera scale are outside this first version's scope.

GitHub Actions runs the synthetic regression tests and evaluates the official 500-board [DeepPCB](https://github.com/tangsanli5201/DeepPCB) test split on a hosted CPU. Each run publishes class-agnostic precision and recall using greedy one-to-one matching at IoU > 0.33. Uninspectable boards contribute missed annotations. DeepPCB includes manually added defects and its authors specify research use. The defective-board test split cannot estimate the false reject rate for healthy production boards. No dataset images are redistributed here.
