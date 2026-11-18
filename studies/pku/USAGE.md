# PKU expanded-box detector

Extract the complete workflow artifact and run from its directory using Python 3.12.

```sh
python -m pip install -r requirements.txt
python detector.py pku-expanded.skops reference.jpg inspected.jpg
```

Provide an intact reference and an aligned inspection image of the same design and field of view. The detector returns JSON containing class-agnostic boxes in inspected-image pixels (left, top, right, bottom, with exclusive upper bounds) and corresponding classifier scores. It uses a fixed 0.50 threshold. Only load model artifacts from a trusted source.

The report contains historical test results on templates 08 and 12. Eight development templates were used for fitting. The study uses unrotated PKU images with synthetic defects and has no healthy-board acceptance benchmark.

Read the status before the boxes. A valid comparison yields `suspect` or `clear`. An incompatible size, low contrast, template mismatch or poor registration yields `uninspectable` and a reason. Check the image acquisition and repeat inspection for those pairs. An unreadable file exits with an error.

The report includes eight leave-one-template-out development folds and the pooled F1. The historical two-template test was previously examined. CI requires grouped F1 of at least 0.76 and each held-out template F1 of at least 0.49, just below the prior 0.7736 pooled and 0.5091 lowest scores. Validating production acceptance requires independent new designs and independently confirmed healthy inspections.
