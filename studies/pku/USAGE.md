# PKU expanded-box detector

Extract the complete workflow artifact and run from its directory using Python 3.12.

```sh
python -m pip install -r requirements.txt
python detector.py pku-expanded.skops reference.jpg inspected.jpg
```

Provide an intact reference and an aligned inspection image of the same design and field of view. The detector returns JSON containing class-agnostic boxes in inspected-image pixels (left, top, right, bottom, with exclusive upper bounds) and corresponding classifier scores. It uses a fixed 0.50 threshold. Only load model artifacts from a trusted source.

The report contains historical test results on templates 08 and 12. Eight development templates were used for fitting. The study uses unrotated PKU images with synthetic defects and has no healthy-board acceptance benchmark.
