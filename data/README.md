# Dataset

The project includes `download_dataset.py` at the project root. From the project folder run:

```bash
python download_dataset.py
```

It downloads the real Indian crop-yield observations CSV to `data/crop_yield.csv` and checks the expected schema and that the file is not truncated. Internet access is required for this one-time download.

Dataset record and attribution:

Ramesh, V. & Kumaresan, P. (2025). *Stacked Ensemble Model for Accurate Crop Yield Prediction Using Machine Learning Techniques*. Mendeley Data, Version 2. https://doi.org/10.17632/ncw2vbcgnk.2 (CC BY 4.0). The record describes 19,689 observations from 1997–2020, 27 Indian states and 3 Union Territories, and 55 crops.

The CSV is fetched from a public GitHub mirror of this Indian crop-yield dataset. The source mirror README reports the same 19,689 observations and matching ten-column schema: https://github.com/atharv-sh/crop_yeild_dataset. Retain the Mendeley attribution in reports and presentations.

`Production` is kept in the raw source CSV for traceability but is excluded from model predictors to avoid target leakage. Yield is the regression target.
