"""Download the public Indian crop-yield CSV used by this project.

The file is a public mirror of the Indian crop-yield dataset described in the
Mendeley Data record (DOI: 10.17632/ncw2vbcgnk.2). The mirror is used because
it exposes the CSV directly for reproducible setup.
"""
from pathlib import Path
from urllib.request import Request, urlopen
import csv

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data" / "crop_yield.csv"
URLS = [
    "https://raw.githubusercontent.com/atharv-sh/crop_yeild_dataset/main/crop_yield.csv",
    "https://raw.githubusercontent.com/jainvandan7/crop-yield-ml/main/crop_yield_in_indian_states.csv",
]
EXPECTED = {"Crop", "Crop_Year", "Season", "State", "Area", "Production",
            "Annual_Rainfall", "Fertilizer", "Pesticide", "Yield"}


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in URLS:
        try:
            req = Request(url, headers={"User-Agent": "CropYield-AI-College-Project/1.0"})
            with urlopen(req, timeout=60) as response:
                raw = response.read()
            text = raw.decode("utf-8-sig")
            header = next(csv.reader(text.splitlines()))
            if not EXPECTED.issubset(set(header)):
                raise ValueError(f"Unexpected CSV columns: {header}")
            if len(text.splitlines()) < 1000:
                raise ValueError("Downloaded file is unexpectedly small")
            DEST.write_bytes(raw)
            print(f"Dataset downloaded: {DEST}")
            print(f"Rows (excluding header): {len(text.splitlines()) - 1:,}")
            print(f"Source mirror: {url}")
            return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise SystemExit("Could not download the dataset. Check your internet connection.\n" + "\n".join(errors))

if __name__ == "__main__":
    main()
