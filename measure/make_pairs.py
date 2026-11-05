from pathlib import Path
import urllib.request
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data/flores200")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "common-parallel-corpora/common-parallel-corpora/"
    "master/data/common-parallel-corpora/"
)


FILES = {
    "dev": {
        "english": "flores-200-dev/eng_Latn.dev",
        "nepali": "flores-200-dev/npi_Deva.dev",
    },
    "devtest": {
        "english": "flores-200-devtest/eng_Latn.devtest",
        "nepali": "flores-200-devtest/npi_Deva.devtest",
    },
}


# --------------------------------------------------
# Download a file
# --------------------------------------------------

def download_file(url, output):

    if output.exists():
        print(f"Already exists: {output}")
        return

    print(f"Downloading:")
    print(f"  {url}")

    urllib.request.urlretrieve(
        url,
        output
    )

    print(f"Saved: {output}")


# --------------------------------------------------
# Download all four files
# --------------------------------------------------

for split, files in FILES.items():

    for language, relative_path in files.items():

        url = BASE_URL + relative_path

        output = DATA_DIR / Path(relative_path).name

        download_file(
            url,
            output
        )


# --------------------------------------------------
# Create English-Nepali sentence pairs
# --------------------------------------------------

pairs = []


for split, files in FILES.items():

    english_file = DATA_DIR / Path(
        files["english"]
    ).name

    nepali_file = DATA_DIR / Path(
        files["nepali"]
    ).name

    print(f"\nProcessing {split}")

    # Read English
    with open(
        english_file,
        encoding="utf-8"
    ) as f:

        english_lines = [
            line.strip()
            for line in f
        ]

    # Read Nepali
    with open(
        nepali_file,
        encoding="utf-8"
    ) as f:

        nepali_lines = [
            line.strip()
            for line in f
        ]

    print(
        f"English sentences: {len(english_lines)}"
    )

    print(
        f"Nepali sentences:  {len(nepali_lines)}"
    )

    # Make sure alignment is preserved
    if len(english_lines) != len(nepali_lines):

        raise ValueError(
            f"{split}: English and Nepali "
            "line counts don't match!"
        )

    # Pair line-by-line
    for english, nepali in zip(
        english_lines,
        nepali_lines
    ):

        if not english or not nepali:
            continue

        pairs.append({
            "split": split,
            "english": english,
            "nepali": nepali
        })


# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------

df = pd.DataFrame(pairs)

df.insert(
    0,
    "id",
    range(len(df))
)


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

output_file = DATA_DIR / "en_ne_pairs.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)


# --------------------------------------------------
# Show results
# --------------------------------------------------

print("\n======================================")
print("FLORES English-Nepali dataset ready")
print("======================================")

print(
    f"Total pairs: {len(df)}"
)

print(
    f"Output: {output_file}"
)

print("\nFirst 10 pairs:\n")

print(
    df.head(10).to_string(index=False)
)