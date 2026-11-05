from pathlib import Path
from llama_cpp import Llama
import pandas as pd


MODEL = str(
    Path(
        "~/Desktop/kv/inference/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
    ).expanduser()
)

llm = Llama(
    model_path=MODEL,
    vocab_only=True,
    verbose=False
)


df = pd.read_csv("./data/flores200/en_ne_pairs.csv")

# Remove missing values
df = df.dropna(subset=["english", "nepali"])

# Make sure they are strings
df["english"] = df["english"].astype(str)
df["nepali"] = df["nepali"].astype(str)


def count_tokens(text):
    tokens = llm.tokenize(
        text.encode("utf-8"),
        add_bos=False
    )

    return len(tokens)


def count_words(text):
    return len(text.split())


# Token counts
df["english_tokens"] = df["english"].apply(count_tokens)
df["nepali_tokens"] = df["nepali"].apply(count_tokens)

# Word counts
df["english_words"] = df["english"].apply(count_words)
df["nepali_words"] = df["nepali"].apply(count_words)

# Fertility
df["english_fertility"] = (
    df["english_tokens"] / df["english_words"]
)

df["nepali_fertility"] = (
    df["nepali_tokens"] / df["nepali_words"]
)

# Token expansion
df["token_expansion_ratio"] = (
    df["nepali_tokens"] / df["english_tokens"]
)


df.to_csv(
    "fertility_results.csv",
    index=False
)

print(f"Processed {len(df)} sentence pairs.")
print(df.head())