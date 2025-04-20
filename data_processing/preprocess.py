import os
import sys
import pandas as pd
import re
from dotenv import load_dotenv

load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH")


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load raw CSV dataset of fashion products.
    Raises a clear error if the file is missing or columns are absent.
    """
    if not path:
        sys.exit("Error: DATASET_PATH environment variable is not set.")
    if not os.path.isfile(path):
        sys.exit(f"Error: Dataset file not found at '{path}'.")
    df = pd.read_csv(path)
    required = {'product_id', 'description'}
    if not required.issubset(df.columns):
        sys.exit(f"Error: CSV missing required columns: {required - set(df.columns)}.")
    return df


def preprocess_text(text: str) -> str:
    """
    Basic text cleaning: lowercase, strip HTML tags, collapse whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    df = load_dataset()
    df['cleaned'] = df['description'].apply(preprocess_text)
    print(df[['product_id', 'cleaned']].head())
