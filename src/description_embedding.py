import torch
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os


# Set up device and sentence transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

# Load the CSV file
csv_path = Path("data/steam_games/meta_data/games_march2025_cleaned.csv")
df = pd.read_csv(csv_path)


# Get descriptions
def get_description(row):
    """
    Select the best available game description from prioritized fields.

    Checks the following columns in order: 
    'detailed_description', 'about_the_game', and 'short_description'.

    Returns the first non-empty, non-null string.
    
    Args:
        row: A row from a pandas DataFrame representing a game.

    Returns:
        str or None: The best available description, or None if all are missing.
    """
    for col in ["detailed_description", "about_the_game", "short_description"]:
        if pd.notna(row[col]) and str(row[col]).strip():
            return row[col]
    return None


# Preprocess descriptions
df["description"] = df.apply(get_description, axis=1)
df = df[df["description"].notna()].reset_index(drop=True)

output_dir = Path("embeddings/description_feats")
output_dir.mkdir(parents=True, exist_ok=True)


# Encode and save
for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding descriptions"):
    appid = row["appid"]
    desc = row["description"]
    embedding = model.encode(desc, convert_to_tensor=True)
    torch.save(embedding.cpu(), output_dir / f"{appid}_description.pt")
