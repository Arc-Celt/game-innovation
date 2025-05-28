import json
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE


# File paths
jsonl_path = "embeddings/description_embeddings.jsonl"
meta_path = "data/steam_games/meta_data/games_march2025_cleaned.csv"
output_path = "embeddings/description_embeddings_2d.csv"

# Load vectors + ids
ids = []
vectors = []

with open(jsonl_path, "r") as f:
    for line in f:
        obj = json.loads(line)
        ids.append(obj.get("label", ""))
        vectors.append(obj["vector"])

vectors = np.array(vectors)

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42, init='random', perplexity=30, max_iter=1000)
embedding_2d = tsne.fit_transform(vectors)

# Reload description from metadata
meta_df = pd.read_csv(meta_path)


def get_description(row):
    for col in ["detailed_description", "about_the_game", "short_description"]:
        if pd.notna(row[col]) and str(row[col]).strip():
            return row[col]
    return None

meta_df["description"] = meta_df.apply(get_description, axis=1)

meta_df["id"] = meta_df["appid"].astype(str)
final_df = pd.DataFrame({
    "id": ids,
    "x": embedding_2d[:, 0],
    "y": embedding_2d[:, 1]
})

final_df = final_df.merge(meta_df[["id", "appid", "name", "description"]], on="id", how="left")
final_df.to_csv(output_path, index=False)
