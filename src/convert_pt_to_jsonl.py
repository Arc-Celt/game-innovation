import torch
import os
import json
from tqdm import tqdm


embedding_dir = "embeddings/description_feats"
output_path = "embeddings/description_embeddings.jsonl"

with open(output_path, "w") as out_file:
    for fname in tqdm(sorted(os.listdir(embedding_dir))):
        if fname.endswith(".pt"):
            vector = torch.load(os.path.join(embedding_dir, fname)).tolist()
            game_id = fname.replace("_description.pt", "")
            entry = {
                "vector": vector,
                "label": game_id
            }
            out_file.write(json.dumps(entry) + "\n")
