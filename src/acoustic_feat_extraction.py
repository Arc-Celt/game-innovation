import torch
import numpy as np
import librosa
from pathlib import Path
import os
import laion_clap
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Set up device and CLAP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
model.load_ckpt()
model.eval()


# Get audio embedding
def extract_audio_embedding(audio_path):
    """
    Extracts the audio embedding from a 48kHz mono .wav file using CLAP.

    Args:
        audio_path (Path): Path to the .wav file

    Returns:
        torch.Tensor: Embedding vector of shape (embedding_dim,)
    """
    try:
        audio_np, sr = librosa.load(str(audio_path), sr=48000, mono=True)
        if sr != 48000:
            raise ValueError(f"Expected 48kHz audio, got {sr}Hz")
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32).reshape(1, -1).to(device)
        audio_embed = model.get_audio_embedding_from_data(x=audio_tensor, use_tensor=True)
        return audio_embed[0]
    except Exception as e:
        print(f"Failed: {audio_path.name} — {e}")
        return None


# # Run on one sample audio file
# audio_path = Path("data/steam_games/game_audios/380_video_1_480p.wav")
# output_path = Path("embeddings/acoustic_feats") / (audio_path.stem + "_audio.pt")
# output_path.parent.mkdir(parents=True, exist_ok=True)

# embedding = extract_audio_embedding(audio_path)
# torch.save(embedding.cpu(), output_path)
# print(f"Saved embedding to: {output_path}")
# print("Embedding shape:", embedding.shape)


# Process all audios
input_dir = Path("data/steam_games/game_audios")
output_dir = Path("embeddings/acoustic_feats")
output_dir.mkdir(parents=True, exist_ok=True)

for audio_path in sorted(input_dir.glob("*.wav")):
    out_path = output_dir / (audio_path.stem + "_audio.pt")
    if out_path.exists():
        continue
    embedding = extract_audio_embedding(audio_path)
    if embedding is not None:
        torch.save(embedding.cpu(), out_path)
