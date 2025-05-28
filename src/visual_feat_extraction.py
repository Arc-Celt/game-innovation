import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
from decord import VideoReader, cpu
from pathlib import Path
import os


# Set up device, processor and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").to(device)
model.eval()


# Sample 10 evenly spaced frames using Decord
def sample_frames_decord(video_path, num_frames=10):
    """
    Extracts evenly spaced frames from a video using Decord.

    Args:
        video_path (Path): Path to the video file
        num_frames (int): Number of frames to sample from the video

    Returns:
        List[PIL.Image]: A list of frames as PIL Images
    """
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(frame) for frame in frames]


# Get image embeddings from SigLIP2
def get_image_embeddings(frames):
    """
    Extracts image embeddings from a list of frames using the SigLIP2 model.

    Args:
        frames (List[PIL.Image]): List of frames as PIL Images

    Returns:
        torch.Tensor: Image embeddings of shape (num_frames, embedding_dim)
    """
    inputs = processor(images=frames, return_tensors="pt", padding=True)
    with torch.no_grad():
        pixel_values = inputs["pixel_values"].to(device)
        features = model.get_image_features(pixel_values=pixel_values)
    return features  # (num_frames, embedding_dim)


# Average embedding over frames
def extract_avg_embedding(video_path):
    """
    Extracts the average embedding from a video.

    Args:
        video_path (Path): Path to the video file

    Returns:
        torch.Tensor: Average embedding of shape (embedding_dim,)
    """
    try:
        frames = sample_frames_decord(video_path)
        if not frames:
            raise ValueError("No frames extracted.")
        embeddings = get_image_embeddings(frames)
        return embeddings.mean(dim=0)
    except Exception as e:
        print(f"Failed to process {video_path}: {e}")
        return None


# # Run on one sample clip
# video_path = Path("data/steam_games/game_clips/380_video_1_480p.mp4")
# output_path = Path("embeddings/visual_feats") / (video_path.stem + ".pt")
# output_path.parent.mkdir(parents=True, exist_ok=True)

# avg_embedding = extract_avg_embedding(video_path)
# torch.save(avg_embedding.cpu(), output_path)
# print(f"Saved embedding to: {output_path}")
# print("Embedding shape:", avg_embedding.shape)


# Process all clips
input_dir = Path("data/steam_games/game_clips")
output_dir = Path("embeddings/visual_feats")
output_dir.mkdir(parents=True, exist_ok=True)

for video_path in sorted(input_dir.glob("*.mp4")):
    out_file = output_dir / (video_path.stem + ".pt")
    if out_file.exists():
        continue
    embedding = extract_avg_embedding(video_path)
    if embedding is not None:
        torch.save(embedding.cpu(), out_file)
