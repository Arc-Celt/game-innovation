import polars as pl
import re
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Initialize vLLM
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.9,
    tensor_parallel_size=2
)

# Define sampling params
sampling_params = SamplingParams(
    temperature=0.2,
    top_p=1.0,
    max_tokens=80,
)

# Define prompt
def make_prompt(review):
    return f"""You are an assistant that always responds in English.

Summarize the player's perception of the game's visual and audio style from the review below in English.

Review: "{review}"

Return a JSON object like this:
{{
  "visual_perception": <string>,
  "audio_perception": <string>
}}

Instructions:
- Always respond in English. Translate any non-English content to English.
- If not mentioned or the review lacks meaningful content, use "None" string.
- Output only the JSON. Nothing else.
"""

# Extract JSON from model output
def extract_json_block(text):
    try:
        match = re.search(r'{\s*\"visual_perception\".*?\"audio_perception\".*?}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return {"visual_perception": None, "audio_perception": None}

# Load full dataset
df = pl.read_csv(
    "../data/user_reviews/processed/all_reviews_processed.csv",
    low_memory=True
)

# # # Test random 200 samples
# df = df.sample(n=200, seed=42)
rows = df.to_dicts()

# Run LLM and collect perception results
results = []
batch_size = 32
batched_rows = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]

for batch in tqdm(batched_rows):
    prompts = [make_prompt(row["review"]) for row in batch]
    responses = llm.generate(prompts, sampling_params)

    for row, res in zip(batch, responses):
        try:
            parsed = extract_json_block(res.outputs[0].text)
        except Exception as e:
            parsed = {"visual_perception": None, "audio_perception": None}
        results.append({
            "recommendationid": row["recommendationid"],
            "appid": row["appid"],
            "game": row["game"],
            "review": row["review"],
            "visual_perception": parsed["visual_perception"],
            "audio_perception": parsed["audio_perception"]
        })

# Save result
output_df = pl.DataFrame(results)
output_df.write_csv("../data/user_reviews/processed/aesthetic_perceptions.csv")
