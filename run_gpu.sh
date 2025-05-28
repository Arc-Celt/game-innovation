#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:2
#SBATCH --ntasks-per-node=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --account=def-lingjzhu
#SBATCH --output=slurm_out/extract_aesthetics_%j.out
#SBATCH --error=slurm_out/extract_aesthetics_%j.err

module load StdEnv/2023 apptainer/1.3.5
# set cache directory
export APPTAINER_CACHEDIR=/scratch/arccelt/cache/apptainer
export HF_DATASETS_CACHE=/scratch/arccelt/cache/huggingface
export TRANSFORMERS_CACHE=/scratch/arccelt/cache/huggingface

#apptainer run -C --nv --home /project/6080355/lingjzhu/llm/home -W $SLURM_TMPDIR -B /home:/cluster_home -B /project -B /scratch /project/6080355/lingjzhu/llm/llm_env.sif bash /project/6080355/lingjzhu/llm/execute.sh
apptainer exec -C --nv \
  --home /scratch/arccelt/vllm \
  -W $SLURM_TMPDIR \
  -B /scratch:/scratch \
  /scratch/arccelt/vllm/pytorch-vllm.sif \
  bash /scratch/arccelt/vllm/run_extract_aesthetics.sh
# apptainer exec -C --nv --home /home/lingjzhu/scratch/vllm_env_home -W $SLURM_TMPDIR -B /project -B /scratch pytorch-vllm.sif bash /project/6080355/lingjzhu/llm/run_llama.sh
#apptainer exec -C --nv --home /home/lingjzhu/scratch/vllm_env_home -W $SLURM_TMPDIR -B /project -B /scratch pytorch-vllm.sif bash /project/6080355/lingjzhu/llm/run_embedding.sh
#apptainer exec -C --nv --home /scratch/lingjzhu/vllm_env_home -W /scratch/lingjzhu/vllm_env_home -B /project -B /scratch pytorch-camel.sif bash /project/6080355/lingjzhu/llm/run_agent_ollama.sh