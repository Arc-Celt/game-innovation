#!/bin/bash
#SBATCH --job-name=siglip_embed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --account=def-lingjzhu
#SBATCH --output=slurm_out/siglip_embed_%j.out
#SBATCH --error=slurm_out/siglip_embed_%j.err

module load StdEnv/2023
module load python/3.12.4
cd /scratch/arccelt
source /scratch/arccelt/venvs/game-inn/bin/activate

start_time=$(date +%s)
start_human=$(date)
echo "🚀 Start time: $start_human"

python src/visual_feat_extraction.py

end_time=$(date +%s)
end_human=$(date)
total_time=$((end_time - start_time))

echo "✅ End time: $end_human"
echo "⏱️  Total time used: ${total_time} seconds"
