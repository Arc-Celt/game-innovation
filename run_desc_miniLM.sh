#!/bin/bash
#SBATCH --job-name=desc_miniLM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --account=def-lingjzhu
#SBATCH --output=slurm_out/desc_miniLM_%j.out
#SBATCH --error=slurm_out/desc_miniLM_%j.err

module load StdEnv/2023
module load python/3.12.4
cd /scratch/arccelt
source /scratch/arccelt/venvs/game-inn/bin/activate

start_time=$(date +%s)
start_human=$(date)
echo "🚀 Start time: $start_human"

python src/description_embedding.py

end_time=$(date +%s)
end_human=$(date)
total_time=$((end_time - start_time))

echo "✅ End time: $end_human"
echo "⏱️  Total time used: ${total_time} seconds"
