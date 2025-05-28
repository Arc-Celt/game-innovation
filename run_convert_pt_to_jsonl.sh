#!/bin/bash
#SBATCH --job-name=convert-jsonl
#SBATCH --output=slurm_out/convert_jsonl_%j.out
#SBATCH --error=slurm_out/convert_jsonl_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8192M
#SBATCH --cpus-per-task=4

module load StdEnv/2023
module load python/3.12.4
cd /scratch/arccelt
source /scratch/arccelt/venvs/game-inn/bin/activate

start_time=$(date +%s)
start_human=$(date)
echo "🚀 Start time: $start_human"

python src/convert_pt_to_jsonl.py

end_time=$(date +%s)
end_human=$(date)
total_time=$((end_time - start_time))

echo "✅ End time: $end_human"
echo "⏱️  Total time used: ${total_time} seconds"
