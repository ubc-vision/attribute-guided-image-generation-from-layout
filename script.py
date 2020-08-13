 #!/bin/bash
# a file for job output, you can check job progress
#SBATCH --output=/ubc/cs/home/m/mark1123/attribute-guided-image-generation-from-layout/logs/train128.out

# a file for errors
#SBATCH --error=/ubc/cs/home/m/mark1123/attribute-guided-image-generation-from-layout/logs/train128.err

# gpus per node
#SBATCH --gres=gpu:1

# number of requested nodes
#SBATCH --nodes=1

# memory per node
#SBATCH --mem=32000
#SBATCH --account=mark1123
#SBATCH --job-name=attribute-guided
#SBATCH --cpus-per-task=4
#SBATCH --partition=edith
#SBATCH --time=24:00:00

conda activate pytorch1.3

cd /ubc/cs/home/m/mark1123/attribute-guided-image-generation-from-layout
srun python train128.py