#!/bin/bash
#SBATCH --partition=gpu       # Specify the partition
#SBATCH --gres=gpu:1          # Request one GPU
#SBATCH --mem=100G            # Request 125GB of memory
#SBATCH --cpus-per-task=4     # Request 4 CPUs
#SBATCH --time=8:00:00        # Set maximum wall clock time
#SBATCH --output=output_%j.txt  # Output file with job ID in its name

# Load necessary modules
module load Python

# Execute the Python script and redirect output to a file
python train.py > output.log