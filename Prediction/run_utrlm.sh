  GNU nano 5.6.1                                                                                                                                       run_utrlm.sh                                                                                                                                                 
#!/bin/bash

# ===== TSUBAME module environment =====
module purge
module load cuda/11.8.0
module load cudnn/8.9.7

# ===== activate conda env =====
source ~/miniconda3/etc/profile.d/conda.sh
conda activate UTRLM

# ===== run python =====
python "$@"

