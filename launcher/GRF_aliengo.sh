#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM

robot_name="aliengo"
exp_name="GRF_UC_${robot_name}_symm_256"

exec_file="paper/experiments/GRF_uncertainty_regression2.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=6"
opt_params="optim.lr=1e-4 optim.max_epochs=250 +sd=100 architecture.embedding_dim=256"
model_params="robot_name=${robot_name}"
exp_params="exp_name=${exp_name} ${opt_params} ${model_params}"

#PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=0 model=NCP  architecture.residual_encoder=False,True lstsq=True gamma=5,   seed=0,1 &
#PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=1 model=NCP  architecture.residual_encoder=False,True lstsq=True gamma=25   seed=0,1 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=2 model=NCP   architecture.residual_encoder=False lstsq=True  gamma=5,50       seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=3 model=NCP   architecture.residual_encoder=False lstsq=True  gamma=25,10       seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=4 model=ENCP  architecture.residual_encoder=False lstsq=True  gamma=50       seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=5 model=ENCP  architecture.residual_encoder=False lstsq=True  gamma=25       seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=6 model=ENCP  architecture.residual_encoder=False lstsq=True  gamma=10       seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=7 model=ENCP  architecture.residual_encoder=False lstsq=True  gamma=5       seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=3 model=ECQR                                                     seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=2 model=CQR                                                      seed=0,1,2 &
#python ${exec_file} ${hydra_kwargs
wait
