#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM

robot_name="go1"
exp_name="kinE_U_${robot_name}"

exec_file="paper/experiments/GRF_uncertainty_regression.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=4"
opt_params="optim.lr=1e-4 optim.max_epochs=800"
model_params="robot_name=${robot_name}"
exp_params="exp_name=${exp_name} ${opt_params} ${model_params}"

PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=0 model=NCP  architecture.residual_encoder=False lstsq=False gamma=1,5,   seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=1 model=NCP  architecture.residual_encoder=False lstsq=False gamma=25     seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=2 model=NCP  architecture.residual_encoder=False lstsq=False gamma=35,50  seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=3 model=NCP  architecture.residual_encoder=False lstsq=False gamma=10     seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=4 model=NCP  architecture.residual_encoder=False lstsq=False gamma=25     seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=5 model=ENCP architecture.residual_encoder=False lstsq=False gamma=35     seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=6 model=ENCP architecture.residual_encoder=False lstsq=False gamma=50     seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=7 model=CQR  optim.lr=1e-4,5e-5 seed=3,4,5 &
#python ${exec_file} ${hydra_kwargs
wait
