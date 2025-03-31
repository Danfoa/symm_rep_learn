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
exp_name="GRF_${robot_name}"

exec_file="paper/experiments/GRF_uncertainty_regression.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=4"
opt_params="optim.lr=5e-4 architecture.residual_encoder=False "
model_params="robot_name=${robot_name}"
exp_params="exp_name=${exp_name} ${opt_params} ${model_params}"

PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=0 model=NCP  lstsq=False gamma=5,10      seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=1 model=NCP  lstsq=False gamma=25        seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=2 model=NCP  lstsq=False gamma=35,50     seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=3 model=ENCP lstsq=False gamma=5         seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=4 model=ENCP lstsq=False gamma=10        seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=5 model=ENCP lstsq=False gamma=25        seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=6 model=ENCP lstsq=False gamma=35,50     seed=0,1,2 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=7 model=CQR  optim.lr=1e-4,5e-5          seed=0,1,2 &
#python ${exec_file} ${hydra_kwargs
wait
