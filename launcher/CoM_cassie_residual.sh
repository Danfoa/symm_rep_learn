#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM

robot_name="cassie"
exp_name="CoM_residual_${robot_name}"

exec_file="paper/experiments/CoM_regression.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=4"
opt_params="optim.train_sample_ratio=0.7 optim.lr=5e-4 optim.max_epochs=50"
model_params="robot_name=${robot_name}"
exp_params="exp_name=${exp_name} ${opt_params} ${model_params}"

python ${exec_file} ${hydra_kwargs} ${exp_params} device=0 model=ENCP  gamma=1  architecture.embedding_dim=256,64 seed=0 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=1 model=ENCP  gamma=10 architecture.embedding_dim=256,64 seed=0 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=2 model=ENCP  gamma=25 architecture.embedding_dim=256,64 seed=0 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=3 model=NCP   gamma=1  architecture.embedding_dim=256,64 seed=1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=4 model=NCP   gamma=10 architecture.embedding_dim=256,64 seed=1 &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=5 model=NCP   gamma=25 architecture.embedding_dim=256,64 seed=1 &
#python ${exec_file} ${hydra_kwargs} ${exp_params} device=6 model=EMLP  optim.lr=5e-4,1e-4                                             seed=0,1 &
#python ${exec_file} ${hydra_kwargs} ${exp_params} device=7 model=MLP   optim.lr=5e-4,1e-4                                             seed=0,1 &
#python ${exec_file} ${hydra_kwargs
wait
