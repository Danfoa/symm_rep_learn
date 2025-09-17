#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM

exp_name="SO2"

exec_file="paper/experiments/rotated_ordered_mnist.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=4"
opt_params="optim.limit_train_batches=0.5"
model_params=""
exp_params="exp_name=${exp_name} ${opt_params} ${model_params}"

PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=1 model=NCP architecture.embedding_dim=5     gamma=0.1 seed=0 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=2 model=NCP architecture.embedding_dim=10    gamma=0.1 seed=0 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=3 model=NCP architecture.embedding_dim=32    gamma=0.1 seed=0 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=4 model=NCP architecture.embedding_dim=64    gamma=0.1 seed=0 &
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=5 model=NCP architecture.embedding_dim=5     gamma=0.1 seed=0 dataset.augment=True&
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=6 model=NCP architecture.embedding_dim=10    gamma=0.1 seed=0 dataset.augment=True&
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=7 model=NCP architecture.embedding_dim=32    gamma=0.1 seed=0 dataset.augment=True&
PYTHONPATH=. python ${exec_file} ${hydra_kwargs} ${exp_params} device=7 model=NCP architecture.embedding_dim=64    gamma=0.1 seed=0 dataset.augment=True&
#python ${exec_file} ${hydra_kwargs
wait
