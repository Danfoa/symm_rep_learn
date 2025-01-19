#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM


exp_label="baseline_final"

exec_file="NCP/examples/symm_GMM.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=3"
opt_params="batch_size=1048 lr=5e-4 max_epochs=7000"
exp_params="exp_label=${exp_label} gmm.n_samples=30000,10000,5000 seed=0,1,2,3 embedding.hidden_layers=3 embedding.hidden_units=64 truncated_op_bias=full_rank"
gamma_ncp="gamma=0.1,0.01"
python ${exec_file} ${hydra_kwargs} ${exp_params} device=0 model=NCP  regular_multiplicitiy=0,5,10  gmm.n_kernels=2,5    ${gamma_ncp}    embedding.embedding_dim=32 ${opt_params} &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=1 model=NCP  regular_multiplicitiy=0,5,10  gmm.n_kernels=10,20  ${gamma_ncp}    embedding.embedding_dim=32 ${opt_params} &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=2 model=ENCP regular_multiplicitiy=0,5,10 gmm.n_kernels=2      ${gamma_ncp}     embedding.embedding_dim=32 ${opt_params} &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=3 model=ENCP regular_multiplicitiy=0,5,10 gmm.n_kernels=5      ${gamma_ncp}     embedding.embedding_dim=32 ${opt_params} &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=4 model=ENCP regular_multiplicitiy=0,5,10 gmm.n_kernels=10,20  ${gamma_ncp}     embedding.embedding_dim=32 ${opt_params} &
python ${exec_file} --multirun hydra.launcher.n_jobs=2 ${exp_params} device=5 model=DRF regular_multiplicitiy=0,5,10  gmm.n_kernels=2,5    ${gamma_ncp}    embedding.embedding_dim=32 ${opt_params} &
python ${exec_file} --multirun hydra.launcher.n_jobs=2 ${exp_params} device=6 model=DRF regular_multiplicitiy=0,5,10  gmm.n_kernels=10     ${gamma_ncp}    embedding.embedding_dim=32 ${opt_params} &
python ${exec_file} --multirun hydra.launcher.n_jobs=2 ${exp_params} device=7 model=DRF regular_multiplicitiy=0,5,10  gmm.n_kernels=20     ${gamma_ncp}    embedding.embedding_dim=32 ${opt_params} &
wait
