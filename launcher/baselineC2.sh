#!/bin/bash

# Function to handle termination signals
cleanup() {
    echo "Terminating all processes..."
    kill 0  # Sends SIGTERM to all processes in the current process group
    exit 0
}

# Trap signals (SIGINT, SIGTERM)
trap cleanup SIGINT SIGTERM


exp_label="C2"

exec_file="NCP/examples/symm_GMM.py"
hydra_kwargs="--multirun hydra.launcher.n_jobs=3"
opt_params="batch_size=2048 lr=5e-4 max_epochs=7000"
exp_params="exp_label=${exp_label} gmm.n_samples=20000 seed=0,1 truncated_op_bias=full_rank "
gamma_ncp="gamma=0.1"
python ${exec_file} ${hydra_kwargs} ${exp_params} device=0 model=NCP  regular_multiplicity=0    ${gamma_ncp} ${opt_params} &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=1 model=NCP  regular_multiplicity=2    ${gamma_ncp} ${opt_params} &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=2 model=ENCP regular_multiplicity=0    ${gamma_ncp} ${opt_params} &
python ${exec_file} ${hydra_kwargs} ${exp_params} device=3 model=ENCP regular_multiplicity=2    ${gamma_ncp} ${opt_params} &
#python ${exec_file} ${hydra_kwargs} ${exp_params} device=4 model=ENCP regular_multiplicity=    ${gamma_ncp} ${opt_params} &
python ${exec_file} --multirun hydra.launcher.n_jobs=1 ${exp_params} device=5 model=DRF regular_multiplicity=0  ${gamma_ncp} ${opt_params} &
python ${exec_file} --multirun hydra.launcher.n_jobs=1 ${exp_params} device=6 model=DRF regular_multiplicity=2  ${gamma_ncp} ${opt_params} &
#python ${exec_file} --multirun hydra.launcher.n_jobs=1 ${exp_params} device=7 model=DRF regular_multiplicity=10  ${gamma_ncp} ${opt_params} &
wait
