import importlib
import subprocess
import sys
for module in ['kooplearn', 'datasets', 'matplotlib', 'ml-confs']: # !! Add here any additional module that you need to install on top of kooplearn
    try:
        importlib.import_module(module)
    except ImportError:
        if module == 'kooplearn':
            module = 'kooplearn[full]'
        # pip install -q {module}
        subprocess.check_call([sys.executable, "-m", "pip", "install", module])
import shutil
from pathlib import Path
import random
from datasets import DatasetDict, interleave_datasets, load_dataset

# main_path = Path(__file__).parent
# data_path = str(main_path / "__data__")
# noisy_data_path = str(main_path / "__data__Noisy")
# configs = ml_confs.from_file(main_path / "configs.yaml")

# print(noisy_data_path)
def make_indices(configs):
    """
    Make indices for the noisy dataset
    
    Returns:
    indices: dict
        A dictionary with the indices for the noisy dataset
    
    """
    # set the seed
    random.seed(configs.rng_seed)

    indices = {"train":[], "test":[]}
    for split in ["train", "test"]:
        ind = 0  # random.randint(0, 9)
        for i in range(configs[f"{split}_samples"]):
            if random.random() > 1 - configs.eta: # Add noise
                next_digit = random.randint(1,9) # Add a random digit. It means that the index of next digit must be different the previous index + 1
                indices[split].append(ind + next_digit)
                # new_dataset[ordered_MNIST[split]["label"][ind + next_digit]].append(ordered_MNIST[split]["image"][ind + next_digit])
                ind = ind + next_digit + 1
            else:
                indices[split].append(ind)
                # new_dataset[ordered_MNIST[split]["label"][ind]].append(ordered_MNIST[split]["image"][ind])
                ind += 1
    return indices
            
def make_noisy_dataset(configs, data_path, noisy_data_path):
    """
    Make the noisy dataset and save it to disk
    
    """
    # Data pipeline
    # print(configs.eta)
    MNIST = load_dataset("mnist", keep_in_memory= False)
    digit_ds = []
    for i in range(configs.classes):
        digit_ds.append(MNIST.filter(lambda example: example["label"] == i, keep_in_memory=False, num_proc=8))
    
    # print(len(digit_ds), digit_ds[0])

    ordered_MNIST = DatasetDict()
    Noisy_ordered_MNIST = DatasetDict()
    indices = make_indices(configs)
    # Order the digits in the dataset and select only a subset of the data
    for split in ["train", "test"]:
        ordered_MNIST[split] = interleave_datasets([ds[split] for ds in digit_ds], split=split, seed=configs.rng_seed)  
        Noisy_ordered_MNIST[split] = ordered_MNIST[split].select(indices=indices[split])
        ordered_MNIST[split] = ordered_MNIST[split].select(range(configs[f"{split}_samples"]))
    
    _tmp_ds = Noisy_ordered_MNIST["train"].train_test_split(test_size=configs.val_ratio, shuffle=False, seed=configs.rng_seed)
    Noisy_ordered_MNIST["train"] = _tmp_ds["train"]
    Noisy_ordered_MNIST["validation"] = _tmp_ds["test"]

    Noisy_ordered_MNIST.set_format(type="torch", columns=["image", "label"])
    Noisy_ordered_MNIST = Noisy_ordered_MNIST.map(
        lambda example: {"image": example["image"] / 255.0, "label": example["label"]},
        batched=True,
        keep_in_memory=False,
        num_proc=2,
    )
    Noisy_ordered_MNIST.save_to_disk(noisy_data_path)

    _tmp_ds = ordered_MNIST["train"].train_test_split(test_size=configs.val_ratio, shuffle=False, seed=configs.rng_seed)
    ordered_MNIST["train"] = _tmp_ds["train"]
    ordered_MNIST["validation"] = _tmp_ds["test"]

    ordered_MNIST.set_format(type="torch", columns=["image", "label"])
    ordered_MNIST = ordered_MNIST.map(
        lambda example: {"image": example["image"] / 255.0, "label": example["label"]},
        batched=True,
        keep_in_memory=False,
        num_proc=2,
    )

    ordered_MNIST.save_to_disk(data_path)
    # configs.to_file(str(data_path / "configs.yaml"))



def main(configs):
    main_path = Path(__file__).parent
    data_path = main_path / "__data__"
    noisy_data_path = main_path / "__data__Noisy"
    # Check if data_path exists, if not preprocess the data
    if not data_path.exists():
        print("Data directory not found, preprocessing data.")
        make_noisy_dataset(configs, data_path, noisy_data_path)
    else:
        # # Try to load the configs.yaml file and compare with the current one, if different, wipe the data_path and preprocess the data
        # _saved_configs = ml_confs.from_file(data_path / "configs.yaml")
        # configs_changed = False
        # for attr in ["train_samples", "test_samples", "classes", "val_ratio"]:
        #     if _saved_configs[attr] != configs[attr]:
        #         configs_changed = True
        # if configs_changed:
        #     print("Configs changed, preprocessing data.")
        #     # Delete the data_path and preprocess the data
        shutil.rmtree(data_path)
        shutil.rmtree(noisy_data_path)
        make_noisy_dataset(configs, data_path, noisy_data_path)
            
        # else:
        #     print("Data already preprocessed.")


if __name__ == "__main__":
    main()
