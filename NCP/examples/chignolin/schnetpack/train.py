import ml_confs
from pathlib import Path
import schnetpack
from NCP.examples.chignolin.schnetpack.input_pipeline import TimeLaggedSampler
from NCP.examples.chignolin.schnetpack.model import GraphNCP
import torch
import lightning
from NCP.examples.chignolin.schnetpack.model import SchNet, SchNet_NCPOperator
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import random_split


def get_dataset(db_path, cutoff):
    data_path = db_path.parent
    db_name = db_path.name.split(".")[0]
    cache_path = str(data_path / "__tmp__")
    nb_list_transform = schnetpack.transform.CachedNeighborList(
        cache_path,
        schnetpack.transform.MatScipyNeighborList(cutoff=cutoff),
        keep_cache=True,
    )
    in_transforms = [
        schnetpack.transform.CastTo32(),
        schnetpack.transform.MatScipyNeighborList(cutoff=cutoff),
    ]
    return schnetpack.data.ASEAtomsData(str(db_path), transforms=in_transforms)


def main():
    main_path = Path(__file__).parent
    configs = ml_confs.from_file(Path.joinpath(main_path, 'configs.yaml'))
    configs.tabulate()

    # Seed everything
    lightning.pytorch.seed_everything(configs.seed)

    # Loading the database
    db_path = Path.joinpath(main_path, 'data/CLN025-0-protein_backbone.db')

    # Loading the dataset
    dataset = get_dataset(db_path, configs.cutoff)
    n_train = round(len(dataset) * 0.8)
    n_test = round(len(dataset) * 0.1)
    n_val = round(len(dataset) * 0.1)
    train_dataset = torch.utils.data.Subset(dataset, range(n_train))
    test_dataset = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
    val_dataset = torch.utils.data.Subset(dataset, range(n_train + n_val, n_train + n_val + n_test))
    batch_sampler_train = TimeLaggedSampler(train_dataset, batch_size=configs.batch_size, lagtime=configs.lagtime,
                                            shuffle=True)
    batch_sampler_val = TimeLaggedSampler(val_dataset, batch_size=configs.batch_size, lagtime=configs.lagtime,
                                          shuffle=True)
    dataloader_train = schnetpack.data.AtomsLoader(train_dataset, batch_sampler=batch_sampler_train, num_workers=12,
                                                   persistent_workers=True)
    dataloader_val = schnetpack.data.AtomsLoader(val_dataset, batch_sampler=batch_sampler_val, num_workers=12,
                                                 persistent_workers=True)

    optimizer = torch.optim.Adam
    optimizer_kwargs = {'lr': 1e-3}
    n_atoms = dataset[0][schnetpack.properties.n_atoms].item()

    NCP_model = SchNet_NCPOperator(U_operator=SchNet, U_operator_configs=configs)

    model = GraphNCP(
        NCP_model,
        configs,
        n_atoms,
        optimizer,
        optimizer_kwargs
    )

    ckpt_path = 'checkpoints/chignolin' + '_lagtime' + str(configs.lagtime)
    ckpt_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=ckpt_path,
        mode='min',
        save_top_k=5,
        save_last=True
    )

    trainer = lightning.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=100,
        log_every_n_steps=1,
        callbacks=[ckpt_callback],
        limit_val_batches=10,
        val_check_interval=100,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, dataloader_train, val_dataloaders=dataloader_val)

if __name__ == "__main__":
    main()
