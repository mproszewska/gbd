import logging
import os
import pathlib
import random
import warnings

import graph_tool
import hydra
import torch

from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from src.models.extra_features import DummyExtraFeatures, ExtraFeatures

from gbd.graph_discrete_flow_model import GraphDiscreteFlowModel

warnings.filterwarnings("ignore", category=PossibleUserWarning)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    seed = random.randint(0, 10000)
    pl.seed_everything(seed)  # cfg.train.seed)
    dataset_config = cfg["dataset"]

    if dataset_config["name"] in [
        "sbm",
        "comm20",
        "planar",
        "tree",
    ]:
        from analysis.visualization import NonMolecularVisualization
        from datasets.spectre_dataset import (
            SpectreGraphDataModule,
            SpectreDatasetInfos,
        )
        from analysis.spectre_utils import (
            PlanarSamplingMetrics,
            SBMSamplingMetrics,
            Comm20SamplingMetrics,
            TreeSamplingMetrics,
        )

        datamodule = SpectreGraphDataModule(cfg)
        if dataset_config["name"] == "sbm":
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config["name"] == "comm20":
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        elif dataset_config["name"] == "planar":
            sampling_metrics = PlanarSamplingMetrics(datamodule)
        elif dataset_config["name"] == "tree":
            sampling_metrics = TreeSamplingMetrics(datamodule)
        else:
            raise NotImplementedError(
                f"Dataset {dataset_config['name']} not implemented"
            )

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)

        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        extra_features = ExtraFeatures(
            cfg.model.extra_features, cfg.model.rrwp_steps, dataset_info=dataset_infos,
        )
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

    elif dataset_config["name"] in [
        "qm9",
        "guacamol",
        "moses",
        "qm9_scaffold",
        "molgenbench",
        "moses_scaffold",
    ]:
        from src.metrics.molecular_metrics import (
            TrainMolecularMetrics,
            SamplingMolecularMetrics,
        )
        from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from src.models.extra_features_molecular import ExtraMolecularFeatures
        from src.analysis.visualization import MolecularVisualization

        if ("qm9" in dataset_config["name"]) and (
            "scaffold" not in dataset_config["name"]
        ):
            from datasets import qm9_dataset

            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            dataset_smiles = qm9_dataset.get_smiles(
                cfg=cfg,
                datamodule=datamodule,
                dataset_infos=dataset_infos,
                evaluate_datasets=False,
            )
        elif dataset_config["name"] == "qm9_scaffold":
            from gbd import qm9_scaffold_dataset
            from datasets import qm9_dataset

            datamodule = qm9_scaffold_dataset.QM9ScaffoldDataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            dataset_smiles = qm9_dataset.get_smiles(
                cfg=cfg,
                datamodule=datamodule,
                dataset_infos=dataset_infos,
                evaluate_datasets=False,
            )
        elif dataset_config["name"] == "guacamol":
            from datasets import guacamol_dataset

            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            dataset_smiles = guacamol_dataset.get_smiles(
                raw_dir=datamodule.train_dataset.raw_dir,
                filter_dataset=cfg.dataset.filter,
            )

        elif dataset_config.name == "moses":
            from datasets import moses_dataset

            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(
                datamodule, cfg, recompute_statistics=False
            )
            dataset_smiles = moses_dataset.get_smiles(
                raw_dir=datamodule.test_dataset.raw_dir,
                filter_dataset=cfg.dataset.filter,
            )
        elif dataset_config.name in ["moses_scaffold"]:
            from gbd.moses_scaffold_dataset import (
                MosesScaffoldDataModule,
                MosesScaffoldInfos,
            )
            from gbd.molgenbench_dataset import get_smiles

            datamodule = MosesScaffoldDataModule(cfg)
            dataset_infos = MosesScaffoldInfos(
                datamodule, cfg, recompute_statistics=False
            )
            dataset_smiles = get_smiles(
                cfg=cfg,
                datamodule=datamodule,
                dataset_infos=dataset_infos,
                evaluate_datasets=False,
            )
        elif dataset_config.name == "molgenbench":
            from gbd import molgenbench_dataset

            datamodule = molgenbench_dataset.MolGenBenchDataModule(cfg)
            dataset_infos = molgenbench_dataset.MolGenBenchInfos(
                datamodule=datamodule, cfg=cfg, recompute_statistics=False
            )
            dataset_smiles = molgenbench_dataset.get_smiles(
                cfg=cfg,
                datamodule=datamodule,
                dataset_infos=dataset_infos,
                evaluate_datasets=False,
            )
        else:
            raise ValueError("Dataset not implemented")

        if cfg.general.no_extra_feat:
            print("Dummy")
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()
        else:
            extra_features = ExtraFeatures(
                cfg.model.extra_features,
                cfg.model.rrwp_steps,
                dataset_info=dataset_infos,
            )
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

        # We do not evaluate novelty during training
        add_virtual_states = "absorbing" == cfg.model.transition
        sampling_metrics = SamplingMolecularMetrics(
            dataset_infos, dataset_smiles, cfg, add_virtual_states=add_virtual_states
        )
        visualization_tools = MolecularVisualization(
            cfg.dataset.remove_h, dataset_infos=dataset_infos
        )

    elif dataset_config["name"] == "tls":
        from datasets import tls_dataset
        from metrics.tls_metrics import TLSSamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = tls_dataset.TLSDataModule(cfg)
        dataset_infos = tls_dataset.TLSInfos(datamodule=datamodule)

        train_metrics = TrainAbstractMetricsDiscrete()
        extra_features = (
            ExtraFeatures(
                cfg.model.extra_features,
                cfg.model.rrwp_steps,
                dataset_info=dataset_infos,
            )
            if cfg.model.extra_features is not None
            else DummyExtraFeatures()
        )
        domain_features = DummyExtraFeatures()

        sampling_metrics = TLSSamplingMetrics(datamodule)

        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    dataset_infos.compute_reference_metrics(
        datamodule=datamodule, sampling_metrics=sampling_metrics,
    )

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
        "extra_features": extra_features,
        "domain_features": domain_features,
        "test_labels": (
            datamodule.test_labels
            if ("qm9" in cfg.dataset.name and cfg.general.conditional)
            else None
        ),
    }

    utils.create_folders(cfg)

    if cfg.general.test_only:
        if not cfg.general.conditional:
            if dataset_config["name"] in ["molgenbench", "moses_scaffold"]:
                dataset_infos.input_dims["y"] += 2
            else:
                dataset_infos.input_dims["y"] += 4

        if cfg.model.add_is_scaffold:
            dataset_infos.input_dims["X"] -= 2
            dataset_infos.input_dims["E"] -= 2
        elif dataset_config["name"] in ["molgenbench", "moses_scaffold"]:
            if cfg.model.transition == "absorbing":
                dataset_infos.input_dims["X"] -= 1
                dataset_infos.input_dims["E"] -= 1

    model = GraphDiscreteFlowModel(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename="{epoch}",
            save_top_k=-1,
            every_n_epochs=cfg.general.sample_every_val
            * cfg.general.check_val_every_n_epochs,
        )
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == "debug":
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=name == "debug",
        enable_progress_bar=False,
        callbacks=callbacks,
        log_every_n_steps=50 if name != "debug" else 1,
        logger=[],
    )
    import torch.distributed as dist

    dist.init_process_group(backend="nccl", init_method="env://")
    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
    else:
        # Start by evaluating test_only_path
        # datamodule.test_dataloader = datamodule.train_dataloader
        model = GraphDiscreteFlowModel.load_from_checkpoint(
            cfg.general.test_only,
            None,
            None,
            True,
            train_metrics=train_metrics,
            sampling_metrics=sampling_metrics,
            cfg=cfg,
            dataset_infos=dataset_infos,
            visualization_tools=visualization_tools,
        )

        logging.info(cfg)
        trainer.test(model, datamodule=datamodule)  # , ckpt_path=cfg.general.test_only)


if __name__ == "__main__":
    main()
    torch.distributed.destroy_process_group()
