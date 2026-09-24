import os
import sys
import random
import time
import wandb

import ast
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import QED, RDConfig
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch, Data
from torch.distributions.categorical import Categorical

from src import utils
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from src.flow_matching.noise_distribution import NoiseDistribution
from src.flow_matching.time_distorter import TimeDistorter
from src.flow_matching.rate_matrix import RateMatrixDesigner
from src.flow_matching.utils import p_xt_g_x1
from src.flow_matching import flow_matching_utils
from src.metrics.train_metrics import TrainLossDiscrete
from src.models.transformer_model import GraphTransformer

from gbd.molgenbench_dataset import load_eval_set_molgenbench, smiles_to_scaffold_data

try:
    import psi4
except ModuleNotFoundError:
    print("PSI4 not found")

from tdc import Oracle, Evaluator

sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
sys.path.append(sa_path)
import sascorer

MOLGENBENCH_GEN_SAMPLES = 200
SAFE_DATA_PATH = "data/safe-drugs.csv"
SAFE_GEN_SAMPLES = 100

def save_smiles_to_sdf(smiles_list, filename):

    writer = Chem.SDWriter(filename)

    for idx, smi in enumerate(smiles_list):
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Skipping invalid SMILES: {smi}")
            continue

        # Set unique _Name field
        mol.SetProp("_Name", str(idx))

        writer.write(mol)

    writer.close()
    print(f"SDF file saved as: {filename}")


def sample_fragment_masks(fragments_X, fragments_E):

    B, N = fragments_X.shape
    device = fragments_X.device

    n_fragments = torch.maximum(fragments_X.amax(dim=1), fragments_E.amax(dim=(1, 2)),)

    # Sample number to keep
    n_selected = (torch.rand(B, device=device) * n_fragments).long() + 1

    K = n_fragments.max()

    # Random scores for each fragment
    scores = torch.rand(B, K, device=device)

    fragment_ids = torch.arange(1, K + 1, device=device,)[None, :]

    # Invalid fragment IDs can never be selected
    scores = scores.masked_fill(fragment_ids > n_fragments[:, None], -torch.inf,)

    # Select exactly n_selected fragments
    threshold = torch.topk(scores, k=K, dim=1, largest=True, sorted=False,).values[
        torch.arange(B, device=device), n_selected - 1,
    ]

    selected = scores >= threshold[:, None]

    # Handle IDs > n_fragments (only relevant because K is batch max)
    fragment_ids = torch.arange(1, K + 1, device=device)[None, :]

    selected &= fragment_ids <= n_fragments[:, None]

    # Add fragment 0
    selected = torch.cat(
        [torch.zeros(B, 1, dtype=torch.bool, device=device), selected,], dim=1,
    )

    mask_X = selected.gather(1, fragments_X)

    mask_E = selected.gather(1, fragments_E.flatten(1)).reshape(B, N, N)

    mask_X &= fragments_X != 0
    mask_E &= fragments_E != 0
    return mask_X, mask_E, n_selected


class GraphDiscreteFlowModel(pl.LightningModule):
    def __init__(
        self,
        cfg,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
        test_labels=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.name = f"{cfg.dataset.name}_{cfg.general.name}"
        self.model_dtype = torch.float32
        self.conditional = cfg.general.conditional
        self.test_labels = test_labels
        self.mask_output = cfg.train.mask_output

        # number of steps used for sampling
        self.sample_T = cfg.sample.sample_steps

        self.ignore_y = cfg.model.ignore_y

        self.input_dims = dataset_infos.input_dims
        self.output_dims = dataset_infos.output_dims
        self.dataset_info = dataset_infos
        self.node_dist = dataset_infos.nodes_dist
        print("max num nodes: ", len(self.node_dist.prob) - 1)
        print("min num nodes: ", torch.where(self.node_dist.prob > 0)[0][0].item())

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.noise_dist = NoiseDistribution(cfg.model.transition, dataset_infos)
        self.limit_dist = self.noise_dist.get_limit_dist()

        # add virtual class when absorbing state refers to a new class
        self.noise_dist.update_input_output_dims(self.input_dims)
        self.noise_dist.update_dataset_infos(self.dataset_info)

        if (
            hasattr(self.dataset_info, "has_scaffold")
            and self.dataset_info.has_scaffold
            and self.cfg.model.add_is_scaffold
        ):
            self.input_dims["X"] += 1
            self.input_dims["E"] += 1
        if not self.conditional:
            self.input_dims["y"] -= 2

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=self.input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train,)

        self.save_hyperparameters(
            ignore=[
                "train_metrics",
                "sampling_metrics",
                "dataset_info.eval_set",
                "input_dims",
            ],
        )

        # logging
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.val_counter = 0
        self.adapt_counter = 0

        # time distortor for both training and sampling steps
        self.time_distorter = TimeDistorter(
            train_distortion=cfg.train.time_distortion,
            sample_distortion=cfg.sample.time_distortion,
            alpha=1,
            beta=1,
        )

        # rate matrix designer
        self.rate_matrix_designer = RateMatrixDesigner(
            rdb=self.cfg.sample.rdb,
            rdb_crit=self.cfg.sample.rdb_crit,
            eta=self.cfg.sample.eta,
            omega=self.cfg.sample.omega,
            limit_dist=self.limit_dist,
        )

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        if self.conditional:
            if torch.rand(1) < 0.1:
                data.y = torch.ones_like(data.y, device=self.device) * -1
        else:
            data.y = -torch.ones(len(data.y), 0, device=self.device)

        if self.ignore_y:
            data.y = -torch.ones_like(data.y)

        if self.cfg.model.keep_scaffold_fixed:
            edge_attr = torch.cat(
                [
                    data.edge_attr,
                    data.scaffold_edge_mask.unsqueeze(-1),
                    data.bond_to_frag.unsqueeze(-1),
                ],
                dim=-1,
            )
            x = torch.cat(
                [
                    data.x,
                    data.scaffold_node_mask.unsqueeze(-1),
                    data.atom_to_frag.unsqueeze(-1),
                ],
                dim=-1,
            )
            dense_data, node_mask = utils.to_dense(
                x, data.edge_index, edge_attr, data.batch,
            )
            dense_data = dense_data.mask(node_mask)
            X, E = dense_data.X, dense_data.E
            scaffold_X_mask = X[:, :, -2].bool()
            scaffold_E_mask = E[:, :, :, -2].bool()
            fragments_X = X[:, :, -1].long()
            fragments_E = E[:, :, :, -1].long()
            X = X[:, :, :-2]
            E = E[:, :, :, :-2]
            if torch.rand(1) < self.cfg.model.skip_scaffold_prob:
                scaffold_X_mask = torch.zeros_like(scaffold_X_mask)
                scaffold_E_mask = torch.zeros_like(scaffold_E_mask)
                subgraph_X_mask, subgraph_E_mask = scaffold_X_mask, scaffold_E_mask
            elif torch.rand(1) < self.cfg.model.sample_fragments_prob:
                subgraph_X_mask, subgraph_E_mask, n_selected = sample_fragment_masks(
                    fragments_X, fragments_E
                )
            else:
                subgraph_X_mask, subgraph_E_mask = scaffold_X_mask, scaffold_E_mask
            noisy_data = self.apply_noise(
                X, E, data.y, node_mask, X_mask=subgraph_X_mask, E_mask=subgraph_E_mask
            )

        else:
            dense_data, node_mask = utils.to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch,
            )

            dense_data = dense_data.mask(node_mask)
            X, E = dense_data.X, dense_data.E
            noisy_data = self.apply_noise(X, E, data.y, node_mask)
            scaffold_X_mask, scaffold_E_mask = None, None
            subgraph_X_mask, subgraph_E_mask = None, None

        extra_data = self.compute_extra_data(
            noisy_data, scaffold_X_mask, scaffold_E_mask
        )
        pred = self.forward(noisy_data, extra_data, node_mask)

        if self.mask_output:
            pred.X[subgraph_X_mask] = X[subgraph_X_mask]
            pred.E[subgraph_E_mask] = E[subgraph_E_mask]

        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=data.y,
            log=i % self.log_every_steps == 0,
        )

        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            log=i % self.log_every_steps == 0,
        )

        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print(
            "Size of the input features",
            self.input_dims["X"],
            self.input_dims["E"],
            self.input_dims["y"],
        )
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
            f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
            f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
            f" -- {time.time() - self.start_epoch_time:.1f}s "
        )
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}"
        )
        if wandb.run:
            wandb.log({"epoch": self.current_epoch}, commit=False)

    def on_validation_epoch_start(self) -> None:
        print("Starting validation...")
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        return

    def on_validation_epoch_end(self) -> None:
        self.val_counter += 1
        if (
            self.val_counter % self.cfg.general.sample_every_val == 0
        ) and self.val_counter > 1:
            print("Starting to sample")

            samples, labels = self.sample(
                is_test=False,
                save_samples=False,
                save_visualization=False,
            )
            filename = os.path.join(
                os.getcwd(),
                f"val_epoch{self.current_epoch}_res_{self.cfg.sample.eta}_{self.cfg.sample.rdb}.pt",
            )
            torch.save(samples, filename)
            self.print(f"Samples saved to {filename}")
            to_log = self.evaluate_samples(
                samples=samples, labels=labels, is_test=False
            )

            # Store results
            filename = os.path.join(
                os.getcwd(),
                f"val_epoch{self.current_epoch}_res_{self.cfg.sample.eta}_{self.cfg.sample.rdb}.txt",
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

        self.print("Finished validation.")

    def molgenbench_eval(self):
        os.makedirs(os.path.join(os.getcwd(), "molgenbench_eval"), exist_ok=True)

        bond_decoder = (
            self.dataset_info.bond_decoder
            if hasattr(self.dataset_info, "bond_decoder")
            else None
        )
        pre_filter = (
            self.dataset_info.pre_filter
            if hasattr(self.dataset_info, "pre_filter")
            else None
        )
        pre_transform = (
            self.dataset_info.pre_transform
            if hasattr(self.dataset_info, "pre_transform")
            else None
        )
        data_list, eval_codes = load_eval_set_molgenbench(
            "eval",
            self.dataset_info.atom_decoder,
            bond_decoder=bond_decoder,
            pre_filter=pre_filter,
            pre_transform=pre_transform,
        )
        if self.cfg.general.test_only and ("epoch" in self.cfg.general.test_only):
            current_epoch = self.cfg.general.test_only.split("epoch=")[-1].replace(
                ".ckpt", ""
            )
        else:
            current_epoch = self.current_epoch

        suffix = f"{self.cfg.sample.time_distortion}_T{self.cfg.sample.sample_steps}_eta{self.cfg.sample.eta}_omega{self.cfg.sample.omega}"

        if self.cfg.general.scaffold_idx is not None:
            if "-" in self.cfg.general.scaffold_idx:
                a, b = self.cfg.general.scaffold_idx.split("-")
            else:
                a = b = self.cfg.general.scaffold_idx
            start, end = int(a), int(b)
        else:
            start, end = 0, 600

        for scaffold_idx, code in enumerate(tqdm(eval_codes, total=len(eval_codes))):
            if (scaffold_idx < start) or (scaffold_idx > end):
                continue

            filename = os.path.join(
                os.getcwd(),
                "molgenbench_eval",
                f"{code}_defog_epoch{current_epoch}_{suffix}.sdf",
            )
            if os.path.exists(filename):
                print(f"Skipping. {filename} exists.")
                continue

            bs = 2 * self.cfg.train.batch_size
            scaffolds = [
                data_list[scaffold_idx][i % len(data_list[scaffold_idx])].to(
                    self.device
                )
                for i in range(bs)
            ]
            scaffolds = [data_to_XE_data(scaffold) for scaffold in scaffolds]
            assert scaffolds[0].idx == scaffold_idx, (scaffolds[0].idx, scaffold_idx)

            samples, labels = self.sample(
                is_test=False,
                save_samples=False,
                save_visualization=False,
                fragment_data=scaffolds,
                num_samples=MOLGENBENCH_GEN_SAMPLES,
            )

            mols_smiles, disconnected, invalid = samples2smiles(
                samples, self.dataset_info.atom_decoder
            )
            print(
                f"scaffold_idx={scaffold_idx}, code={code} mols={len(mols_smiles)} disconnected={disconnected} invalid={invalid}"
            )

            save_smiles_to_sdf(mols_smiles, filename)
            del samples, labels, mols_smiles

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.sampling_metrics.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        return

    def on_test_epoch_end(self, dataloader=None) -> None:
        if dataloader is not None:
            self.sampling_metrics.reset()
            for i, data in enumerate(dataloader):
                self.test_step(data, i)

        if self.cfg.sample.search:
            print("Starting sampling optimization...")
            self.search_hyperparameters()

        elif self.cfg.sample.eval_task in ["molgenbench", "safe", "de_novo"]:
            if self.cfg.sample.eval_task == "molgenbench":
                self.molgenbench_eval()
            elif self.cfg.sample.eval_task == "safe":
                self.safe_eval()
            elif self.cfg.sample.eval_task == "de_novo":
                self.de_novo_eval()
            else:
                assert False, self.cfg.sample.eval_task

        else:
            if "epoch" in self.cfg.general.test_only:
                current_epoch = self.cfg.general.test_only.split("epoch=")[-1].replace(
                    ".ckpt", ""
                )
            else:
                current_epoch = self.current_epoch
            print("Starting to sample")
            samples, labels = self.sample(
                is_test=True,
                save_samples=self.cfg.general.save_samples,
                save_visualization=False,
            )
            to_log = self.evaluate_samples(samples=samples, labels=labels, is_test=True)

            # Store results
            filename = os.path.join(
                os.getcwd(),
                f"test_epoch{current_epoch}_{self.cfg.general.final_model_samples_to_generate}_{self.cfg.sample.time_distortion}_{self.cfg.sample.sample_steps}_{self.cfg.sample.eta}_{self.cfg.sample.omega}.txt",
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

            print("Finished testing.")

    def sample(
        self,
        is_test,
        save_samples,
        save_visualization,
        fragment_data=None,
        num_samples=None,
        num_nodes=None,
    ):
        try:
            trainer_attached = self.trainer is not None
        except RuntimeError:
            trainer_attached = False
            self.print = print

        # Load generated samples if they exist
        if self.cfg.general.generated_path:
            self.print("Loading generated samples...")
            with open(self.cfg.general.generated_path, "rb") as f:
                samples = pickle.load(f)
            # Set labels to None
            labels = [None] * len(samples)
            return samples, labels

        # Otherwise, generate new samples
        if is_test:
            samples_to_generate = (
                self.cfg.general.final_model_samples_to_generate
                * self.cfg.general.num_sample_fold
            )
            samples_left_to_generate = (
                self.cfg.general.final_model_samples_to_generate
                * self.cfg.general.num_sample_fold
            )
            samples_left_to_save = self.cfg.general.final_model_samples_to_save
            chains_left_to_save = self.cfg.general.final_model_chains_to_save

        else:
            samples_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

        if num_samples is not None:
            samples_to_generate = num_samples
            samples_left_to_generate = num_samples
        samples = []
        labels = []
        graph_id = 0
        while samples_left_to_generate > 0:
            self.print(
                f"Samples left to generate: {samples_left_to_generate}/"
                f"{samples_to_generate}",
                end="",
                flush=True,
            )

            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            num_chain_steps = min(self.number_chain_steps, self.sample_T)
            cur_samples, cur_labels = self.sample_batch(
                graph_id,
                to_generate,
                num_nodes=num_nodes,
                save_final=to_save,
                keep_chain=chains_save,
                number_chain_steps=num_chain_steps,
                save_visualization=save_visualization,
                fragment_data=fragment_data,
            )
            samples.extend(cur_samples)
            labels.extend(cur_labels)

            graph_id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        if save_samples:
            self.print("Saving the generated graphs")

            # saving in txt version
            filename = "graphs.txt"
            with open(filename, "w") as f:
                for item in samples:
                    f.write(f"N={item[0].shape[0]}\n")
                    atoms = item[0].tolist()
                    f.write("X: \n")
                    for at in atoms:
                        f.write(f"{at} ")
                    f.write("\n")
                    f.write("E: \n")
                    for bond_list in item[1]:
                        for bond in bond_list:
                            f.write(f"{bond} ")
                        f.write("\n")
                    f.write("\n")

            # saving in pkl version
            with open(f"generated_samples_rank{self.local_rank}.pkl", "wb") as f:
                pickle.dump(samples, f)

            print("Generated graphs saved.")

        return samples, labels

    def evaluate_samples(
        self, samples, labels, is_test, save_filename="",
    ):
        print("Computing sampling metrics...")

        to_log = {}
        samples_to_evaluate = self.cfg.general.final_model_samples_to_generate
        if is_test:
            for i in range(self.cfg.general.num_sample_fold):
                cur_samples = samples[
                    i * samples_to_evaluate : (i + 1) * samples_to_evaluate
                ]
                cur_labels = labels[
                    i * samples_to_evaluate : (i + 1) * samples_to_evaluate
                ]

                cur_to_log = self.sampling_metrics.forward(
                    cur_samples,
                    ref_metrics=self.dataset_info.ref_metrics,
                    name=f"self.name_{i}",
                    current_epoch=self.current_epoch,
                    val_counter=-1,
                    test=is_test,
                    local_rank=self.local_rank,
                    labels=cur_labels
                    if (
                        (self.conditional and not self.ignore_y)
                    )
                    else None,
                )

                if i == 0:
                    to_log = {i: [cur_to_log[i]] for i in cur_to_log}
                else:
                    to_log = {i: to_log[i] + [cur_to_log[i]] for i in cur_to_log}

                filename = os.path.join(
                    os.getcwd(),
                    f"epoch{self.current_epoch}_res_fold{i}_{save_filename}.txt",
                )
                with open(filename, "w") as file:
                    for key, value in cur_to_log.items():
                        file.write(f"{key}: {value}\n")

            to_log = {
                i: (np.array(to_log[i]).mean(), np.array(to_log[i]).std())
                for i in to_log
            }
        else:
            if not is_test:
                labels = None
            to_log = self.sampling_metrics.forward(
                samples,
                ref_metrics=self.dataset_info.ref_metrics,
                name=self.cfg.general.name,
                current_epoch=self.current_epoch,
                val_counter=-1,
                test=is_test,
                local_rank=self.local_rank,
                labels=labels
                if (self.conditional and not self.ignore_y)
                else None,
            )

        return to_log

    def apply_noise(self, X, E, y, node_mask, t=None, X_mask=None, E_mask=None):
        """Sample noise and apply it to the data."""

        # Sample a timestep t.
        bs = X.size(0)
        if t is None:
            t_float = self.time_distorter.train_ft(bs, self.device)
        else:
            t_float = t

        # sample random step
        X_1_label = torch.argmax(X, dim=-1)
        E_1_label = torch.argmax(E, dim=-1)
        prob_X_t, prob_E_t = p_xt_g_x1(
            X1=X_1_label, E1=E_1_label, t=t_float, limit_dist=self.limit_dist
        )
        if X_mask is not None:
            if prob_X_t.shape[-1] != X.shape[-1]:
                X_padded = F.pad(X, (0, 1))
                E_padded = F.pad(E, (0, 1))
                prob_X_t[X_mask] = X_padded[X_mask]
                prob_E_t[E_mask] = E_padded[E_mask]
            else:
                prob_X_t[X_mask] = X[X_mask]
                prob_E_t[E_mask] = E[E_mask]

        # step 4 - sample noised data
        sampled_t = flow_matching_utils.sample_discrete_features(
            probX=prob_X_t, probE=prob_E_t, node_mask=node_mask
        )
        noise_dims = self.noise_dist.get_noise_dims()
        X_t = F.one_hot(sampled_t.X, num_classes=noise_dims["X"])
        E_t = F.one_hot(sampled_t.E, num_classes=noise_dims["E"])

        # step 5 - create the PlaceHolder
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {
            "t": t_float,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }

        return noisy_data

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        num_nodes=None,
        save_visualization: bool = False,
        fragment_data=None,
    ):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(
                batch_size, device=self.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes

        n_max = torch.max(n_nodes).item()

        # Build the masks
        arange = (
            torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)

        if fragment_data is not None:
            if isinstance(fragment_data, list):
                scaffolds = [random.choice(fragment_data) for _ in range(batch_size)]
                n_nodes = torch.tensor([len(s.X) for s in scaffolds]).to(self.device)
                n_max = torch.max(n_nodes).item()
                arange = (
                    torch.arange(n_max, device=self.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                node_mask = arange < n_nodes.unsqueeze(1)
                X = torch.zeros(
                    len(scaffolds), n_max, scaffolds[0].X.shape[-1], device=self.device
                )
                E = torch.zeros(
                    len(scaffolds),
                    n_max,
                    n_max,
                    scaffolds[0].E.shape[-1],
                    device=self.device,
                )
                scaffold_X_mask = torch.zeros_like(X[:, :, 0])
                scaffold_E_mask = torch.zeros_like(E[:, :, :, 0])
                y = torch.stack([s.y for s in scaffolds])
                for i, s in enumerate(scaffolds):
                    n = len(s.X)
                    X[i, :n] = s.X
                    E[i, :n, :n] = s.E
                    scaffold_X_mask[i, :n] = s.scaffold_X_mask
                    scaffold_E_mask[i, :n, :n] = s.scaffold_E_mask
                data = Data(
                    X=X,
                    E=E,
                    scaffold_X_mask=scaffold_X_mask.bool(),
                    scaffold_E_mask=scaffold_E_mask.bool(),
                    y=y,
                ).to(self.device)
            else:
                scaffolds = [fragment_data.clone()] * batch_size

                data = Data(
                    X=torch.stack([s.X for s in scaffolds]),
                    E=torch.stack([s.E for s in scaffolds]),
                    scaffold_X_mask=torch.stack([s.scaffold_X_mask for s in scaffolds]),
                    scaffold_E_mask=torch.stack([s.scaffold_E_mask for s in scaffolds]),
                    y=torch.stack([s.y for s in scaffolds]),
                ).to(self.device)

            data.y = (
                torch.ones(len(data.X), 0, device=self.device, dtype=torch.float) * -1
            )
            scaffold_X, scaffold_E = data.X, data.E
            scaffold_X_mask, scaffold_E_mask = (
                data.scaffold_X_mask,
                data.scaffold_E_mask,
            )

            z_T = flow_matching_utils.sample_discrete_feature_noise(
                limit_dist=self.noise_dist.get_limit_dist(), node_mask=node_mask
            )

            if scaffold_X.shape[-1] != z_T.X.shape[-1]:  # absorbing state
                scaffold_X_pad = F.pad(scaffold_X, (0, 1))
                scaffold_E_pad = F.pad(scaffold_E, (0, 1))

                z_T.X[scaffold_X_mask] = scaffold_X_pad[scaffold_X_mask]
                z_T.E[scaffold_E_mask] = scaffold_E_pad[scaffold_E_mask]
            else:
                scaffold_X_pad, scaffold_E_pad = None, None
                z_T.X[scaffold_X_mask] = scaffold_X[scaffold_X_mask]
                z_T.E[scaffold_E_mask] = scaffold_E[scaffold_E_mask]

        else:
            z_T = flow_matching_utils.sample_discrete_feature_noise(
                limit_dist=self.noise_dist.get_limit_dist(), node_mask=node_mask
            )
            scaffold_X, scaffold_E = None, None
            scaffold_X_mask, scaffold_E_mask = None, None
            scaffold_X_pad, scaffold_E_pad = None, None

        X, E, y = z_T.X.float(), z_T.E.float(), z_T.y.float()
        # Init chain storing variables
        assert (E == torch.transpose(E, 1, 2)).all()
        chain_X_size = torch.Size((number_chain_steps + 1, keep_chain, X.size(1)))
        chain_E_size = torch.Size(
            (number_chain_steps + 1, keep_chain, E.size(1), E.size(2))
        )
        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)
        chain_times = torch.zeros((number_chain_steps + 1, keep_chain))
        chain_time_unit = 1 / number_chain_steps

        # Store initial graph
        if keep_chain > 0:
            sampled_initial = z_T.mask(node_mask, collapse=True)
            chain_X[0] = sampled_initial.X[:keep_chain]
            chain_E[0] = sampled_initial.E[:keep_chain]
            chain_times[0] = torch.zeros((keep_chain))

        for t_int in tqdm(range(0, self.cfg.sample.sample_steps)):
            # this state
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
            t_norm = t_array / (self.cfg.sample.sample_steps)
            if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = t_array + 1
            s_norm = s_array / (self.cfg.sample.sample_steps)

            # using round for precision
            write_index = int(np.ceil(np.round(s_norm[0].item() / chain_time_unit, 6)))

            # Distort time
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.cfg.sample.time_distortion
            )
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.cfg.sample.time_distortion
            )

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                t_norm,
                s_norm,
                X,
                E,
                y,
                node_mask,
                scaffold_X_mask=scaffold_X_mask,
                scaffold_E_mask=scaffold_E_mask,
            )

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            if scaffold_X_mask is not None:
                if scaffold_X_pad is not None:
                    X[scaffold_X_mask] = scaffold_X_pad[scaffold_X_mask]
                    E[scaffold_E_mask] = scaffold_E_pad[scaffold_E_mask]
                else:
                    X[scaffold_X_mask] = scaffold_X[scaffold_X_mask]
                    E[scaffold_E_mask] = scaffold_E[scaffold_E_mask]
                sampled_s.X, sampled_s.E = X, E

            # Save the first keep_chain graphs
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]
            chain_times[write_index] = s_norm.flatten()[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:

            # Repeat last frame 10x to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            chain_times = torch.cat(
                [chain_times, chain_times[-1:].repeat(10, 1)], dim=0
            )
            assert chain_X.size(0) == (number_chain_steps + 1 + 10)

        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)

        chain_X, chain_E, _ = self.noise_dist.ignore_virtual_classes(
            chain_X, chain_E, y
        )

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            # Visualize chains
            self.print("Visualizing chains...")
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)  # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(
                    current_path,
                    f"chains/{self.cfg.general.name}/"
                    f"epoch{self.current_epoch}/"
                    f"chains/molecule_{batch_id + i}",
                )
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(
                        result_path,
                        chain_X[:, i, :].numpy(),
                        chain_E[:, i, :].numpy(),
                        chain_times[:, i].numpy(),
                    )
                self.print(
                    "\r{}/{} complete".format(i + 1, num_molecules), end="", flush=True
                )
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list

    def compute_step_probs(self, R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e):
        step_probs_X = R_t_X * dt  # type: ignore # (B, D, S)
        step_probs_E = R_t_E * dt  # (B, D, S)

        # Calculate the on-diagnoal step probabilities
        # 1) Zero out the diagonal entries
        # assert (E_t.argmax(-1) < 4).all()
        step_probs_X.scatter_(-1, X_t.argmax(-1)[:, :, None], 0.0)
        step_probs_E.scatter_(-1, E_t.argmax(-1)[:, :, :, None], 0.0)

        # 2) Calculate the diagonal entries such that the probability row sums to 1
        step_probs_X.scatter_(
            -1,
            X_t.argmax(-1)[:, :, None],
            (1.0 - step_probs_X.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs_E.scatter_(
            -1,
            E_t.argmax(-1)[:, :, :, None],
            (1.0 - step_probs_E.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )

        # step 2 - merge to the original formulation
        prob_X = step_probs_X.clone()
        prob_E = step_probs_E.clone()

        return prob_X, prob_E

    def sample_p_zs_given_zt(
        self,
        t,
        s,
        X_t,
        E_t,
        y_t,
        node_mask,
        target_y=None,
        scaffold_X_mask=None,
        scaffold_E_mask=None,
        # , condition
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        bs, n, dx = X_t.shape
        _, _, _, de = E_t.shape
        dt = (s - t)[0]

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }

        extra_data = self.compute_extra_data(
            noisy_data, scaffold_X_mask, scaffold_E_mask
        )
        pred = self.forward(noisy_data, extra_data, node_mask)
        # Normalize predictions,
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
        limit_x = self.limit_dist.X
        limit_e = self.limit_dist.E

        G_1_pred = pred_X, pred_E
        G_t = X_t, E_t

        R_t_X, R_t_E = self.rate_matrix_designer.compute_graph_rate_matrix(
            t, node_mask, G_t, G_1_pred,
        )

        prob_X, prob_E = self.compute_step_probs(
            R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e
        )

        if s[0] == 1.0:
            prob_X, prob_E = pred_X, pred_E

        sampled_s = flow_matching_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask
        )

        X_s = F.one_hot(sampled_s.X, num_classes=len(limit_x)).float()
        E_s = F.one_hot(sampled_s.E, num_classes=len(limit_e)).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        if self.conditional:
            y_to_save = y_t
        else:
            y_to_save = torch.zeros([y_t.shape[0], 0], device=self.device)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)

        out_one_hot = out_one_hot.mask(node_mask).type_as(y_t)
        out_discrete = out_discrete.mask(node_mask, collapse=True).type_as(y_t)
        return out_one_hot, out_discrete

    def compute_extra_data(
        self, noisy_data, scaffold_X_mask=None, scaffold_E_mask=None
    ):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""

        extra_features = self.extra_features(noisy_data)

        # one additional category is added for the absorbing transition
        X, E, y = self.noise_dist.ignore_virtual_classes(
            noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]
        )
        noisy_data_to_mol_feat = noisy_data.copy()
        noisy_data_to_mol_feat["X_t"] = X
        noisy_data_to_mol_feat["E_t"] = E
        noisy_data_to_mol_feat["y_t"] = y
        extra_molecular_features = self.domain_features(noisy_data_to_mol_feat)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data["t"]
        extra_y = torch.cat((extra_y, t), dim=1)

        if self.cfg.model.add_is_scaffold:
            extra_X = torch.cat([extra_X, scaffold_X_mask.unsqueeze(-1)], dim=-1)
            extra_E = torch.cat([extra_E, scaffold_E_mask.unsqueeze(-1)], dim=-1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def de_novo_eval(self):
        if self.cfg.general.test_only and ("epoch" in self.cfg.general.test_only):
            current_epoch = self.cfg.general.test_only.split("epoch=")[-1].replace(
                ".ckpt", ""
            )
        else:
            current_epoch = self.current_epoch
        os.makedirs(os.path.join(os.getcwd(), "denovo_eval"), exist_ok=True)
        filename = os.path.join(
            os.getcwd(), "denovo_eval", f"defog_epoch{current_epoch}.txt"
        )
        samples, labels = self.sample(
            is_test=False,
            save_samples=False,
            save_visualization=False,
            fragment_data=None,
            num_samples=1000,
        )
        smiles, disconnected, invalid = samples2smiles(
            samples, self.dataset_info.atom_decoder
        )
        scores = eval_smiles(smiles)
        print_scores(scores, filename)

    def safe_eval(self):
        if self.cfg.general.test_only and ("epoch" in self.cfg.general.test_only):
            current_epoch = self.cfg.general.test_only.split("epoch=")[-1].replace(
                ".ckpt", ""
            )
        else:
            current_epoch = self.current_epoch

        safe_data = pd.read_csv(SAFE_DATA_PATH)
        tasks = [
            "linker_generation",
            "scaffold_morphing",
            "motif_extension",
            "scaffold_decoration",
            "superstructure_generation",
        ]

        for task in tasks:
            scores = {
                "Validity": [],
                "Uniqueness": [],
                "Diversity": [],
                "QED": [],
                "SA": [],
            }
            for i, (s, fragment) in enumerate(
                zip(safe_data["smiles"], safe_data[task])
            ):
                if task == "linker_generation":
                    fragment = ast.literal_eval(fragment)
                    fragment = f"{fragment[0]}.{fragment[-1]}"

                if hasattr(self.dataset_info, "bond_decoder"):
                    bond_decoder = self.dataset_info.bond_decoder
                else:
                    bond_decoder = None
                data = fragment_to_data(
                    s, fragment, self.dataset_info.atom_decoder, bond_decoder
                )
                if data is None:
                    print("Invalid fragment", fragment)
                    continue
                data = data_to_XE_data(data)

                num_nodes = len(data.X)
                samples, labels = self.sample(
                    is_test=False,
                    save_samples=False,
                    save_visualization=False,
                    fragment_data=data,
                    num_samples=100,
                    num_nodes=num_nodes,
                )
                smiles, disconnected, invalid = samples2smiles(
                    samples, self.dataset_info.atom_decoder
                )
                score = eval_smiles(smiles)
               
                scores["Validity"].append(score["Validity"])
                scores["Uniqueness"].append(score["Uniqueness"])
                if score["Diversity"] is not None:
                    scores["Diversity"].append(score["Diversity"])
                    scores["QED"].append(score["QED"])
                    scores["SA"].append(score["SA"])
                print(task, i, score)
            print(task)

            os.makedirs(os.path.join(os.getcwd(), f"safe_eval_{task}"), exist_ok=True)
            suffix = f"{self.cfg.sample.time_distortion}_T{self.cfg.sample.sample_steps}_eta{self.cfg.sample.eta}_omega{self.cfg.sample.omega}"
            filename = os.path.join(
                os.getcwd(),
                f"safe_eval_{task}",
                f"defog_epoch{current_epoch}_{suffix}.txt",
            )
            print_scores(scores, filename)



def samples2smiles(samples, atom_decoder):
    smiles = []
    disconnected, invalid = 0, 0
    for l, molecule in enumerate(samples):
        mol = build_molecule_with_partial_charges(
            molecule[0],
            molecule[1],
            [atom_decoder[i] for i in range(len(atom_decoder))],
        )
        smile = mol2smiles(mol)
        if smile is not None:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if len(mol_frags) > 1:
                print("Disconnected molecule", mol, mol_frags)
                disconnected += 1
                smiles.append(None)
            else:
                smiles.append(smile)
        else:
            print("Invalid molecule obtained.")
            invalid += 1
            smiles.append(None)

    return smiles, disconnected, invalid


def print_scores(scores, filename=None):
    if not isinstance(scores["Validity"], list):
        scores = {k: [v] for k, v in scores.items()}
    print(f"Validity,{np.mean(scores['Validity'])}")
    print(f"Uniqueness,{np.mean(scores['Uniqueness'])}")
    print(f"Diversity,{np.mean(scores['Diversity'])}")
    print(f"QED,{np.mean(scores['QED'])}")
    print(f"SA,{np.mean(scores['SA'])}")
    if filename is not None:
        with open(filename, "a") as f:
            f.write(f"Validity,{np.mean(scores['Validity'])}\n")
            f.write(f"Uniqueness,{np.mean(scores['Uniqueness'])}\n")
            f.write(f"Diversity,{np.mean(scores['Diversity'])}\n")
            f.write(f"QED,{np.mean(scores['QED'])}\n")
            f.write(f"SA,{np.mean(scores['SA'])}\n")
        print(f"Saved to {filename}")


def data_to_XE_data(data):
    edge_attr = torch.cat(
        [data.edge_attr, data.scaffold_edge_mask.unsqueeze(-1)], dim=-1
    )
    x = torch.cat([data.x, data.scaffold_node_mask.unsqueeze(-1)], dim=-1)
    batch = torch.zeros(len(x), device=x.device).long()
    dense_data, node_mask = utils.to_dense(x, data.edge_index, edge_attr, batch=batch)
    X, E = dense_data.X.squeeze(0), dense_data.E.squeeze(0)
    scaffold_X_mask = X[:, -1].bool()
    scaffold_E_mask = E[:, :, -1].bool()

    X = X[:, :-1]
    E = E[:, :, :-1]
    idx = data.idx if hasattr(data, "idx") else None
    return Data(
        X=X,
        E=E,
        scaffold_X_mask=scaffold_X_mask,
        scaffold_E_mask=scaffold_E_mask,
        y=data.y,
        idx=idx,
    )


def fragment_to_data(smile, query, atom_decoder, bond_decoder=None):
    mol = Chem.MolFromSmiles(smile, sanitize=True)
    mol = Chem.RemoveHs(mol)
    if bond_decoder is None:
        bond_decoder = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    if BT.AROMATIC not in bond_decoder:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    if mol is None:
        raise ValueError("Invalid mol")
    N = mol.GetNumAtoms()

    if isinstance(atom_decoder, list):
        while atom_decoder[-1] == "Y":
            del atom_decoder[-1]
        atom_decoder = {atom: i for i, atom in enumerate(atom_decoder)}

    type_idx = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in atom_decoder:
            return None
        type_idx.append(atom_decoder[atom.GetSymbol()])

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond_decoder[bond.GetBondType()] + 1]

    if len(row) == 0:
        return None

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bond_decoder) + 1).to(torch.float)
    x = F.one_hot(torch.tensor(type_idx), num_classes=len(atom_decoder)).float()

    query = Chem.MolFromSmiles(query, sanitize=False)
    Chem.SanitizeMol(
        query,
        sanitizeOps=(
            Chem.SanitizeFlags.SANITIZE_PROPERTIES
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
        ),
    )
    if BT.AROMATIC not in bond_decoder:
        Chem.Kekulize(query, clearAromaticFlags=True)
    if query is None:
        raise ValueError("Bad query smiles")

    node_mask = torch.zeros(N, dtype=torch.long)
    edge_mask = torch.zeros(edge_index.shape[1], dtype=torch.long)

    parts = Chem.GetMolFrags(
        strip_dummies_keep_neighbors(query), asMols=True, sanitizeFrags=False,
    )

    edges = {
        (edge_index[0, i].item(), edge_index[1, i].item()): i
        for i in range(edge_index.shape[1])
    }
    for part in parts:
        for match in mol.GetSubstructMatches(part):
            for a in match:
                node_mask[a] = 1

            for bond in part.GetBonds():
                q_begin = bond.GetBeginAtomIdx()
                q_end = bond.GetEndAtomIdx()

                mol_begin = match[q_begin]
                mol_end = match[q_end]

                if (mol_begin, mol_end) in edges:
                    idx = edges[(mol_begin, mol_end)]
                    edge_mask[idx] = 1
                    idx = edges[(mol_end, mol_begin)]
                    edge_mask[idx] = 1

    qed_value = QED.qed(mol)
    sa_value = sascorer.calculateScore(mol)
    y = torch.tensor([[qed_value, sa_value]])
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        scaffold_node_mask=node_mask,
        scaffold_edge_mask=edge_mask,
    )


def oracle_sa(smiles):
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
        return sascorer.calculateScore(mol) if mol else None

    return [
        sascorer.calculateScore(mol)
        if (mol := Chem.MolFromSmiles(s)) is not None
        else None
        for s in smiles
    ]


def eval_smiles(smiles):
    evaluator = Evaluator("diversity")
    oracle_qed = Oracle("qed")
    # oracle_sa = Oracle('sa')
    num_samples = len(smiles)
    smiles = [s for s in smiles if s is not None]
    if len(smiles) == 0:
        return {
            "Validity": 0,
            "Uniqueness": 0,
            "Diversity": None,
            "QED": None,
            "SA": None,
        }
    df = pd.DataFrame(
        {"smiles": smiles, "qed": oracle_qed(smiles), "sa": oracle_sa(smiles)}
    )
    val = len(df["smiles"]) / num_samples
    df = df.drop_duplicates("smiles")
    uniq = len(df["smiles"]) / len(smiles)
    if len(df["smiles"]) == 1:
        div = 0
    else:
        div = evaluator(df["smiles"])
    qed = df["qed"].mean()
    sa = df["sa"].mean()
    return {"Validity": val, "Uniqueness": uniq, "Diversity": div, "QED": qed, "SA": sa}


def strip_dummies_keep_neighbors(mol):
    rw = Chem.RWMol(mol)

    dummy_ids = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]

    for idx in sorted(dummy_ids, reverse=True):
        rw.RemoveAtom(idx)

    return rw.GetMol()
