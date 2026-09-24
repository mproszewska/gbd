import os
import pathlib
import sys
from tqdm import tqdm
from typing import Any, Sequence
import glob
import pickle

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT

import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph

from src import utils
from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from src.analysis.rdkit_functions import compute_molecular_metrics

from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import QED
from rdkit.Chem import RDConfig

sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
sys.path.append(sa_path)
import sascorer

from gbd.molgenbench_dataset import smiles_to_scaffold_data


ATOMS = {
    "O": 0,
    "P": 1,
    "N": 2,
    "F": 3,
    "Cl": 4,
    "S": 5,
    "C": 6,
    "Br": 7,
    "I": 8,
    "Si": 9,
    "H": 10,
}
# ATOMS = ['O', 'P', 'N', 'F', 'Cl', 'S', 'C', 'Br', 'I', 'Si', 'H']
BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2}
MAX_NODES = 40


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class MosesScaffoldDataset(InMemoryDataset):
    train_url = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv"
    val_url = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv"
    test_url = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv"

    def __init__(
        self,
        stage,
        root,
        filter_dataset: bool,
        remove_h: bool,
        remove_aromatic: bool,
        atom_decoder=None,
        bond_decoder=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.stage = stage
        self.filter_dataset = filter_dataset
        self.remove_h = remove_h
        self.remove_aromatic = remove_aromatic
        self.atom_decoder = atom_decoder
        self.bond_decoder = bond_decoder

        if self.stage == "train":
            self.file_idx = 0
        elif self.stage == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ["train_moses.csv", "val_moses.csv", "test_moses.csv"]

    @property
    def split_file_name(self):
        return ["train_moses.csv", "val_moses.csv", "test_moses.csv"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [os.path.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.filter_dataset:
            return [
                "train_filtered.pt",
                "test_filtered.pt",
                "test_scaffold_filtered.pt",
            ]
        else:
            return ["train.pt", "test.pt", "test_scaffold.pt"]

    def download(self):
        import rdkit  # noqa

        train_path = download_url(self.train_url, self.raw_dir)
        os.rename(train_path, os.path.join(self.raw_dir, "train_moses.csv"))

        test_path = download_url(self.test_url, self.raw_dir)
        os.rename(test_path, os.path.join(self.raw_dir, "val_moses.csv"))

        valid_path = download_url(self.val_url, self.raw_dir)
        os.rename(valid_path, os.path.join(self.raw_dir, "test_moses.csv"))

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        path = self.split_paths[self.file_idx]
        smiles_list = pd.read_csv(path)["SMILES"].values

        data_list = []
        smiles_kept = []
        for i, smiles in enumerate(tqdm(smiles_list, total=len(smiles_list))):
            data = smiles_to_scaffold_data(
                i, smiles, None, self.atom_decoder, self.bond_decoder, self.remove_h
            )
            if data is None:
                continue

            if self.filter_dataset:
                # Try to build the molecule again from the graph. If it fails, do not add it to the training set
                dense_data, node_mask = utils.to_dense(
                    data.x, data.edge_index, data.edge_attr, data.batch
                )
                dense_data = dense_data.mask(node_mask, collapse=True)
                X, E = dense_data.X, dense_data.E

                assert X.size(0) == 1
                atom_types = X[0]
                edge_types = E[0]
                mol = build_molecule_with_partial_charges(
                    atom_types, edge_types, self.atom_decoder
                )
                smiles = mol2smiles(mol)
                if smiles is not None:
                    try:
                        mol_frags = Chem.rdmolops.GetMolFrags(
                            mol, asMols=True, sanitizeFrags=True
                        )
                        if len(mol_frags) == 1:
                            data_list.append(data)
                            smiles_kept.append(smiles)

                    except Chem.rdchem.AtomValenceException:
                        print("Valence error in GetmolFrags")
                    except Chem.rdchem.KekulizeException:
                        print("Can't kekulize molecule")
            else:
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
                smiles_kept.append(smiles)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

        if self.filter_dataset:
            smiles_save_path = os.path.join(
                pathlib.Path(self.raw_paths[0]).parent, f"new_{self.stage}.smiles"
            )
            print(f"Saved smiles to {smiles_save_path}")
            with open(smiles_save_path, "w") as f:
                f.writelines("%s\n" % s for s in smiles_kept)
            print(f"Number of molecules kept: {len(smiles_kept)} / {len(smiles_list)}")


class MosesScaffoldDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir

        if cfg.dataset.skip_no_scaffold:

            def pre_filter(data):
                return data.scaffold_node_mask.float().mean() != 0

        else:
            pre_filter = None

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        self.filter_dataset = cfg.dataset.filter
        self.remove_h = cfg.dataset.remove_h
        self.remove_aromatic = cfg.dataset.remove_aromatic

        if cfg.dataset.defog_atoms_and_bonds:
            atom_decoder = ["C", "N", "S", "O", "F", "Cl", "Br", "H"]
            bond_decoder = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        else:
            atom_decoder = sorted(ATOMS, key=ATOMS.get)
            bond_decoder = BONDS

        datasets = {
            "train": MosesScaffoldDataset(
                stage="train",
                root=root_path,
                filter_dataset=self.filter_dataset,
                remove_h=self.remove_h,
                remove_aromatic=self.remove_aromatic,
                atom_decoder=atom_decoder,
                bond_decoder=bond_decoder,
                pre_filter=pre_filter,
            ),
            "val": MosesScaffoldDataset(
                stage="val",
                filter_dataset=self.filter_dataset,
                remove_h=self.remove_h,
                remove_aromatic=self.remove_aromatic,
                atom_decoder=atom_decoder,
                bond_decoder=bond_decoder,
                root=root_path,
            ),
            "test": MosesScaffoldDataset(
                stage="test",
                filter_dataset=self.filter_dataset,
                remove_h=self.remove_h,
                remove_aromatic=self.remove_aromatic,
                atom_decoder=atom_decoder,
                bond_decoder=bond_decoder,
                root=root_path,
            ),
        }
        train_len = len(datasets["train"].data.idx)
        val_len = len(datasets["val"].data.idx)
        test_len = len(datasets["test"].data.idx)

        print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
        super().__init__(cfg, datasets)


class MosesScaffoldInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.compute_fcd = cfg.dataset.compute_fcd
        self.has_scaffold = True

        self.name = cfg.dataset.name
        self.remove_h = cfg.dataset.remove_h
        self.remove_aromatic = cfg.dataset.remove_aromatic

        if cfg.dataset.defog_atoms_and_bonds:
            print("Original DeFoG atoms and bonds")
            self.atom_decoder = ["C", "N", "S", "O", "F", "Cl", "Br", "H"]
            self.atom_encoder = {atom: i for i, atom in enumerate(self.atom_decoder)}
            self.bond_decoder = {
                BT.SINGLE: 0,
                BT.DOUBLE: 1,
                BT.TRIPLE: 2,
                BT.AROMATIC: 3,
            }
            self.atom_weights = {
                0: 12,
                1: 14,
                2: 32,
                3: 16,
                4: 19,
                5: 35.4,
                6: 79.9,
                7: 1,
            }
            self.valencies = [4, 3, 4, 2, 1, 1, 1, 1]
            self.num_atom_types = len(self.atom_decoder)
            self.max_weight = 350

            self.n_nodes = torch.tensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    3.097634362347889692e-06,
                    1.858580617408733815e-05,
                    5.007842264603823423e-05,
                    5.678996240021660924e-05,
                    1.244216400664299726e-04,
                    4.486406978685408831e-04,
                    2.253012731671333313e-03,
                    3.231865121051669121e-03,
                    6.709992419928312302e-03,
                    2.289564721286296844e-02,
                    5.411050841212272644e-02,
                    1.099515631794929504e-01,
                    1.223291903734207153e-01,
                    1.280680745840072632e-01,
                    1.445975750684738159e-01,
                    1.505961418151855469e-01,
                    1.436946094036102295e-01,
                    9.265746921300888062e-02,
                    1.820066757500171661e-02,
                    2.065089574898593128e-06,
                ]
            )
            self.max_n_nodes = (
                len(self.n_nodes) - 1 if self.n_nodes is not None else None
            )
            self.node_types = torch.tensor(
                [
                    0.722338,
                    0.13661,
                    0.163655,
                    0.103549,
                    0.1421803,
                    0.005411,
                    0.00150,
                    0.0,
                ]
            )
            self.edge_types = torch.tensor(
                [0.89740, 0.0472947, 0.062670, 0.0003524, 0.0486]
            )
            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[:7] = torch.tensor(
                [0.0, 0.1055, 0.2728, 0.3613, 0.2499, 0.00544, 0.00485]
            )
        else:
            self.atom_encoder = ATOMS
            self.atom_decoder = sorted(ATOMS, key=ATOMS.get)
            self.bond_decoder = BONDS
            self.atom_weights = {
                0: 16,
                1: 30.9,
                2: 14,
                3: 19,
                4: 35.4,
                5: 32,
                6: 12,
                7: 79.9,
                8: 126.9,
                9: 28,
                10: 1,
            }

            self.valencies = [2, 3, 3, 1, 1, 4, 4, 1, 1, 4, 1]
            self.num_atom_types = len(self.atom_decoder)
            self.max_weight = 350
            self.max_n_nodes = MAX_NODES

            self.n_nodes = torch.tensor(
                [
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    0.0000e00,
                    1.7043e-05,
                    4.8855e-05,
                    5.5104e-05,
                    1.2271e-04,
                    4.4424e-04,
                    2.2161e-03,
                    3.2023e-03,
                    6.6574e-03,
                    2.2760e-02,
                    5.3961e-02,
                    1.0981e-01,
                    1.2227e-01,
                    1.2811e-01,
                    1.4476e-01,
                    1.5074e-01,
                    1.4386e-01,
                    9.2786e-02,
                    1.8178e-02,
                    2.2723e-06,
                ]
            )

            self.node_types = torch.tensor(
                [
                    0.1038,
                    0.0000,
                    0.1364,
                    0.0143,
                    0.0055,
                    0.0164,
                    0.7220,
                    0.0015,
                    0.0000,
                    0.0000,
                    0.0000,
                ]
            )

            self.edge_types = torch.tensor(
                [8.9733e-01, 7.4160e-02, 2.8154e-02, 3.5466e-04]
            )

            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = torch.zeros(79)
            self.valency_distribution[0:7] = torch.tensor(
                [0.0000, 0.1062, 0.2937, 0.3554, 0.2397, 0.0000, 0.0050]
            )

        if recompute_statistics:
            np.set_printoptions(suppress=True, precision=5)
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes, self.n_nodes.shape)
            np.savetxt("n_counts.txt", self.n_nodes.numpy())
            self.node_types = datamodule.node_types()  # There are no node types
            print("Distribution of node types", self.node_types, self.node_types.shape)
            np.savetxt("atom_types.txt", self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types, self.edge_types.shape)
            np.savetxt("edge_types.txt", self.edge_types.numpy())

            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies, valencies.shape)
            np.savetxt("valencies.txt", valencies.numpy())
            self.valency_distribution = valencies
