import os
import pathlib
import sys
from typing import Any, Sequence
from tqdm import tqdm
import glob
import pickle


import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import QED, RDConfig, BRICS

from src import utils
from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from src.analysis.rdkit_functions import compute_molecular_metrics

sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
sys.path.append(sa_path)
import sascorer

DATA_DIR = "MolGenBench/data/MolGenBench_Version3"
MOLGENBENCH_ATOMS = {
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
MOLGENBENCH_BONDS = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2}  # , BT.AROMATIC: 3}
MAX_NODES = 40
HIT2LEAD_PATTERN = (
    f"{DATA_DIR}/*/reference_active_molecules/Hit2Lead//top5_common_scaffold_info.csv"
)
REMOVE_H = True
REMOVE_AROMATIC = True


def smiles_to_scaffold_data(
    idx,
    s,
    scaffold_smiles=None,
    atom_decoder=None,
    bonds_decoder=None,
    stage="test",
    remove_h=True,
    remove_aromatic=True,
):
    mol = Chem.MolFromSmiles(s)
    if remove_h:
        mol = Chem.RemoveAllHs(mol)

    if scaffold_smiles is not None:
        scaffold = Chem.MolFromSmiles(scaffold_smiles, sanitize=False)
        if scaffold is None:
            assert False, f"Invalid scaffold smile {scaffold_smiles}"
        if remove_h:
            scaffold = Chem.RemoveAllHs(scaffold, sanitize=False)
    else:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if remove_aromatic:
            Chem.Kekulize(scaffold, clearAromaticFlags=True)

    if remove_aromatic:
        Chem.Kekulize(mol, clearAromaticFlags=True)

    if mol is None or scaffold is None:
        return None
    match = mol.GetSubstructMatch(scaffold)
    scaffold_atoms = set(match) if match else set()

    if stage == "test":
        if len(scaffold_atoms) == 0:
            return None
    else:
        if len(scaffold_atoms) == 0:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            match = mol.GetSubstructMatch(scaffold)
            scaffold_atoms = set(match) if match else set()

    if len(scaffold_atoms) == 0:
        return None

    N = mol.GetNumAtoms()
    atom_features = []
    atom_mask = []
    edge_mask = []

    if isinstance(atom_decoder, list):
        atom_decoder = {k: i for i, k in enumerate(atom_decoder)}

    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in atom_decoder:
            if atom not in scaffold_atoms:
                sym = "C"  # To handle models trained without atoms in molgenbench
            else:
                return None
        atom_features.append(atom_decoder[sym])
        atom_mask.append(1 if atom.GetIdx() in scaffold_atoms else 0)

    # Add fragment info
    for atom in mol.GetAtoms():
        atom.SetIntProp("origAtomIdx", atom.GetIdx())
    broken = BRICS.BreakBRICSBonds(mol)
    frags = Chem.GetMolFrags(broken, asMols=False)
    atom_to_frag = [0] * mol.GetNumAtoms()
    bond_to_frag = []

    for frag_idx, frag in enumerate(frags):
        for atom_idx in frag:
            atom = broken.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() == 0:
                continue
            atom_to_frag[atom.GetIntProp("origAtomIdx")] = frag_idx + 1

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        if bond.GetBondType() not in bonds_decoder:
            print("Invalid bond")
            return None
        edge_type += 2 * [bonds_decoder[bond.GetBondType()] + 1]
        in_scaffold = (start in scaffold_atoms) and (end in scaffold_atoms)
        edge_mask.append(in_scaffold)
        edge_mask.append(in_scaffold)

        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        frag1 = atom_to_frag[a1]
        frag2 = atom_to_frag[a2]
        if frag1 == frag2:
            bond_to_frag += 2 * [frag1]
        else:
            bond_to_frag += 2 * [0]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds_decoder) + 1).to(torch.float)
    edge_mask = torch.tensor(edge_mask).bool()
    atom_mask = torch.tensor(atom_mask).bool()

    atom_to_frag = torch.tensor(atom_to_frag, dtype=torch.long)
    bond_to_frag = torch.tensor(bond_to_frag, dtype=torch.long)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    edge_mask = edge_mask[perm]
    bond_to_frag = bond_to_frag[perm]

    x = F.one_hot(torch.tensor(atom_features), num_classes=len(atom_decoder)).float()
    if atom_mask.sum() == 0:
        print("Empty", idx, s, scaffold_smiles)

    qed_value = QED.qed(mol)
    sa_value = sascorer.calculateScore(mol)
    y = torch.tensor([[qed_value, sa_value]])

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        idx=idx,
        scaffold_node_mask=atom_mask,
        scaffold_edge_mask=edge_mask,
        atom_to_frag=atom_to_frag,
        bond_to_frag=bond_to_frag,
    )


def load_eval_set_molgenbench(
    stage, atom_decoder, bond_decoder, pre_filter, pre_transform
):
    if isinstance(atom_decoder, list):
        while atom_decoder[-1] == "Y":
            del atom_decoder[-1]
        atom_decoder = {atom: i for i, atom in enumerate(atom_decoder)}
    if bond_decoder is None:
        bond_decoder = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    files = glob.glob(HIT2LEAD_PATTERN)
    files = sorted(files)
    df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
    data_list = []
    eval_codes = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        code = row["UniProt_ID"] + "_" + row["SeriseID"]
        eval_codes.append(code)
        smiles = row["FilteredMols"].split(",")
        scaffold_smiles = row["Scaffold_SMILES"]
        datas = []
        for s in smiles:
            data = smiles_to_scaffold_data(
                i,
                s,
                scaffold_smiles,
                atom_decoder,
                bond_decoder,
                stage == "test",
                remove_h=REMOVE_H,
                remove_aromatic=REMOVE_AROMATIC,
            )
            if data is None:
                continue
            if pre_filter is not None and not pre_filter(data):
                continue
            if pre_transform is not None:
                data = pre_transform(data)

            datas.append(data)
        if stage == "eval":
            data_list.append(datas)
        else:
            data_list += datas

    return data_list, eval_codes


class MolGenBenchDataset(InMemoryDataset):

    hit2lead_pattern = HIT2LEAD_PATTERN
    all_smiles_pattern = f"{DATA_DIR}/*/reference_active_molecules/*_map_smiles.pkl"

    def __init__(
        self,
        stage,
        root,
        remove_h: bool,
        remove_aromatic: bool,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_all_data=False,
    ):
        self.stage = stage
        self.root = root
        self.remove_h = remove_h
        self.remove_aromatic = remove_aromatic

        super().__init__(root, transform, pre_transform, pre_filter)
        self.file_idx = {"train": 0, "val": 1, "test": 2}[self.stage]
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])
        self.eval_codes = []
        self.use_all_data = use_all_data

        self.atom_encoder = MOLGENBENCH_ATOMS
        self.atom_decoder = sorted(MOLGENBENCH_ATOMS, key=MOLGENBENCH_ATOMS.get)
        self.bonds_decoder = MOLGENBENCH_BONDS

        if self.stage in ["test"]:
            with open(self.processed_paths[3], "r") as f:
                for l in f:
                    i, code = l.strip().split(" ")
                    assert int(i) == len(self.eval_codes), (i, len(self.eval_codes))
                    self.eval_codes.append(code)

    def process(self):
        atom_encoder = MOLGENBENCH_ATOMS
        atom_decoder = sorted(MOLGENBENCH_ATOMS, key=MOLGENBENCH_ATOMS.get)
        bonds_decoder = MOLGENBENCH_BONDS

        RDLogger.DisableLog("rdApp.*")
        assert self.remove_h

        if self.stage in ["val", "test"]:
            data_list, eval_codes = load_eval_set_molgenbench(
                self.stage,
                atom_decoder,
                bonds_decoder,
                pre_filter=self.pre_filter,
                pre_transform=self.pre_transform,
                remove_h=self.remove_h,
                remove_atomatic=self.remove_aromatic,
            )

            if self.stage == "test":
                with open(self.processed_paths[3], "w") as f:
                    for i, code in enumerate(eval_codes):
                        f.write(f"{i} {code}\n")

        else:
            files = glob.glob(self.all_smiles_pattern)
            files = sorted(files)
            all_smiles = []
            for filename in tqdm(files, total=len(files)):
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                all_smiles += data.values()

            print("All smiles:", len(all_smiles))
            all_smiles = list(set(all_smiles))
            print("All unique smiles:", len(all_smiles))

            data_list = []
            count_scaffolds = 0
            count_filtered = 0
            for i, s in enumerate(tqdm(all_smiles)):
                data = smiles_to_scaffold_data(
                    i,
                    s,
                    None,
                    MOLGENBENCH_ATOMS,
                    MOLGENBENCH_BONDS,
                    self.stage == "test",
                    remove_h=self.remove_h,
                    remove_aromatic=self.remove_aromatic,
                )
                if data is None:
                    count_filtered += 1
                    continue
                count_scaffolds += data.scaffold_node_mask.float().mean().item()
                if self.pre_filter is not None and not self.pre_filter(data):
                    count_filtered += 1
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                if len(data.x) > MAX_NODES:
                    continue

                data_list.append(data)

                if len(data_list) % 1000 == 0:
                    print(f"Scaffolds {count_scaffolds/len(data_list):.4f}")
            print(f"Scaffolds {count_scaffolds/len(data_list):.4f}")
            print(f"Filtered (skipped) {count_filtered}")

        self.file_idx = {"train": 0, "val": 1, "test": 2}[self.stage]
        print("Size", self.stage, len(data_list))
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

    @property
    def processed_file_names(self):
        return ["proc_tr.pt", "proc_val.pt", "proc_test.pt", "eval_codes.txt"]


class MolGenBenchDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        if cfg.dataset.skip_no_scaffold:

            def pre_filter(data):
                return data.scaffold_node_mask.float().mean() != 0

        else:
            pre_filter = None
        self.pre_filter = pre_filter
        self.remove_h = cfg.dataset.remove_h
        self.remove_aromatic = cfg.dataset.remove_aromatic
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {
            "train": MolGenBenchDataset(
                stage="train",
                root=root_path,
                pre_filter=pre_filter,
                remove_h=self.remove_h,
                remove_aromatic=self.remove_aromatic,
            ),
            "val": MolGenBenchDataset(
                stage="val",
                root=root_path,
                remove_h=self.remove_h,
                remove_aromatic=self.remove_aromatic,
            ),
            "test": MolGenBenchDataset(
                stage="test",
                root=root_path,
                remove_h=self.remove_h,
                remove_aromatic=self.remove_aromatic,
            ),
        }
        train_len = len(datasets["train"].data.idx)
        val_len = len(datasets["val"].data.idx)
        test_len = len(datasets["test"].data.idx)

        print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
        super().__init__(cfg, datasets)


class MolGenBenchInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.compute_fcd = cfg.dataset.compute_fcd
        self.eval_codes = datamodule.test_dataset.eval_codes
        self.eval_set = datamodule.test_dataset
        self.pre_filter = datamodule.pre_filter
        self.pre_transform = None
        self.has_scaffold = True

        self.name = "molgenbench"
        self.remove_h = cfg.dataset.remove_h
        self.remove_aromatic = cfg.dataset.remove_aromatic

        self.atom_encoder = MOLGENBENCH_ATOMS
        self.atom_decoder = sorted(MOLGENBENCH_ATOMS, key=MOLGENBENCH_ATOMS.get)
        self.bonds_decoder = MOLGENBENCH_BONDS
        self.valencies = [2, 5, 3, 1, 1, 4, 4, 1, 1, 4, 1]
        self.num_atom_types = len(MOLGENBENCH_ATOMS)
        self.max_n_nodes = MAX_NODES
        self.max_weight = 390
        self.atom_weights = {
            0: 16,
            1: 31,
            2: 14,
            3: 19,
            4: 35,
            5: 32,
            6: 12,
            7: 80,
            8: 127,
            9: 28,
            10: 1,
        }
        self.n_nodes = torch.tensor(
            [
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                5.7158e-05,
                1.3510e-04,
                1.5589e-04,
                4.2089e-04,
                6.7031e-04,
                1.2315e-03,
                1.4445e-03,
                2.3331e-03,
                3.3931e-03,
                4.8688e-03,
                6.7135e-03,
                9.2648e-03,
                1.3484e-02,
                1.7631e-02,
                2.1699e-02,
                2.6454e-02,
                3.2273e-02,
                3.8109e-02,
                4.4417e-02,
                4.8896e-02,
                5.6332e-02,
                6.0629e-02,
                6.3690e-02,
                6.5581e-02,
                6.6106e-02,
                6.6054e-02,
                6.2926e-02,
                5.7236e-02,
                5.1266e-02,
                4.5144e-02,
                4.1221e-02,
                3.5038e-02,
                3.0496e-02,
                2.4277e-02,
                6.7550e-05,
                5.7158e-05,
                2.5981e-05,
                4.6766e-05,
                2.0785e-05,
                3.1177e-05,
                1.5589e-05,
                1.0392e-05,
                1.0392e-05,
                5.1962e-06,
                1.0392e-05,
                0.0000e00,
                5.1962e-06,
                5.1962e-06,
                0.0000e00,
                5.1962e-06,
                0.0000e00,
                5.1962e-06,
                0.0000e00,
                5.1962e-06,
                5.1962e-06,
                0.0000e00,
                0.0000e00,
                5.1962e-06,
                0.0000e00,
                5.1962e-06,
                1.0392e-05,
            ]
        )

        self.node_types = torch.tensor(
            [
                8.5941e-02,
                1.7483e-04,
                1.3577e-01,
                2.0295e-02,
                8.6609e-03,
                1.1538e-02,
                7.3592e-01,
                1.4684e-03,
                2.0942e-04,
                2.5675e-05,
                0.0000,
            ]
        )

        self.edge_types = torch.tensor(
            [9.2652e-01, 3.2542e-02, 3.6271e-03, 2.1603e-04]
        )  # , 3.7099e-02])

        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        self.valency_distribution = torch.zeros(199)
        self.valency_distribution[0:7] = torch.tensor(
            [0.0000, 0.1032, 0.2359, 0.3758, 0.2752, 0.0044, 0.0055]
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
            assert False


def get_smiles(cfg, datamodule, dataset_infos, evaluate_datasets=False):

    return {
        "train": get_loader_smiles(
            cfg,
            datamodule.train_dataloader(),
            dataset_infos,
            "train",
            evaluate_dataset=evaluate_datasets,
        ),
        "val": get_loader_smiles(
            cfg,
            datamodule.val_dataloader(),
            dataset_infos,
            "val",
            evaluate_dataset=evaluate_datasets,
        ),
        "test": get_loader_smiles(
            cfg,
            datamodule.test_dataloader(),
            dataset_infos,
            "test",
            evaluate_dataset=evaluate_datasets,
        ),
    }


def get_loader_smiles(
    cfg, dataloader, dataset_infos, split_key, evaluate_dataset=False,
):
    datadir = cfg.dataset.datadir
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = f"{split_key}_smiles.npy"
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print(f"Dataset {split_key} smiles were found.")
        smiles = np.load(smiles_path).tolist()
    else:
        print(f"Computing dataset {split_key} smiles...")
        smiles = compute_molgenbench_smiles(atom_decoder, dataloader)
        np.save(smiles_path, np.array(smiles))

    if evaluate_dataset:
        # Convert loader to molecules
        assert (
            dataset_infos is not None
        ), "If wanting to evaluate dataset, need to pass dataset_infos"
        all_molecules = []
        for i, data in enumerate(dataloader):
            dense_data, node_mask = utils.to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print(
            "Evaluating the dataset -- number of molecules to evaluate",
            len(all_molecules),
        )
        # load train smiles
        train_smiles_file_name = f"train_smiles.npy"
        train_smiles_path = os.path.join(root_dir, datadir, train_smiles_file_name)
        train_smiles = np.load(train_smiles_path)
        # get evaluation and output
        metrics = compute_molecular_metrics(
            molecule_list=all_molecules,
            train_smiles=train_smiles,
            dataset_info=dataset_infos,
        )

    return smiles


def compute_molgenbench_smiles(atom_decoder, train_dataloader):
    """
    :param dataset_name: qm9 or qm9_second_half
    :return:
    """
    print(f"\tConverting MolGenBench dataset to SMILES...")
    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(
                molecule[0], molecule[1], atom_decoder
            )
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        if i % 1000 == 0:
            print(
                "\tConverting MolGenBench dataset to SMILES {0:.2%}".format(
                    float(i) / len_train
                )
            )
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles
