import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from rdkit import Chem
import torch
from torch_geometric.data import Data


def mol_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES string")

    node_features = []

    for atom in mol.GetAtoms():
        node_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetMass()
        ])

    x = torch.tensor(node_features, dtype=torch.float)

    edge_index = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(
            edge_index,
            dtype=torch.long
        ).t().contiguous()

    return Data(
        x=x,
        edge_index=edge_index
    )