from metatensor.torch.atomistic import System, ModelOutput

from metatrain.experimental.nanopet import NanoPET
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from e3nn import o3

import torch

from graph2mat.bindings.e3nn import E3nnGraph2Mat


def tensormap_to_e3nn(tmap):
    all_irreps = []
    all_vals = []
    for labels, block in tmap.items():
        l = labels["o3_lambda"]
        all_irreps.append(o3.Irreps([(len(block.properties), (l, (-1) ** l))]))

        if block.shape[0] != 0:
            all_vals.append(block.values.reshape(block.values.shape[0], -1))

    if len(all_vals) == 0:
        all_vals = torch.zeros(0, 0)

    else:
        all_vals = torch.concatenate(all_vals, dim=1)
    out_irreps = sum(all_irreps, o3.Irreps())

    return all_vals, out_irreps


def flatten_tensormap(tmap):
    all_vals = []
    for labels, block in tmap.items():
        all_vals.append(block.values.reshape(block.values.shape[0], -1))

    all_vals = torch.cat(all_vals, dim=1)

    return all_vals


class MatrixNanoPET(torch.nn.Module):
    """Model that wraps a MACE model to produce a matrix output."""

    def __init__(
        self, nanopet: NanoPET, readout_per_interaction: bool = False, **kwargs
    ):
        super().__init__()

        self.nanopet = nanopet

        _, self.nanopet_out_irreps = tensormap_to_e3nn(
            tmap=self.nanopet.dataset_info.targets["matrix"].layout
        )

        self.readout_per_interaction = readout_per_interaction

        edge_hidden_irreps = kwargs.pop("edge_hidden_irreps", None)

        self.matrix_readouts = E3nnGraph2Mat(
            irreps=dict(
                # node_attrs_irreps=self.mace.interactions[0].node_attrs_irreps,
                node_feats_irreps=self.nanopet_out_irreps,
                # edge_attrs_irreps=self.mace.interactions[0].edge_attrs_irreps,
                # edge_feats_irreps=self.mace.interactions[0].edge_feats_irreps,
                edge_hidden_irreps=edge_hidden_irreps,
            ),
            **kwargs,
        )

    def forward(self, data, compute_force=False, **kwargs):
        systems = []
        for i in range(len(data)):
            example = data[i]
            types = example["point_types"]
            positions = example["positions"]
            cell = example["cell"]
            pbc = torch.tensor([True, True, True])

            system = System(types, positions, cell, pbc)

            system = get_system_with_neighbor_lists(
                system, get_requested_neighbor_lists(self.nanopet)
            )

            systems.append(system)

        nanopet_out = self.nanopet(
            systems, outputs={"matrix": ModelOutput(per_atom=True)}
        )

        node_features = flatten_tensormap(nanopet_out["matrix"])

        node_labels, edge_labels = self.matrix_readouts(
            data=data,
            node_feats=node_features,
        )

        return {**nanopet_out, "node_labels": node_labels, "edge_labels": edge_labels}
