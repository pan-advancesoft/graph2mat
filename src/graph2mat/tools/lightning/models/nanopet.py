from metatrain.experimental.nanopet import NanoPET
from metatrain.utils.architectures import get_default_hypers

from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.experimental.nanopet import NanoPET

from metatrain.experimental.nativepet import NativePET

from metatrain.utils.data import DatasetInfo

from typing import Type, Union, Optional

from e3nn import o3
import torch

# from context import mace
from graph2mat.core.data.metrics import OrbitalMatrixMetric, block_type_mse
from graph2mat import BasisTableWithEdges

from graph2mat.bindings.e3nn import (
    E3nnSimpleNodeBlock,
    E3nnSimpleEdgeBlock,
    E3nnGraph2Mat,
    E3nnInteraction,
    E3nnEdgeMessageBlock,
)

from graph2mat.models.nanopet import MatrixNanoPET

from graph2mat.tools.lightning import LitBasisMatrixModel


class LitNanoPETMatrixModel(LitBasisMatrixModel):
    model: NanoPET

    def __init__(
        self,
        root_dir: str = ".",
        basis_files: Union[str, None] = None,
        basis_table: Union[BasisTableWithEdges, None] = None,
        no_basis: Optional[dict] = None,
        # NanoPET OPTIONS
        d_pet: int = 128,
        num_gnn_layers: int = 2,
        # NanoPET OPTIONS end
        edge_hidden_irreps: Union[o3.Irreps, str] = "4x0e+4x1o+4x2e",
        symmetric_matrix: bool = False,
        preprocessing_nodes: Optional[Type[torch.nn.Module]] = None,
        preprocessing_edges: Optional[Type[torch.nn.Module]] = None,
        preprocessing_edges_reuse_nodes: bool = True,
        node_block_readout: Type[torch.nn.Module] = E3nnSimpleNodeBlock,
        edge_block_readout: Type[torch.nn.Module] = E3nnSimpleEdgeBlock,
        readout_per_interaction: bool = False,
        optim_wdecay: float = 5e-7,
        optim_amsgrad: bool = True,
        optim_lr: float = 1e-3,
        loss: Type[OrbitalMatrixMetric] = block_type_mse,
        initial_node_feats: str = "OneHotZ",
        version: str = "new",
    ):
        model_cls = MatrixNanoPET  # if version == "new" else OrbitalMatrixMACE

        super().__init__(
            root_dir=root_dir,
            basis_files=basis_files,
            basis_table=basis_table,
            no_basis=no_basis,
            loss=loss,
            initial_node_feats="OneHotZ",
            model_cls=model_cls,
        )
        self.save_hyperparameters()

        if isinstance(edge_hidden_irreps, str):
            edge_hidden_irreps = o3.Irreps(edge_hidden_irreps)

        num_subtargets = 40
        target = {
            "type": {
                "spherical": {
                    "irreps": [
                        {"o3_lambda": 0, "o3_sigma": 1},
                        {"o3_lambda": 1, "o3_sigma": -1},
                        {"o3_lambda": 2, "o3_sigma": 1},
                    ]
                }
            },
            "per_atom": True,
            "num_subtargets": num_subtargets,
            "quantity": "matrix",
            "unit": "eV",
        }

        target_info = get_generic_target_info(target)

        hypers = {
            **get_default_hypers("experimental.nativepet")["model"],
            "d_pet": d_pet,
            "num_gnn_layers": num_gnn_layers,
        }

        nanopet = NativePET(
            hypers,
            DatasetInfo(
                length_unit="angstrom",
                atomic_types=torch.arange(len(self.basis_table.basis)),
                targets={
                    "matrix": target_info,
                },
            ),
        )

        from graph2mat.bindings.torch import TorchMatrixBlock, TorchGraph2Mat

        class MyG2M(TorchGraph2Mat):
            def __init__(self, irreps, **kwargs):
                kwargs["matrix_block_cls"] = TorchMatrixBlock
                super().__init__(**kwargs)

        class NodeOp(torch.nn.Module):
            def __init__(self, i_basis, j_basis, symmetry):
                super().__init__()
                self.in_dim = num_subtargets * (1 + 3 + 5)
                self.out_shape = (len(i_basis), len(j_basis))
                self.symmetry = symmetry

                # self.linear = torch.nn.Linear(self.in_dim, 10 * self.out_shape[0])
                # self.linear = torch.nn.Linear(
                #     self.in_dim, self.out_shape[0] * self.out_shape[0]
                # )

                internal_size = 10
                self.linear = torch.nn.Linear(self.in_dim, internal_size)

                self.weights = torch.nn.Parameter(
                    torch.zeros(
                        internal_size,
                        internal_size,
                        self.out_shape[0],
                        self.out_shape[1],
                    )
                )

                # self.linear = torch.nn.Sequential(
                #     torch.nn.Linear(self.in_dim, 100),
                #     torch.nn.ReLU(),
                #     torch.nn.Linear(100, self.out_shape[0] * self.out_shape[0]),
                # )

            def forward(self, node_feats):
                # out = self.linear(node_feats)

                # out = out.reshape(-1, 10, self.out_shape[0])
                # return torch.einsum("bnx, bny -> bxy", out, out)

                # out = out.reshape(-1, self.out_shape[0], self.out_shape[0])

                node_feats = self.linear(node_feats)

                out = torch.einsum(
                    "ni, ijkl, nj -> nkl",
                    node_feats,
                    self.weights,
                    node_feats,
                )

                return out

        class EdgeOp(torch.nn.Module):
            def __init__(self, i_basis, j_basis, symmetry):
                super().__init__()
                self.in_dim = num_subtargets * (1 + 3 + 5)
                self.out_shape = (len(i_basis), len(j_basis))

                # self.linear1 = torch.nn.Linear(self.in_dim, 10 * self.out_shape[0])
                # self.linear2 = torch.nn.Linear(self.in_dim, 10 * self.out_shape[1])

                internal_size = 10

                self.linear = torch.nn.Linear(self.in_dim, internal_size)

                self.weights = torch.nn.Parameter(
                    torch.zeros(
                        internal_size,
                        internal_size,
                        self.out_shape[0],
                        self.out_shape[1],
                    )
                )

            def forward(self, node_feats):
                # out1 = self.linear1(node_feats[0]).reshape(-1, 10, self.out_shape[0])
                # out2 = self.linear2(node_feats[1]).reshape(-1, 10, self.out_shape[1])

                out1 = self.linear(node_feats[0])
                out2 = self.linear(node_feats[1])

                return torch.einsum(
                    "ni, ijkl, nj -> nkl",
                    out1,
                    self.weights,
                    out2,
                )

                return torch.einsum("bnx, bny -> bxy", out1, out2)

        node_block_readout = NodeOp
        edge_block_readout = EdgeOp
        graph2mat_cls = MyG2M
        # graph2mat_cls = E3nnGraph2Mat

        self.init_model(
            nanopet=nanopet,
            readout_per_interaction=readout_per_interaction,
            unique_basis=self.basis_table.basis,
            edge_hidden_irreps=edge_hidden_irreps,
            symmetric=symmetric_matrix,
            preprocessing_nodes=preprocessing_nodes,
            preprocessing_edges=preprocessing_edges,
            preprocessing_edges_reuse_nodes=preprocessing_edges_reuse_nodes,
            node_operation=node_block_readout,
            edge_operation=edge_block_readout,
            graph2mat_cls=graph2mat_cls,
        )

    def configure_optimizers(self):
        param_options = dict(
            params=self.model.parameters(),
            lr=self.hparams.optim_lr,
            amsgrad=self.hparams.optim_amsgrad,
        )

        return torch.optim.Adam(**param_options)
