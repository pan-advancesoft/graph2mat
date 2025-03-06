from metatrain.experimental.nanopet import NanoPET
from metatrain.utils.architectures import get_default_hypers

from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.experimental.nanopet import NanoPET

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
            "num_subtargets": 20,
            "quantity": "matrix",
            "unit": "eV",
        }

        target_info = get_generic_target_info(target)

        hypers = {
            **get_default_hypers("experimental.nanopet")["model"],
            "d_pet": d_pet,
            "num_gnn_layers": num_gnn_layers,
        }

        nanopet = NanoPET(
            hypers,
            DatasetInfo(
                length_unit="angstrom",
                atomic_types=[0, 1],
                targets={
                    "matrix": target_info,
                },
            ),
        )

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
        )

    def configure_optimizers(self):
        param_options = dict(
            params=self.model.parameters(),
            lr=self.hparams.optim_lr,
            amsgrad=self.hparams.optim_amsgrad,
        )

        return torch.optim.Adam(**param_options)
