"""Model loading utilities for the FACET-II 571 injector surrogate model."""

import sys
from pathlib import Path
from lume_torch.models import TorchModel, TorchModule

from facet2_inj_ml_model_571 import lume_model_utils
from facet2_inj_ml_model_571.lume_model_utils import (
    CovMeanTorchModel,
    CovMeanTorchModule,
)

# Register as top-level module so torch.load can unpickle classes serialized
# when lume_model_utils lived outside the package.
sys.modules.setdefault("lume_model_utils", lume_model_utils)


_MODEL_CONFIGS = {
    "machine": "lumetorchyaml-machine/injector_machine.yaml",
    "sim": "lumetorchyaml-sim/injector_simulator.yaml",
}

_FULL_MODEL_CONFIGS = {
    "machine": "lumetorchyaml-machine-full/injector_machine.yaml",
    "sim": "lumetorchyaml-sim-full/injector_simulator.yaml",
}


def get_resource_path(filename):
    """Get the absolute path to a resource file."""
    package_dir = Path(__file__).parent
    resource_path = package_dir / "resources" / filename
    
    if not resource_path.exists():
        raise FileNotFoundError(f"Resource file not found: {resource_path}")
    
    return str(resource_path)


def load_model(input_space="machine", full=False):
    """
    Load the FACET-II 571 injector surrogate model.

    Parameters
    ----------
    input_space : str, optional
        Which input space the model expects. ``"machine"`` (default) accepts
        machine PV values; ``"sim"`` accepts simulator parameters.
    full : bool, optional
        If True, load the full model that predicts covariance_matrix (6x6)
        and all 6 phase-space means (mean_x, mean_px, mean_y, mean_py,
        mean_t, mean_pz). If False (default), load the covariance-only model.

    Returns
    -------
    TorchModel or CovMeanTorchModel
        Loaded model instance ready for inference.
    
    Example
    -------
        >>> from facet2_inj_ml_model_571 import load_model
        >>> model = load_model()                          # cov-only, machine PVs
        >>> model_full = load_model(full=True)            # cov + phase-space means
        >>> model_sim = load_model("sim", full=True)      # sim inputs, full

    """
    configs = _FULL_MODEL_CONFIGS if full else _MODEL_CONFIGS
    if input_space not in configs:
        raise ValueError(
            f"Unknown input_space {input_space!r}; expected one of {list(configs)}"
        )
    config_path = get_resource_path(configs[input_space])
    
    if full:
        return CovMeanTorchModel(config_path)
    return TorchModel(config_path)