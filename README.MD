# FACET-II Injector ML Model — Screen 571

This repository contains the packaged surrogate model for predicting beam properties at screen 571 in the FACET-II injector beamline.

It includes both inference assets and the training implementation.

## Model Description

The model was trained on IMPACT-T simulation data to predict beam properties at screen 571. It consumes 19 scalar inputs (machine PVs or simulator parameters) and predicts:

- A full **6×6 beam covariance matrix** (from 21 lower-triangular Cholesky factors)
- All **6 phase-space mean values**: mean_x (m), mean_px (eV/c), mean_y (m), mean_py (eV/c), mean_t (s), mean_pz (eV/c) (optional, via `full=True`)

The network has a shared backbone with two heads: a Cholesky head (21 outputs → `C = L @ L^T`) and a mean head (6 outputs). Loss is computed in per-element normalized covariance space.

### M-normalization

Covariance targets are trained in M-normalized space. The diagonal normalization matrix is:

```
M_DIAG = [1e3, 1e-6, 1e3, 1e-6, 1e12, 1e-6]
```

The lume-torch output transformer automatically denormalizes: `C_phys = M_inv @ C_norm @ M_inv^T`.

## LUME-Torch Interface

`load_model()` returns a lume-torch model configured for either machine-PV or simulator-parameter inputs, with optional mean predictions.

```python
from facet2_inj_ml_model_571 import load_model

model = load_model()                        # cov-only, machine PVs
model = load_model("sim")                   # cov-only, sim parameters
model_full = load_model(full=True)          # cov + phase-space means
model_full = load_model("sim", full=True)   # sim inputs, full output
```

### Machine-input model (default)

`load_model()` or `load_model("machine")` applies a PV-to-simulator affine transform followed by normalization.

Runtime input variables (19):

| # | Machine PV |
|---|---|
| 1 | `QUAD:IN10:121:BCTRL` |
| 2 | `KLYS:LI10:21:AMPL` |
| 3 | `KLYS:LI10:21:PHAS` |
| 4 | `KLYS:LI10:31:PHAS` |
| 5 | `L0AF_scale:rf_field_scale` |
| 6 | `KLYS:LI10:41:PHAS` |
| 7 | `L0BF_scale:rf_field_scale` |
| 8 | `QUAD:IN10:361:BCTRL` |
| 9 | `QUAD:IN10:371:BCTRL` |
| 10 | `QUAD:IN10:425:BCTRL` |
| 11 | `QUAD:IN10:441:BCTRL` |
| 12 | `QUAD:IN10:511:BCTRL` |
| 13 | `QUAD:IN10:525:BCTRL` |
| 14 | `SOLN:IN10:121:BCTRL` |
| 15 | `QUAD:IN10:122:BCTRL` |
| 16 | `distgen:VCC` |
| 17 | `distgen:t_dist:sigma_t:value` |
| 18 | `TORO:IN10:591:TMIT_PC` |
| 19 | `impact_VCC_Cal` |

### Simulator-input model

`load_model("sim")` applies normalization only (no PV mapping).

### Output

**Covariance-only** (`full=False`, default):

- `covariance_matrix`: `torch.float32` tensor with shape `[6, 6]`

**Full** (`full=True`):

- `covariance_matrix`: `torch.float32` tensor with shape `[6, 6]`
- `mean_x`: scalar (m)
- `mean_px`: scalar (eV/c)
- `mean_y`: scalar (m)
- `mean_py`: scalar (eV/c)
- `mean_t`: scalar (s)
- `mean_pz`: scalar (eV/c)

## Training Column Mapping

The machine-input model maps PVs to simulator parameters automatically via an affine transform.

| Sim parameter | Machine PV | Scaling | Offset |
|---|---|---|---|
| `CQ10121:b1_gradient` | `QUAD:IN10:121:BCTRL` | -2.1 | 0 |
| `GUNF:rf_field_scale` | `KLYS:LI10:21:AMPL` | 7.898e-7 | 0 |
| `GUNF:theta0_deg` | `KLYS:LI10:21:PHAS` | 1.0 | 152.3 |
| `SOL10111:solenoid_field_scale` | `SOLN:IN10:121:BCTRL` | 1.6 | 0 |
| `SQ10122:b1_gradient` | `QUAD:IN10:122:BCTRL` | -2.1 | 0 |
| `distgen:t_dist:sigma_t:value` | (same) | 1.0 | -1.17 |
| `distgen:total_charge:value` | `TORO:IN10:591:TMIT_PC` | 1.0 | 0 |
| `L0AF_scale:rf_field_scale` | (same) | 1.0 | -62380013.2 |
| `L0AF_phase:theta0_deg` | `KLYS:LI10:31:PHAS` | 1.0 | 25.5 |
| `L0BF_scale:rf_field_scale` | (same) | 1.0 | -59886109.4 |
| `L0BF_phase:theta0_deg` | `KLYS:LI10:41:PHAS` | 1.0 | 137.5 |
| `QA10361` | `QUAD:IN10:361:BCTRL` | -1.08 | 0 |
| `QA10371` | `QUAD:IN10:371:BCTRL` | -1.08 | 0 |
| `QE10425` | `QUAD:IN10:425:BCTRL` | -1.08 | 0 |
| `QE10441` | `QUAD:IN10:441:BCTRL` | -1.08 | 0 |
| `QE10511` | `QUAD:IN10:511:BCTRL` | -1.08 | 0 |
| `QE10525` | `QUAD:IN10:525:BCTRL` | -1.08 | 0 |
| `distgen:VCC` | (same) | 1.0 | 0 |
| `impact_VCC_Cal` | (same) | 1.0 | -7.02e-6 |

## Model Architecture

19 inputs → shared backbone → Cholesky head (21 outputs → 6×6 cov) + mean head (2 outputs):

```
Backbone:
  Linear(19 -> 100), ELU
  Linear(100 -> 200), ELU, Dropout(0.05)
  Linear(200 -> 200), ELU, Dropout(0.05)
  Linear(200 -> 300), ELU, Dropout(0.05)
  Linear(300 -> 300), ELU, Dropout(0.05)
  Linear(300 -> 200), ELU, Dropout(0.05)
  Linear(200 -> 100), ELU, Dropout(0.05)
  Linear(100 -> 100), ELU
  Linear(100 -> 100), ELU

Cholesky head:
  Linear(100 -> 21) -> build L (lower triangular) -> C = L @ L^T

Mean head:
  Linear(100 -> 6) -> [mean_x, mean_px, mean_y, mean_py, mean_t, mean_pz]
```

### Normalization

Before training, the code computes:

- `x_mean`, `x_std` from the training split inputs
- `y_mean`, `y_std` from the training split Cholesky targets
- `cov_mean`, `cov_std` from the reconstructed training split covariance matrices
- `mean_y_mean`, `mean_y_std` from the 6 phase-space mean columns (mean_x, mean_px, mean_y, mean_py, mean_t, mean_pz)

## Usage

Install:

```bash
pip install -e .
```

Covariance-only inference:

```python
from facet2_inj_ml_model_571 import load_model

model = load_model()
result = model.evaluate({
    "QUAD:IN10:121:BCTRL": -0.015,
    "KLYS:LI10:21:AMPL": 40.17,
    "KLYS:LI10:21:PHAS": -77.42,
    # ... (all 19 inputs)
})
cov = result["covariance_matrix"]   # shape [6, 6]
```

Full inference (covariance + means):

```python
model = load_model(full=True)
result = model.evaluate({...})      # same 19 inputs
cov = result["covariance_matrix"]   # shape [6, 6]
mean_x = result["mean_x"]          # scalar tensor (m)
mean_px = result["mean_px"]        # scalar tensor (eV/c)
mean_y = result["mean_y"]          # scalar tensor (m)
mean_py = result["mean_py"]        # scalar tensor (eV/c)
mean_t = result["mean_t"]          # scalar tensor (s)
mean_pz = result["mean_pz"]        # scalar tensor (eV/c)
```

## Custom lume-torch Classes

The full model uses custom lume-torch subclasses (in `lume_model_utils.py`) to handle mixed output types (NDVariable for the 6×6 matrix + ScalarVariable for means):

- `CovMeanTorchModel`: Custom `_parse_outputs` that splits a flat (batch, 42) tensor into the 6×6 matrix and 6 scalar outputs
- `CovMeanTorchModule`: Custom `_dictionary_to_tensor` that flattens NDVariables before concatenation
- `FullOutputDenormTransform`: Handles the tuple model output, applies M-denormalization to the covariance and passes through the mean predictions, returns a flat tensor

## Training Modules

The training implementation lives in `facet2_inj_ml_model_571/training.py`.

For compatibility with serialized PyTorch artifacts that reference `train.CovarianceSurrogateModel`, the package registers a `train` compatibility alias from `facet2_inj_ml_model_571/__init__.py`.

Expected data requirements:

- The input features are the seven simulator/training columns listed in the table above.
- The targets are 21 lower-triangular Cholesky columns named `cov_chol_0` through `cov_chol_20`.
- The repository does not include dataset generation or preprocessing code for producing these CSV files from raw simulation data.

If your previous training workflow created those CSVs in a separate repository, that preprocessing step is still required before retraining here.

