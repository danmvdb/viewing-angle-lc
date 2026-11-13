# Analytic diffusion light curves

Code accompanying *An analytic diffusion model for viewing-angle effects in
explosive transients* by D. van der Boom and D. Kushnir.

Calculate bolometric radioactive light curves for homologously expanding,
uniform-density ejecta with constant grey opacity and a uniform initial 56Ni
source. Supported geometries are an off-centre spherical source in spherical
ejecta, and centred sources in full-oblate or first-order slightly-oblate ejecta.
Viewing-angle luminosities use an Eddington grey-atmosphere kernel.

## Installation

Requires Python 3.8 or newer. From the cloned repository directory, install the
code and its dependencies (NumPy, SciPy, and PyYAML):

```sh
python -m pip install .
```

## Quick start

Run the supplied off-centre spherical configuration from the repository directory:

```sh
python -m run_experiment --config_path experiment_params_cfg.yaml
```

This writes `analytic_LC.npz` and `analytic_LC_config.yaml`. To change the
geometry, use one of the complete, commented configurations:

| Geometry | Configuration | Geometry parameters (YAML) |
| --- | --- | --- |
| Off-centre sphere | [Example](examples/off_center_sphere/experiment_params_cfg.yaml) | `xs`, `z_ig` |
| Oblate | [Example](examples/oblate/experiment_params_cfg.yaml) | `mu_s`, `mu_1`, `f` |
| Slightly oblate | [Example](examples/slightly_oblate/experiment_params_cfg.yaml) | `xs`, `eps`, `eps_s` |

The off-centre and oblate examples are the analytic calculations of Figs 2 and 3
of the paper (complete gamma-ray deposition, `tdep: null`); each file explains
how to switch on partial gamma-ray trapping.

For example, run the oblate model in its own directory:

```sh
cd examples/oblate
python -m run_experiment --config_path experiment_params_cfg.yaml
```

Relative input and output paths refer to the **working directory**. The YAML
field `url` is the output filename prefix: `url: analytic_LC` produces the two
files above. The saved YAML records the full configuration, effective spatial
settings, and provenance, and can be reused as runner input.

## Python example

For uniform-density ejecta of 1 Msun, kinetic energy 1e51 erg, and opacity
0.2 cm²/g, let 0.6 Msun of 56Ni fill a sphere of 60 per cent of the ejecta
volume, displaced by 0.1 of the ejecta radius toward `cos(theta) = +1`.
This example assumes full gamma-ray trapping:

```python
import numpy as np
from spherical_diffusion_solutions import (
    DAY, MSUN, light_curve_offcenter_viewing_angle_rm_degeneracy_1a,
)

M, E_kin, kappa, c = 1.0 * MSUN, 1e51, 0.2, 2.998e10
v_o = np.sqrt(10 * E_kin / (3 * M))              # outer velocity [cm/s]
t_unit = 9 * kappa * M / (4 * np.pi * c * v_o)   # v_o^2/D [s^2], ~7.4e12

f_ni = 0.6
t, L_total, L_view, cos_view = light_curve_offcenter_viewing_angle_rm_degeneracy_1a(
    t_unit=t_unit,
    xs=f_ni ** (1 / 3),    # source radius / ejecta radius
    z_off=0.1,             # source displacement / ejecta radius
    Mni56=f_ni * M,
    tdep=None,             # full gamma-ray trapping
    tmax=80 * DAY,
)

for mu, L in zip(cos_view, L_view):
    i = np.argmax(L)
    print(f"cos(theta) = {mu:+.1f}:  t_peak = {t[i] / DAY:5.2f} d,  "
          f"L_peak = {L[i]:.3e} erg/s")
```

The observer facing the source sees an earlier, brighter peak: approximately
14.0 days and 1.68e43 erg/s at `cos(theta) = +0.9`, compared with 16.7 days and
1.09e43 erg/s at `-0.9`. Early negative luminosities indicate an unconverged
mode sum; see [Resolution and limitations](#resolution-and-limitations).

## Inputs and outputs

**YAML uses days for `tdep` and `tmax`; Python uses seconds.** Both interfaces
use grams for `Mni56`, s² for the diffusion scale, and erg/s for luminosity.
Saved times and deposition-table times are always in seconds. Here `v_o` is
the outer ejecta velocity (the equal-volume reference velocity for slightly-oblate
ejecta), and `D` is the diffusion coefficient in velocity space.

| YAML name | Python name | Meaning |
| --- | --- | --- |
| `tbar` | `t_unit` | Diffusion scale: `v_o²/D` for spherical and slightly-oblate ejecta; `a²/D` for full-oblate ejecta, with velocity-space focal length `a` |
| `xs` | `xs` | Source radius / reference ejecta radius |
| `z_ig` | `z_off` | Axial source displacement / ejecta radius |
| `mu_s`, `mu_1` | `xi_s`, `xi1` | Ejecta and source oblate coordinates |
| `f` | `f` | Source / ejecta focal-length ratio |
| `eps`, `eps_s` | `eps`, `eps_s` | Ejecta and source quadrupole distortions; negative means oblate |
| `cos_theta_edges` | `obs_cos_edges` | Viewing-bin edges in `cos(theta)` |

The Python functions return `(t, L_total, L_view, cos_view)`. `L_total` is
angle-integrated; each row of `L_view` is an isotropic-equivalent luminosity.
By default (`observer: centre`), curves are evaluated at the centres of ten
bins uniform in `cos(theta)`, from `-0.9` to `+0.9`; all analytic results in the
paper use this choice. Use `observer: bin` (Python `observer="bin"`) to average
each curve over its bin instead.

The NPZ contains `t`, `lc_1D` (= `L_total`), `lc_2D` (= `L_view`), and
`cos_theta_edges`. With YAML `cos_theta_edges: null`, only `t` and `lc_1D`
are saved. The Python equivalent is `obs_cos_edges=None`.

### Gamma-ray deposition

| Deposition model | YAML settings | Python arguments |
| --- | --- | --- |
| Full trapping | `tdep: null`, `fdep_file: null` | `tdep=None`, `fdep_file=None` (default) |
| Escape law `1-exp(-(tdep/t)²)` | `tdep: 40.0`, `fdep_file: null` | `tdep=40*DAY` |
| Tabulated fraction | `tdep: null`, `fdep_file: fdep.txt` | `fdep_file="fdep.txt"` with `tdep=None` |

Omitting `tdep` gives full trapping, in YAML and in Python. A deposition table has two whitespace-separated columns: time
[s] and gamma-ray deposition fraction. Positron heating remains fully trapped.
See the [solver reference](docs/solver_reference.md) for interpolation rules,
geometry definitions, additional Python examples, and advanced options.

## Resolution and limitations

The analytic light curves in the paper and the analytic entries in its
supplementary CSV tables use these default settings:

| Geometry | Spatial modes | Time integration |
| --- | --- | --- |
| Spherical | `lmax=100, nmax=100` | `Nt=8000, oversample=32` |
| Oblate | `nmax=100, pmax=200, nleg=None` | `Nt=8000, oversample=32` |
| Slightly oblate | `nmax=200` roots in each of `l=0,2` | `Nt=8000, oversample=32` |

Times are uniform in `t²`; `Nt=8000` returns 7999 samples, starting at zero
and omitting the final node at `tmax`. Check convergence for each new geometry
by increasing the mode counts, `Nt`, and `oversample` until the curves are
stable over the interval of interest. The earliest rise, before a few days
for the paper examples, can require more modes. Negative
luminosities trigger a warning and require a convergence check before use.

Spherical sources must satisfy `xs > 0` and `xs + abs(z_off) <= 1`; full-oblate
sources must fit inside the ejecta. The slightly-oblate solver is valid to
first order in small `eps` and `eps_s`. Detailed validity conditions and
numerical controls are in the [solver reference](docs/solver_reference.md).

## Citation

Please cite the accompanying paper when using this code:

D. van der Boom and D. Kushnir, *An analytic diffusion model for viewing-angle
effects in explosive transients*, arXiv:2611.15905
([arxiv.org/abs/2611.15905](https://arxiv.org/abs/2611.15905)).

```bibtex
@misc{vanderBoom2026,
      title={An analytic diffusion model for viewing-angle effects in explosive transients},
      author={Daniella van der Boom and Doron Kushnir},
      year={2026},
      eprint={2611.15905},
      archivePrefix={arXiv},
      primaryClass={astro-ph.HE},
      url={https://arxiv.org/abs/2611.15905},
}
```

Also identify the code version or commit used for your calculation; the paper's
calculations used release v1.0.
The Monte Carlo radiative-transfer (URILIGHT) light curves used for comparison
in the paper are not part of this repository.

## License

MIT; see [LICENSE](LICENSE).
