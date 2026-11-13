# Solver reference

See the [README](../README.md) for installation, a quick start, units, and
output definitions. The Python solvers use seconds for times, grams for masses,
s² for `t_unit`, and erg/s for luminosities.

## Running without installation

From a directory containing your configuration, give the path to the runner:

```sh
python /path/to/viewing-angle-lc/run_experiment.py --config_path experiment_params_cfg.yaml
```

NumPy, SciPy, and PyYAML must already be installed in the Python environment.
Relative input and output paths refer to the working directory.

## Geometry and Python interfaces

### Off-centre sphere

```python
from spherical_diffusion_solutions import (
    DAY, MSUN, light_curve_offcenter_viewing_angle_rm_degeneracy_1a,
)

t, L_total, L_view, cos_view = light_curve_offcenter_viewing_angle_rm_degeneracy_1a(
    t_unit=7.408376170266691e12,
    xs=0.8435498098224316, z_off=0.09985858534149691,
    Mni56=0.604 * MSUN,
    tmax=80 * DAY,
)
```

Here `xs` and `z_off` are the source radius and axial displacement divided
by the ejecta radius `R`; require `xs > 0` and `xs + abs(z_off) <= 1`.
`allow_outside=True` (YAML `allow_outside: true`) accepts a source that sticks
out of the ejecta, with a warning: the closed-form
projection is evaluated for the full source ball and the heating is still
normalised by its full volume, so the energy deposited outside the ejecta is
missing from `L_total`.
The diffusion scale uses `v_o²/D`, where `v_o` is the outer ejecta velocity
and `D` is the diffusion coefficient in velocity space.

### Oblate solution

```python
from oblate_diffusion_solutions import (
    DAY, MSUN, light_curve_oblate_viewing_angle_rm_degeneracy_1a,
)

t, L_total, L_view, cos_view = light_curve_oblate_viewing_angle_rm_degeneracy_1a(
    t_unit=2.04192369390989e12,
    xi_s=1.3016357335912518, xi1=1.0383933866388468,
    f=0.8377565625725295, Mni56=0.3321340527707114 * MSUN,
    tdep=39.967429931442005 * DAY,
)
```

The ejecta equatorial/polar semi-axes are `a*cosh(xi_s)`, `a*sinh(xi_s)`;
the source semi-axes are `f*a*cosh(xi1)`, `f*a*sinh(xi1)`. Both source axes
must fit inside the ejecta, with `xi_s`, `xi1`, and `f` positive.
The diffusion scale uses `a²/D`, with ejecta focal length `a` in velocity space.

### Slightly-oblate solution

```python
from oblate_diffusion_solutions import (
    DAY, MSUN,
    light_curve_slightly_oblate_viewing_angle_rm_degeneracy_1a,
)

t, L_total, L_view, cos_view = (
    light_curve_slightly_oblate_viewing_angle_rm_degeneracy_1a(
        t_unit=5.96e12, xs=0.8435, eps=-0.06, eps_s=-0.03,
        Mni56=0.5*MSUN, tdep=35*DAY,
    )
)
```

This solves the time-dependent diffusion problem to **first order** in the
quadrupole distortions of the ejecta and source surfaces:

```text
v_boundary(mu) = v_o * (1 + eps * P2(mu))
v_source(mu)   = v_o * xs * (1 + eps_s * P2(mu))
P2(mu)        = (3*mu**2 - 1)/2
```

`v_o` is the equal-volume reference ejecta velocity, `0 < xs <= 1`, and negative
`eps` or `eps_s` means oblate. The diffusion scale is **`v_o²/D`**, in s²;
it differs from the focal-length scale of the full-oblate solver. Both
distortions should be small compared with one. Large distortions and crossing
surfaces emit warnings; the formal first-order expansion can still be evaluated.

Only the monopole and quadrupole are needed. The angle-integrated luminosity
is the centred spherical solution at first order. The observer luminosity is

```text
L_view = L_total + 4*pi*(13/40)*(F2 - 4*eps*F0)*P2(cos_view)
L_total = 4*pi*F0
```

`observer="bin"` averages `P2` analytically over each bin. The quasi-static
limit is `Phi2/Phi0 = 3*eps_s*xs**2 - 8*eps`; the calculation retains the full
time dependence.

### Returned light curves

`L_total` is angle-integrated. Each row of `L_view` is an isotropic-equivalent
luminosity, ordered by `cos_view`. The default is ten bin centres from
`cos(theta)=-0.9` to `0.9`. Supply `obs_cos_edges` for other bins,
`observer="bin"` for bin averages, or `obs_cos_edges=None` for total luminosity only.
For `observer="centre"`, edges may extend beyond `[-1,1]` if their centres are
valid directions (e.g. bins centred on the poles). Bin averages
require all edges within `[-1,1]`.

## Numerical controls

At the default resolution, spherical `nmax` counts radial roots for each degree
`0..lmax`. Oblate `nmax` includes even angular indices `0,2,...,100`, with
`pmax` radial roots each. The oblate solver grows its Legendre basis automatically;
the default surface orders are `n_src=400, n_norm=800, n_split=160`.

The output nodes are uniform in `t²`. `Nt=8000` spans zero to `tmax`, with
the last node omitted, giving 7999 samples. Each interval has 32 internal
intervals (255969 internal nodes). Output values are selected directly from
these nodes. Radioactive heating is piecewise linear in `t²` internally;
exponential mode responses are integrated analytically on each interval.

In YAML, omitted or null `nmax`, `lmax`, and `pmax` select the relevant
solver defaults in the [README](../README.md#resolution-and-limitations).
`oversample`, `observer`, `n_bin`, and the oblate controls `nleg`, `n_src`,
`n_norm`, `n_split`, and `scan_per_spacing` can also be set in YAML.
The full configuration dataclass is in [run_experiment.py](../run_experiment.py).

## Deposition tables

Tables need at least two distinct positive times. Rows are sorted, nonpositive
times are discarded, and fractions are clipped to `[0,1]`. Interpolation is
linear in `t²`, holding endpoint fractions outside the tabulated range.
The fraction multiplies gamma-ray heating; positron heating remains fully trapped.

## Reusing spatial modes

For repeated heating calculations with the same geometry, construct
`OffCenterSphere(...)`, `OblateSpheroids(...)`, or `SlightlyOblate(...)` once
and call its `light_curve(t_unit=..., Mni56=..., ...)` method. This reuses
the spatial modes. Function docstrings describe all parameters and array shapes.

## Source layout

- [spherical_diffusion_solutions](../spherical_diffusion_solutions): off-centre spherical solver.
- [oblate_diffusion_solutions](../oblate_diffusion_solutions): full-oblate and slightly-oblate solvers.
- [_diffusion_common.py](../_diffusion_common.py): shared heating, observer, and time-integration routines.
