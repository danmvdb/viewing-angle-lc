"""Calculate a diffusion light curve from a YAML configuration file.

Usage, from the directory that should receive the output:
    python -m run_experiment --config_path experiment_params_cfg.yaml
(after `pip install .`), or without installing:
    python /path/to/viewing-angle-lc/run_experiment.py --config_path experiment_params_cfg.yaml

tbar is in s^2, Mni56 in grams, and tdep/tmax in days. Output times are in
seconds and luminosities in erg/s. Relative input and output paths refer to
the working directory. See examples/ for the three geometries.
"""

import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
import socket
import time
from typing import Optional, Tuple, get_args, get_origin

import numpy as np
import scipy
import yaml

from _diffusion_common import DAY, MSUN, observer_cosines
from spherical_diffusion_solutions import OffCenterSphere
from oblate_diffusion_solutions import OblateSpheroids, SlightlyOblate


@dataclass
class ExperimentConfig:
    """YAML fields. Geometry: sphere (xs, z_ig), oblate (mu_s, mu_1, f),
    slightly_oblate (xs, eps, eps_s)."""

    fit_type: str = "sphere"              # sphere, oblate, slightly_oblate
    url: str = "analytic_LC"              # output prefix, relative to cwd
    tbar: float = 7.1620e12               # t_unit [s^2]
    Mni56: float = 0.6*MSUN               # initial 56Ni mass [g]
    tdep: Optional[float] = None          # days; omitted/null means full trapping
    fdep_file: Optional[str] = None       # time [s], fraction table; tdep: null
    xs: Optional[float] = 0.84            # spherical reference source radius
    z_ig: Optional[float] = 0.0           # source offset for sphere
    allow_outside: bool = False           # sphere: accept xs+|z_ig|>1 with a warning
    mu_s: Optional[float] = None          # xi_s for oblate
    mu_1: Optional[float] = None          # xi1 for oblate
    f: float = 1.0                       # oblate source/ejecta focal-length ratio
    eps: Optional[float] = None           # slightly-oblate boundary distortion
    eps_s: Optional[float] = None         # slightly-oblate source distortion
    cos_theta_edges: Optional[Tuple[float, ...]] = tuple(np.linspace(-1., 1., 11))
    observer: str = "centre"             # centre or bin
    Nt: int = 8000
    tmax: float = 80.0                   # days
    oversample: int = 32
    nmax: Optional[int] = None            # null: sphere/oblate 100; slightly 200
    lmax: Optional[int] = None            # sphere: null uses 100
    pmax: Optional[int] = None            # oblate: null uses 200
    nleg: Optional[int] = None            # oblate: adaptive Legendre basis
    n_src: int = 400                     # oblate source-surface quadrature
    n_norm: int = 800                    # oblate mode-norm quadrature
    n_split: int = 160                   # oblate visible-surface quadrature
    n_bin: int = 8                       # observer-bin quadrature
    scan_per_spacing: float = 8.0        # oblate radial root scan


class ConfigLoader(yaml.SafeLoader):
    """Safe loader that also reads the !!python/tuple tag (PyYAML's default
    Dumper writes tuples with it)."""


ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def _convert(value, annotation, name):
    """Handle scientific-notation strings and optional values from YAML."""
    args = get_args(annotation)
    if value is None:
        if type(None) in args:
            return None
        raise ValueError(f"{name} cannot be null")
    if type(None) in args:
        annotation = next(a for a in args if a is not type(None))
    try:
        if get_origin(annotation) is tuple:
            if not isinstance(value, (list, tuple)):
                raise ValueError("expected a sequence")
            return tuple(float(v) for v in value)
        if annotation is bool:
            if not isinstance(value, bool):
                raise ValueError("expected true or false")
            return value
        if annotation is int:
            converted = int(value)
            if isinstance(value, bool) or float(value) != converted:
                raise ValueError("expected an integer")
            return converted
        if annotation is float:
            if isinstance(value, bool):
                raise ValueError("expected a number")
            return float(value)
        if annotation is str and isinstance(value, str):
            return value
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid {name}: {value!r} ({exc})") from exc
    raise ValueError(f"Invalid {name}: {value!r}")


def load_config(path):
    with path.open() as stream:
        data = yaml.load(stream, Loader=ConfigLoader)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping of parameter names to values")
    data.pop("provenance", None)  # Saved configurations can be run again.
    known = {field.name: field for field in fields(ExperimentConfig)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {', '.join(sorted(map(str, unknown)))}")
    return ExperimentConfig(**{name: _convert(value, known[name].type, name)
                               for name, value in data.items()})


def compute_light_curve(cfg):
    if cfg.fit_type not in ("sphere", "oblate", "slightly_oblate"):
        raise ValueError("fit_type must be 'sphere', 'oblate' or 'slightly_oblate'")
    for name, value in (("tbar", cfg.tbar), ("tmax", cfg.tmax)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(cfg.Mni56) or cfg.Mni56 < 0:
        raise ValueError("Mni56 must be finite and nonnegative")
    if cfg.tdep is not None and (not np.isfinite(cfg.tdep) or cfg.tdep <= 0):
        raise ValueError("tdep must be finite and positive, or null")
    if cfg.Nt < 2 or cfg.oversample < 1:
        raise ValueError("Nt must be >= 2 and oversample must be >= 1")
    if cfg.fdep_file is not None and cfg.tdep is not None:
        raise ValueError("fdep_file and tdep are alternatives; set tdep: null when using fdep_file")
    fdep_file = None
    if cfg.fdep_file is not None:
        fdep_file = str(Path(cfg.fdep_file).expanduser().resolve())
        if not Path(fdep_file).is_file():
            raise FileNotFoundError(f"fdep_file not found: {fdep_file}")
    if cfg.cos_theta_edges is not None:
        observer_cosines(cfg.cos_theta_edges, cfg.observer, n_bin=cfg.n_bin)
    elif cfg.observer not in ("centre", "bin"):
        raise ValueError("observer must be 'centre' or 'bin'")

    params = {} if cfg.nmax is None else dict(nmax=cfg.nmax)
    if cfg.fit_type == "sphere":
        if cfg.xs is None or cfg.z_ig is None:
            raise ValueError("sphere requires xs and z_ig")
        if cfg.lmax is not None:
            params["lmax"] = cfg.lmax
        geom = OffCenterSphere(cfg.xs, cfg.z_ig, **params,
                               allow_outside=cfg.allow_outside)
    elif cfg.fit_type == "oblate":
        if cfg.mu_s is None or cfg.mu_1 is None:
            raise ValueError("oblate requires mu_s (xi_s) and mu_1 (xi1)")
        if cfg.pmax is not None:
            params["pmax"] = cfg.pmax
        geom = OblateSpheroids(cfg.mu_s, cfg.mu_1, f=cfg.f, **params,
                               nleg=cfg.nleg, n_src=cfg.n_src, n_norm=cfg.n_norm,
                               scan_per_spacing=cfg.scan_per_spacing, verbose=True)
    else:
        if cfg.xs is None or cfg.eps is None or cfg.eps_s is None:
            raise ValueError("slightly_oblate requires xs, eps and eps_s")
        geom = SlightlyOblate(cfg.xs, cfg.eps, cfg.eps_s, **params)

    print(f"Spatial settings: {geom.numerical_settings}", flush=True)
    kwargs = dict(t_unit=cfg.tbar, Mni56=cfg.Mni56,
                  tdep=None if cfg.tdep is None else cfg.tdep*DAY,
                  fdep_file=fdep_file, tmax=cfg.tmax*DAY, Nt=cfg.Nt,
                  obs_cos_edges=cfg.cos_theta_edges, observer=cfg.observer,
                  oversample=cfg.oversample, n_bin=cfg.n_bin)
    if cfg.fit_type == "oblate":
        kwargs["n_split"] = cfg.n_split
    return geom.light_curve(**kwargs), geom.numerical_settings


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def save_light_curve(cfg, result, spatial_settings, config_path):
    t, total, view, centres = result
    # url is an output prefix; an extension, if given, is dropped.
    prefix = Path(cfg.url).expanduser()
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = prefix.with_name(prefix.name + ".npz")
    yaml_path = prefix.with_name(prefix.name + "_config.yaml")
    arrays = dict(t=t, lc_1D=total)
    if view is not None:
        arrays.update(lc_2D=view, cos_theta_edges=np.asarray(cfg.cos_theta_edges, dtype=float))
    np.savez(npz_path, **arrays)
    payload = asdict(cfg)
    payload["provenance"] = dict(
        created=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
        solver="viewing-angle-lc", solver_directory=str(Path(__file__).resolve().parent),
        config_path=str(config_path.resolve()), numpy=np.__version__, scipy=scipy.__version__,
        spatial_settings=spatial_settings,
    )
    yaml_path.write_text(yaml.safe_dump(_plain(payload), sort_keys=False))
    print(f"Saved light curves to: {npz_path.resolve()}")
    print(f"Saved config to: {yaml_path.resolve()}")
    print(f"Samples: {t.size}; viewing directions: {0 if centres is None else centres.size}")
    return npz_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config_path", type=Path, default=Path("experiment_params_cfg.yaml"),
                        help="YAML parameter file (default: experiment_params_cfg.yaml in cwd)")
    args = parser.parse_args()
    config_path = args.config_path.expanduser()
    cfg = load_config(config_path)
    if not cfg.url:
        raise ValueError("url must be a nonempty output prefix")
    print(f"Diffusion light curve: {cfg.fit_type}")
    print(f"Config: {config_path.resolve()}")
    print(f"tbar={cfg.tbar:.6g} s^2; Mni56={cfg.Mni56/MSUN:.6g} Msun; "
          f"tmax={cfg.tmax:g} d; Nt={cfg.Nt}; oversample={cfg.oversample}", flush=True)
    start = time.perf_counter()
    result, settings = compute_light_curve(cfg)
    save_light_curve(cfg, result, settings, config_path)
    print(f"Completed in {time.perf_counter()-start:.3f} s", flush=True)


if __name__ == "__main__":
    main()
