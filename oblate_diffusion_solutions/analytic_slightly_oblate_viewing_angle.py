"""First-order diffusion about a sphere with quadrupole-distorted surfaces.

In equal-volume ejecta-radius units the surfaces are
    r_b(mu) = 1 + eps*P_2(mu),
    r_src(mu) = xs*(1 + eps_s*P_2(mu)).
Oblate distortions have negative eps. This is the time-dependent first-order
problem, with only l=0 and l=2 modes. It requires neither spheroidal functions
nor a quasi-static approximation. The diffusion scale is t_unit=v_o^2/D [s^2].

Let K_beta[g] be convolution with exp(-beta*tau), and H_beta[g]=g-beta*K_beta[g].
H is evaluated through increments of g, avoiding subtraction of large terms.
With b=eps*F0 and w_n=Csrc_n*alpha_n*j_3(alpha_n), the quadrupole is
    F2 = -4*b + 3*eps_s*xs^2*Q/(4*pi)
         - sum_n (w_n/beta_n)*H_beta_n[Q]
         - 2*(sum_n H_beta_n[b] + tail*db/dtau),
where tail=1/14-sum_n 1/beta_n is the part of the l=2 sum rule
sum_n 1/beta_n = 1/14 carried by the roots beyond nmax.
All time responses use piecewise-linear forcing and summed kernels, with FFT
convolution on long grids; no mode-by-time solution arrays are stored.
"""

import warnings

import numpy as np
from scipy.signal import fftconvolve

from _diffusion_common import (DAY, MSUN, build_kernels, convolve,
                                heating_ni56, observer_cosines, sph_jn_all)
from spherical_diffusion_solutions.analytic_spherical_off_center_viewing_angle import (
    bessel_zeros,
)

DEFAULT_EDGES = np.linspace(-1.0, 1.0, 11)
W2 = 13.0 / 40.0


def eps_from_delta(delta):
    """First-order boundary distortion for flattening delta=1-v_pol/v_eq."""
    return -2.0 * delta / 3.0


def eps_s_from_f_q(f, q, xs):
    """Source distortion satisfying 3*eps_s*xs^2=-f^2*(1-q^2)."""
    return -f**2 * (1.0 - q**2) / (3.0 * xs**2)


def quasi_static_multipole(xs, eps, eps_s):
    """Return the quasi-static (F2/F0, Phi2/Phi0); Phi2/Phi0 is X."""
    a2 = 3.0 * eps_s * xs**2 - 4.0 * eps
    return a2, a2 - 4.0 * eps


def _complement_kernels(beta, weights, h, count):
    """Kernels for sum(weights*H_beta[g]), exact for linear g per interval.

    H_beta[g]_k = exp(-beta*k*h)*g_0
        + sum_{i=1}^k (g_i-g_{i-1})*exp(-beta*(k-i)*h)*(1-exp(-beta*h))/(beta*h).
    The two outputs multiply increments and the initial value, respectively.
    """
    weights = np.atleast_2d(weights)
    order = np.argsort(beta)
    beta, weights = beta[order], weights[:, order]
    step = beta * h
    ramp = -np.expm1(-step) / step
    kernel = np.zeros((weights.shape[0], count))
    initial = np.zeros_like(kernel)
    chunk = max(1, int(2e6 // count))
    for first in range(0, beta.size, chunk):
        b = beta[first:first + chunk]
        stop = min(count, int(np.ceil(40.0 / (b[0] * h))) + 2)
        exponential = np.exp(-np.outer(b, np.arange(stop) * h))
        w = weights[:, first:first + chunk]
        kernel[:, :stop] += (w * ramp[first:first + chunk]) @ exponential
        initial[:, :stop] += w @ exponential
    return kernel, initial


def _apply_complement(kernel, initial, g):
    increments = np.diff(g, prepend=g[0])
    method = fftconvolve if g.size > 12000 else np.convolve
    return np.array([method(increments, row)[:g.size] + g[0] * start
                     for row, start in zip(kernel, initial)])


class SlightlyOblate:
    """Reusable l=0,2 modes for a centred, slightly aspherical source/ejecta.

    xs is the source radius divided by the ejecta radius in the spherical
    reference geometry, with 0<xs<=1. eps and eps_s are the boundary and source
    P_2 amplitudes. Both should be small compared with one. nmax is the number
    of radial roots in each sector; analytical corrections accelerate the l=2
    sums. Large distortions or crossing surfaces emit a warning because the
    formal first-order expansion can still be evaluated outside its physical
    range. A geometry object reuses its spatial modes across heating calls.
    """

    def __init__(self, xs, eps, eps_s, nmax=200):
        if not np.all(np.isfinite([xs, eps, eps_s])) or not 0 < xs <= 1:
            raise ValueError("geometry must be finite with 0 < xs <= 1")
        if not isinstance(nmax, (int, np.integer)) or nmax < 1:
            raise ValueError("nmax must be an integer >= 1")
        if max(abs(eps), abs(eps_s)) > 0.35:
            warnings.warn("Large distortion in first-order theory; check against "
                          "the full-oblate solution.", RuntimeWarning, stacklevel=2)
        p2 = np.array([-0.5, 1.0])
        outer, source = 1.0 + eps*p2, xs*(1.0 + eps_s*p2)
        if np.any(outer <= 0) or np.any(source <= 0) or np.any(source > outer + 1e-12):
            warnings.warn("Distorted surfaces cross or have nonpositive radius; "
                          "the formal first-order result may be unphysical.",
                          RuntimeWarning, stacklevel=2)
        self.xs, self.eps, self.eps_s, self.nmax = xs, eps, eps_s, nmax
        self.numerical_settings = dict(nmax=nmax)
        a0, a2 = bessel_zeros(2, nmax)[[0, 2]]
        self.alpha0, self.alpha2 = a0, a2
        self.beta0, self.beta2 = a0**2, a2**2
        j1 = sph_jn_all(1, a0)[1]
        self.weight0 = (3.0*sph_jn_all(1, a0*xs)[1]/(a0*xs)
                        * a0/(2.0*np.pi*j1))
        j3 = sph_jn_all(3, a2)[3]
        self.source_weights = 3.0*eps_s*sph_jn_all(2, a2*xs)[2]/(2.0*np.pi*a2*j3)
        self.source_static = 3.0*eps_s*xs**2/(4.0*np.pi)
        self.tail = max(1.0/14.0 - np.sum(1.0/self.beta2), 0.0)

    @property
    def energy_sum_rule(self):
        """Late-time L/Q of the truncated monopole; tends to one."""
        return float(4.0*np.pi*np.sum(self.weight0/self.beta0))

    def _flux_multipoles(self, Q, h):
        """Fluxes for supplied heating Q on a uniform tau grid, initially empty."""
        count = Q.size
        W, W0 = build_kernels(self.beta0, self.weight0, h, count,
                              chunk=max(1, int(2e6 // count)))
        F0 = convolve(W, W0, Q, method="fft" if count > 12000 else "direct")[0]
        F0[0] = 0.0
        F2 = np.zeros_like(F0)
        if self.eps_s:
            W, initial = _complement_kernels(self.beta2, self.source_weights, h, count)
            F2 += self.source_static*Q - _apply_complement(W, initial, Q)[0]
        if self.eps:
            b = self.eps*F0
            W, initial = _complement_kernels(self.beta2, np.ones(self.nmax), h, count)
            boundary = _apply_complement(W, initial, b)[0]
            W, initial = _complement_kernels(self.beta0, self.weight0, h, count)
            db = self.eps*_apply_complement(W, initial, Q)[0]
            F2 -= 4.0*b + 2.0*(boundary + self.tail*db)
        # The finite source-tail closure has an instantaneous residual at tau=0;
        # the physical initial condition is zero luminosity in every direction.
        F2[0] = 0.0
        return F0, F2

    def light_curve(self, t_unit, Mni56, tdep=None, fdep_file=None,
                    tmax=80*DAY, Nt=8000, obs_cos_edges=DEFAULT_EDGES,
                    observer="centre", oversample=32, n_bin=8,
                    return_multipoles=False):
        """Return (t, L_total, L_view, cos_view), optionally followed by multipoles.

        Times are seconds, t_unit is v_o^2/D [s^2], Mni56 is grams, and luminosity
        is erg/s. tdep or fdep_file controls gamma deposition; positrons remain
        trapped. Time sampling, defaults and shapes match the spherical
        and oblate solvers: Nt nodes uniform in t^2, the final node omitted, and oversample
        internal intervals per output interval. obs_cos_edges=None requests
        total luminosity only. observer='centre' evaluates P_2 at bin centres;
        observer='bin' averages it analytically (independent of n_bin).

        return_multipoles=True appends a dict with t, tau, Q, F0, F2, a2=F2/F0,
        Phi2_over_Phi0=a2-4*eps, L_norm=L_total, and tail/grid diagnostics.
        All time arrays have the same Nt-1 samples as the light curves. F0 and
        F2 include the physical heating normalization. a2 is zero where F0 is
        nonpositive.
        """
        for name, value in [("t_unit", t_unit), ("tmax", tmax)]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(Mni56) or Mni56 < 0:
            raise ValueError("Mni56 must be finite and nonnegative")
        if tdep is not None and (not np.isfinite(tdep) or tdep <= 0):
            raise ValueError("tdep must be finite and positive, or None")
        if tdep is not None and fdep_file is not None:
            raise ValueError("specify either fdep_file or tdep, not both")
        for name, value, minimum in [("Nt", Nt, 2), ("oversample", oversample, 1)]:
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        centres, moment = None, None
        if obs_cos_edges is not None:
            _, centres = observer_cosines(obs_cos_edges, observer, n_bin=n_bin)
            if observer == "centre":
                moment = (3.0*centres**2 - 1.0)/2.0
            else:
                lo, hi = np.asarray(obs_cos_edges)[:-1], np.asarray(obs_cos_edges)[1:]
                moment = (lo*lo + lo*hi + hi*hi - 1.0)/2.0
        count = (Nt - 1)*oversample + 1
        tau = np.linspace(0.0, 0.5*tmax**2/t_unit, count)
        t = np.sqrt(2.0*tau*t_unit)
        Q = heating_ni56(t, Mni56, t_unit=t_unit, tdep=tdep, fdep_file=fdep_file)
        if not np.all(np.isfinite(Q)):
            raise ValueError("Heating must be finite at every internal time")
        F0, F2 = self._flux_multipoles(Q, tau[1] - tau[0])
        select = slice(0, -1, oversample)
        t, tau, Q, F0, F2 = (a[select] for a in (t, tau, Q, F0, F2))
        L_total = 4.0*np.pi*F0
        # Project the multipole itself, avoiding division by small F0 on the rise.
        Phi2 = F2 - 4.0*self.eps*F0
        L_view = (None if moment is None else
                  L_total[None, :] + 4.0*np.pi*W2*moment[:, None]*Phi2[None, :])
        curves = L_total[None, :] if L_view is None else np.vstack([L_total, L_view])
        if not np.all(np.isfinite(curves)):
            raise RuntimeError("Non-finite luminosity; increase the numerical resolution")
        if np.any(curves < -1e-10*np.max(np.abs(curves), axis=1, keepdims=True)):
            warnings.warn("Negative luminosity exceeds roundoff. Increase resolution; "
                          "large distortions can also invalidate first-order theory.",
                          RuntimeWarning, stacklevel=2)
        out = (t, L_total, L_view, centres)
        if not return_multipoles:
            return out
        a2 = np.divide(F2, F0, out=np.zeros_like(F0), where=F0 > 0)
        multipoles = dict(t=t, tau=tau, Q=Q, F0=F0, F2=F2, a2=a2,
                          Phi2_over_Phi0=a2 - 4.0*self.eps, L_norm=L_total,
                          nmax_tail_residual=self.tail, n_tau_internal=count)
        return out + (multipoles,)


def light_curve_slightly_oblate_viewing_angle_rm_degeneracy_1a(
        t_unit, xs, eps, eps_s, Mni56, tdep=None, tmax=80*DAY, Nt=8000,
        nmax=200, obs_cos_edges=DEFAULT_EDGES, fdep_file=None,
        observer="centre", oversample=32, n_bin=8, verbose=False,
        return_multipoles=False):
    """56Ni/56Co light curves for first-order boundary and source distortions.

    xs, eps, eps_s define r_b=1+eps*P_2 and r_src=xs*(1+eps_s*P_2).
    t_unit=v_o^2/D [s^2] uses the equal-volume reference velocity, Mni56 is grams,
    other times are seconds, and luminosities are erg/s. nmax counts radial
    roots in each of l=0,2. See SlightlyOblate.light_curve for all time, observer,
    deposition and return_multipoles options. Returns t, L_total, L_view,
    cos_view, optionally followed by the multipole dict.
    """
    geom = SlightlyOblate(xs, eps, eps_s, nmax=nmax)
    if verbose:
        print(f"Slightly-oblate geometry: {2*nmax} modes; "
              f"quasi-static X={quasi_static_multipole(xs, eps, eps_s)[1]:+.6f}")
    return geom.light_curve(t_unit, Mni56, tdep=tdep, fdep_file=fdep_file,
                            tmax=tmax, Nt=Nt, obs_cos_edges=obs_cos_edges,
                            observer=observer, oversample=oversample, n_bin=n_bin,
                            return_multipoles=return_multipoles)
