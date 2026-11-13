"""Numerical tools for axisymmetric diffusion light curves.

In tau=t^2/(2*t_unit), each spatial mode obeys
    dA/dtau = -beta*A + C*Q(tau).
Surface luminosities are linear combinations of these amplitudes. The sum of
exponential responses is integrated exactly against piecewise-linear heating
on a uniform tau grid, then convolved using an FFT on long grids.

Spherical Bessel functions use Miller recurrence with small-argument series.
The observer kernel integrates an Eddington grey atmosphere over visible
surface normals. Radioactive power is the 56Ni -> 56Co -> 56Fe chain, including
fully deposited positron heating (Nadyozhin 1994; Wygoda et al. 2019).
"""

import warnings

import numpy as np
from scipy.signal import fftconvolve

DAY = 86400.0
MSUN = 1.989e33


def _start_order(x):
    return np.ceil(x + 8.0 * np.cbrt(x) + 25.0).astype(int)


def sph_jn_all(kmax, x, derivative=False):
    """
    j_k(x) for k = 0..kmax at every element of x.

    Parameters
    ----------
    kmax : int
    x    : array_like, x >= 0
    derivative : if True also return j_k'(x)

    Returns
    -------
    J  : ndarray, shape (kmax+1,) + x.shape
    dJ : same shape (only if derivative=True)
    """
    x = np.asarray(x, dtype=float)
    shape = x.shape
    x = x.ravel()
    if np.any(~np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("sph_jn_all: x must be finite and >= 0")
    if not isinstance(kmax, (int, np.integer)) or kmax < 0:
        raise ValueError("kmax must be a nonnegative integer")
    n = x.size
    J = np.zeros((kmax + 2, n))          # one extra order for the derivative

    tiny = x < 1e-6
    xs = np.where(tiny, 1.0, x)          # placeholder, overwritten below

    # start above both the turning point and kmax: for x << kmax the high orders
    # are tiny but still needed (relative accuracy), e.g. j_50(0.3); overflow on
    # the way down is handled by the rescaling below
    N = np.maximum(_start_order(xs), kmax + 15)
    Ntop = int(N.max()) if n else 0
    jp1 = np.zeros(n)                    # j~_{k+1}
    jk = np.zeros(n)                     # j~_k
    inv_x = 1.0 / xs
    seed = 1e-100
    for k in range(Ntop, 0, -1):
        start = N == k
        if start.any():
            jk = np.where(start, seed, jk)
        if k <= kmax + 1:
            J[k] = jk
        jm1 = (2 * k + 1) * inv_x * jk - jp1
        jp1, jk = jk, jm1
        if (k & 7) == 0:
            big = np.abs(jk) > 1e150
            if big.any():
                s = np.where(big, 1e-150, 1.0)
                jk = jk * s
                jp1 = jp1 * s
                J[k:] *= s
    J[0] = jk

    # normalise with the exact j_0, j_1 (least squares: they never vanish together)
    sx, cx = np.sin(xs), np.cos(xs)
    j0 = sx * inv_x
    j1 = sx * inv_x**2 - cx * inv_x
    m = np.maximum(np.abs(J[0]), np.abs(J[1]))
    a, b = J[0] / m, J[1] / m
    scale = (j0 * a + j1 * b) / ((a * a + b * b) * m)
    J *= scale

    if tiny.any():
        # j_k = x^k/(2k+1)!! [1-x^2/(2(2k+3))+O(x^4)].
        # Keep every representable order, including at very small nonzero x.
        xt = x[tiny]
        leading = np.ones_like(xt)
        for k in range(kmax + 2):
            if k:
                leading = leading * xt / (2 * k + 1)
            J[k, tiny] = leading * (1.0 - xt**2 / (2.0 * (2 * k + 3)))

    if not derivative:
        return J[:kmax + 1].reshape((kmax + 1,) + shape)

    dJ = np.empty((kmax + 1, n))
    dJ[0] = -J[1]
    if kmax >= 1:
        k = np.arange(1, kmax + 1)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            dJ[1:] = J[:kmax] - (k + 1) * J[1:kmax + 1] / xs
        if tiny.any():
            # Differentiate the series directly: no division by x at zero,
            # and no loss from underflowing j_k before dividing by x.
            leading_der = np.ones_like(xt) / 3.0
            for k in range(1, kmax + 1):
                if k > 1:
                    leading_der = leading_der * xt / (2 * k + 1)
                dJ[k, tiny] = leading_der * (
                    k - (k + 2) * xt**2 / (2.0 * (2 * k + 3)))
    return (J[:kmax + 1].reshape((kmax + 1,) + shape),
            dJ.reshape((kmax + 1,) + shape))


def legendre_all(kmax, u, derivative=False):
    """
    P_k(u) for k = 0..kmax (and P_k'(u) if derivative=True), shape (kmax+1,)+u.shape.

    The derivative uses P_k' = P_{k-2}' + (2k-1) P_{k-1}, which has no
    1/(1-u^2) and is therefore accurate up to u = +-1.
    """
    u = np.asarray(u, dtype=float)
    P = np.empty((kmax + 1,) + u.shape)
    P[0] = 1.0
    if kmax >= 1:
        P[1] = u
    for k in range(2, kmax + 1):
        P[k] = ((2 * k - 1) * u * P[k - 1] - (k - 1) * P[k - 2]) / k
    if not derivative:
        return P
    dP = np.zeros_like(P)
    if kmax >= 1:
        dP[1] = 1.0
    for k in range(2, kmax + 1):
        dP[k] = dP[k - 2] + (2 * k - 1) * P[k - 1]
    return P, dP


def gauss_legendre(a, b, n):
    """n-point Gauss-Legendre nodes and weights on [a, b]."""
    x, w = np.polynomial.legendre.leggauss(n)
    h = 0.5 * (b - a)
    return h * x + 0.5 * (a + b), h * w


def phi_integral(A, B):
    """Vectorised W(A, B); A >= 0."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    A, B = np.broadcast_arrays(A, B)

    def F(p):
        return (A**2 * (0.5 * p + 0.25 * np.sin(2.0 * p))
                + (2.0 * A * B + 2.0 * A / 3.0) * np.sin(p)
                + (B**2 + 2.0 * B / 3.0) * p)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(A > 1e-30, -B / np.where(A > 1e-30, A, 1.0), 0.0)
    plim = np.where(ratio <= -1.0, np.pi,
                    np.where(ratio >= 1.0, 0.0, np.arccos(np.clip(ratio, -1, 1))))
    W = 2.0 * F(plim)                        # F is odd in p
    noA = A <= 1e-30
    W = np.where(noA, np.where(B > 0.0, 2.0 * np.pi * B * (B + 2.0 / 3.0), 0.0), W)
    return W


def observer_cosines(obs_cos_edges, observer="centre", n_bin=8):
    """
    Observer directions and weights for each viewing-angle bin.

    observer = 'centre' : the bin centre
    observer = 'bin'    : a Gauss average over the bin in cos(theta_o), the
                          quantity a Monte Carlo code collects in that bin.

    Returns a list of (cos_nodes, weights) per bin, weights summing to 1, and
    the bin centres.
    """
    e = np.asarray(obs_cos_edges, dtype=float)
    if (e.ndim != 1 or e.size < 2 or not np.all(np.isfinite(e)) or
            np.any(np.diff(e) <= 0)):
        raise ValueError("obs_cos_edges must be increasing, finite edges")
    if observer not in ("centre", "bin"):
        raise ValueError("observer must be 'centre' or 'bin'")
    if not isinstance(n_bin, (int, np.integer)) or n_bin < 1:
        raise ValueError("n_bin must be an integer >= 1")
    centres = 0.5 * (e[:-1] + e[1:])
    if observer == "bin" and np.any(np.abs(e) > 1):
        raise ValueError("observer='bin' requires edges within [-1, 1]")
    if np.any(np.abs(centres) > 1 + 1e-12):
        raise ValueError("observer bin centres must be within [-1, 1]")
    centres = np.clip(centres, -1.0, 1.0)
    out = []
    for lo, hi, c in zip(e[:-1], e[1:], centres):
        if observer == "centre":
            out.append((np.array([c]), np.array([1.0])))
        elif observer == "bin":
            x, w = gauss_legendre(lo, hi, n_bin)
            out.append((x, w / w.sum()))
        else:
            raise ValueError("observer must be 'centre' or 'bin'")
    return out, centres


def consts_ni56():
    """c1, c2, c3 [erg/s per Msun of 56Ni] and the e-folding times tau1, tau2 [s]."""
    c1 = 6.45e43
    c2 = 1.38e43
    c3 = 4.64e41
    tau1 = 6.075 * DAY / np.log(2.0)
    tau2 = 77.233 * DAY / np.log(2.0)
    return c1, c2, c3, tau1, tau2


def load_fdep_table(fdep_file):
    """Read a (t [s], f_dep) table; returns sorted t > 0 and f_dep clipped to [0, 1]."""
    data = np.loadtxt(fdep_file)
    if data.ndim < 2 or data.shape[1] < 2:
        raise ValueError(f"f_dep file '{fdep_file}' must have >= 2 rows of "
                         f"2 columns (t [s], f_dep)")
    t, f = data[:, 0].astype(float), data[:, 1].astype(float)
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(f)):
        raise ValueError("f_dep table times and fractions must be finite")
    good = t > 0.0
    t, f = t[good], f[good]
    if t.size < 2:
        raise ValueError(f"f_dep file '{fdep_file}' has < 2 rows with t > 0")
    order = np.argsort(t)
    if np.any(np.diff(t[order]) <= 0):
        raise ValueError("f_dep table must have distinct positive times")
    return t[order], np.clip(f[order], 0.0, 1.0)


def fdep_of_t(t, t_unit, tdep=None, fdep_file=None):
    """Gamma-ray deposition fraction at times t [s]."""
    if tdep is not None and fdep_file is not None:
        raise ValueError("specify either fdep_file or tdep, not both")
    t = np.asarray(t, dtype=float)
    if fdep_file is not None:
        tt, ff = load_fdep_table(fdep_file)
        # Interpolate linearly in the diffusion time tau = t^2/(2*t_unit).
        return np.interp(0.5 * t**2 / t_unit, 0.5 * tt**2 / t_unit, ff)
    if tdep is None:
        return np.ones_like(t)
    with np.errstate(divide="ignore", over="ignore"):
        return np.where(t > 0.0, -np.expm1(-(tdep / np.where(t > 0, t, 1.0))**2), 1.0)


def heating_ni56(t, Mni56, t_unit=1.0, tdep=None, fdep_file=None):
    """Deposited power Q_dep(t) [erg/s] at times t [s] for Mni56 [g] of 56Ni."""
    c1, c2, c3, tau1, tau2 = consts_ni56()
    s = Mni56 / MSUN
    t = np.asarray(t, dtype=float)
    e1, e2 = np.exp(-t / tau1), np.exp(-t / tau2)
    fd = fdep_of_t(t, t_unit, tdep=tdep, fdep_file=fdep_file)
    return s * (fd * (c1 * e1 + c2 * e2) + c3 * (e2 - e1))


def _phi_L(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    s = x < 1e-2
    xs = x[s]
    out[s] = 0.5 - xs / 6.0 + xs**2 / 24.0 - xs**3 / 120.0 + xs**4 / 720.0
    xl = x[~s]
    out[~s] = (xl + np.expm1(-xl)) / xl**2
    return out


def _phi_R(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    s = x < 1e-2
    xs = x[s]
    out[s] = 0.5 - xs / 3.0 + xs**2 / 8.0 - xs**3 / 30.0 + xs**4 / 144.0
    xl = x[~s]
    out[~s] = (-np.expm1(-xl) - xl * np.exp(-xl)) / xl**2
    return out


def build_kernels(beta, weights, h, Nt, chunk=256, cutoff=40.0):
    """
    Impulse-response kernels W (interior nodes) and W0 (node i = 0).

    Parameters
    ----------
    beta    : (Nm,) decay rates in tau (> 0)
    weights : (Nout, Nm) g_jm * C_m
    h       : grid step in tau
    Nt      : number of grid points
    cutoff  : a mode is dropped beyond beta*tau > cutoff (e^-40 = 4e-18)

    Returns
    -------
    W, W0 : (Nout, Nt)
    """
    beta = np.asarray(beta, dtype=float)
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    order = np.argsort(beta)
    beta = beta[order]
    weights = weights[:, order]
    Nout = weights.shape[0]
    PL = np.zeros((Nout, Nt))
    PR = np.zeros((Nout, Nt))
    n = np.arange(Nt) * h
    x = beta * h
    wL = weights * (h * _phi_L(x))
    wR = weights * (h * _phi_R(x))
    for i0 in range(0, beta.size, chunk):
        b = beta[i0:i0 + chunk]
        ncut = int(min(Nt, np.ceil(cutoff / (b[0] * h)) + 2))
        E = np.exp(-np.outer(b, n[:ncut]))
        PL[:, :ncut] += wL[:, i0:i0 + chunk] @ E
        PR[:, :ncut] += wR[:, i0:i0 + chunk] @ E
    W = PL.copy()
    W[:, 1:] += PR[:, :-1]
    W0 = np.zeros_like(PR)
    W0[:, 1:] = PR[:, :-1]
    return W, W0


def convolve(W, W0, Q, method="direct"):
    """L_k = sum_{i=1}^{k} Q_i W[k-i] + Q_0 W0[k] for every row of W."""
    Q = np.asarray(Q, dtype=float)
    Nt = Q.size
    Qp = Q.copy()
    Qp[0] = 0.0
    out = np.empty((W.shape[0], Nt))
    for j in range(W.shape[0]):
        if method == "fft":
            c = fftconvolve(Qp, W[j])[:Nt]
        elif method == "direct":
            c = np.convolve(Qp, W[j])[:Nt]
        else:
            raise ValueError("method must be 'direct' or 'fft'")
        out[j] = c + Q[0] * W0[j]
    return out


def solve(geom, t_unit, Mni56, tdep=None, fdep_file=None, tmax=80*DAY,
          Nt=8000, oversample=32, obs_cos_edges=None, observer="centre",
          observer_options=None):
    """Return (t, L_total, L_view, cos_view) for radioactive heating.

    Nt nodes span [0,tmax] uniformly in t^2; the final node is omitted from
    the returned arrays. Each output interval has `oversample` internal
    intervals. Output luminosities are selected from those internal nodes.
    t_unit has units s^2, Mni56 is in g, all times are in s, and L is in erg/s.
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
    if (np.any(~np.isfinite(geom.beta)) or np.any(geom.beta <= 0) or
            np.any(~np.isfinite(geom.B)) or np.any(~np.isfinite(geom.C))):
        raise ValueError("Non-finite or invalid spatial mode coefficients")
    rows = [np.asarray(geom.B)[None, :]]
    cos_view = None
    if obs_cos_edges is not None:
        G, cos_view = geom.observer_matrix(obs_cos_edges, observer,
                                           **dict(observer_options or {}))
        rows.append(G)
    weights = np.vstack(rows)*geom.C
    internal_points = (Nt-1)*oversample+1
    tau = np.linspace(0.0, 0.5*tmax**2/t_unit, internal_points)
    h = tau[1]-tau[0]
    t = np.sqrt(2.0*tau*t_unit)
    Q = heating_ni56(t, Mni56, t_unit=t_unit, tdep=tdep, fdep_file=fdep_file)
    if not np.all(np.isfinite(Q)):
        raise ValueError("Heating must be finite at every internal time")
    W, W0 = build_kernels(geom.beta, weights, h, internal_points,
                          chunk=max(16, int(2e6//internal_points)))
    L = convolve(W, W0, Q, method="fft" if internal_points > 12000 else "direct")
    L = L[:, ::oversample]
    t = t[::oversample]
    L[:, 0] = 0.0
    if not np.all(np.isfinite(L)):
        raise RuntimeError("Non-finite luminosity; increase the numerical resolution")
    peak = np.max(np.abs(L), axis=1, keepdims=True)
    if np.any(L < -1e-10*peak):
        warnings.warn("Negative luminosity exceeds roundoff. Increase the mode counts "
                      "and integration resolution before using these times.",
                      RuntimeWarning, stacklevel=2)
    return t[:-1], L[0, :-1], L[1:, :-1] if obs_cos_edges is not None else None, cos_view
