"""Diffusion from a centred oblate 56Ni source in oblate spheroidal ejecta.

In focal-length units a=1 the ejecta semi-axes are
    A=cosh(xi_s), C=sinh(xi_s),
and the source semi-axes are f*cosh(xi1), f*sinh(xi1).

For even axisymmetric modes, angular functions solve
    ((1-eta^2) S')' + (lambda+c^2 eta^2) S = 0
in a normalized Legendre basis. Regular Helmholtz functions are expanded as
    psi(r,theta) = sum_k i^k d_k j_k(c*r) P_k(cos(theta)).
Their radial boundary zeros define c. The Legendre basis grows until its
coefficient tail is resolved. Mode norms use the Rellich surface identity;
uniform source projections use the divergence theorem with Laplacian psi=-c^2 psi.
The characteristic diffusion scale is t_unit=a^2/D in s^2.
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal

from _diffusion_common import (DAY, MSUN, gauss_legendre, legendre_all,
                                observer_cosines, phi_integral, solve, sph_jn_all)

DEFAULT_EDGES = np.linspace(-1.0, 1.0, 11)


def source_geometry_is_valid(xi_s, xi1, f=1.0):
    """True when the source spheroid fits inside the ejecta spheroid."""
    return (np.all(np.isfinite([xi_s, xi1, f])) and
            xi_s > 0 and xi1 > 0 and f > 0 and
            f * np.cosh(xi1) <= np.cosh(xi_s) + 1e-14 and
            f * np.sinh(xi1) <= np.sinh(xi_s) + 1e-14)


def _eta2_tridiagonal(nleg):
    """eta^2 in the normalised Legendre basis of even degrees 0, 2, ..., 2(nleg-1):
    diagonal and off-diagonal (degree k couples to k and k +- 2)."""
    k = 2.0 * np.arange(nleg)

    def a(q):
        q = np.asarray(q, dtype=float)
        return np.where(q > 0, q / np.sqrt(np.abs((2 * q - 1) * (2 * q + 1))), 0.0)

    return k, a(k + 1)**2 + a(k)**2, a(k[:-1] + 1) * a(k[:-1] + 2)


class _Angular:
    """Even-parity, m = 0 oblate angular functions in the Legendre basis."""

    def __init__(self, nleg):
        self.nleg = nleg
        self.k, self.e2d, self.e2o = _eta2_tridiagonal(nleg)
        self.kk1 = self.k * (self.k + 1)
        self.w = np.sqrt((2 * self.k + 1) / 2.0)          # dbar -> d (P_k coeffs)
        self.sgn = (-1.0)**(np.arange(nleg))              # i^k for even k
        self.kmax = int(self.k[-1])

    def lowest(self, c, J):
        """Eigenvectors of the J lowest eigenvalues at one c: (nleg, J)."""
        c2 = c * c
        return eigh_tridiagonal(self.kk1 - c2 * self.e2d, -c2 * self.e2o,
                                select="i", select_range=(0, J - 1))[1]

    def vectors(self, c, j):
        """The j-th eigenvector at each c[i] (index j[i]): (n, nleg)."""
        c = np.atleast_1d(c)
        j = np.broadcast_to(np.atleast_1d(j), c.shape)
        out = np.empty((c.size, self.nleg))
        for i in range(c.size):
            c2 = c[i] * c[i]
            out[i] = eigh_tridiagonal(self.kk1 - c2 * self.e2d, -c2 * self.e2o,
                                      select="i",
                                      select_range=(int(j[i]), int(j[i])))[1][:, 0]
        return out


def _R_from_vectors(ang, v, c, x):
    """R(c, x) = sum_k i^k d_k j_k(c x) / S(1), v: (n, nleg) normalised eigenvectors."""
    d = v * ang.w
    Jb = sph_jn_all(ang.kmax, np.asarray(c) * x)[::2].T          # (n, nleg)
    return np.sum(ang.sgn * d * Jb, axis=1) / d.sum(1)


def oblate_eigenvalues(xi_s, nmax, pmax, nleg=None, scan_per_spacing=8,
                       block=256, maxiter=100):
    """
    c[j, p]: the (p+1)-th zero of R_{2j}(c, sinh xi_s), j = 0..nmax/2.

    The zeros are bracketed on a grid with `scan_per_spacing` points per
    pi/(cosh xi_s) (< the zero spacing pi/sinh xi_s), and each bracket is
    refined by the Anderson-Bjorck variant of regula falsi (superlinear, never
    leaves the bracket), vectorised over all brackets.
    """
    if not np.isfinite(xi_s) or xi_s <= 0:
        raise ValueError("xi_s must be finite and positive")
    for name, value, minimum in [("nmax", nmax, 0), ("pmax", pmax, 1),
                                  ("block", block, 1), ("maxiter", maxiter, 1)]:
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(scan_per_spacing) or scan_per_spacing < 4:
        raise ValueError("scan_per_spacing must be finite and >= 4")
    x = np.sinh(xi_s)
    J = nmax // 2 + 1
    c_est = (pmax + J + 5) * np.pi / x
    automatic_basis = nleg is None
    if automatic_basis:
        nleg = J + 30 + int(3.0 * np.sqrt(c_est))
    if not isinstance(nleg, (int, np.integer)) or nleg < J + 2:
        raise ValueError("nleg must be an integer >= nmax//2 + 3")
    ang = _Angular(nleg)
    dc = np.pi / (scan_per_spacing * np.cosh(xi_s))
    # A heuristic is only a starting point. At large angular/radial mode
    # counts it can truncate substantial Legendre coefficients, even though
    # zeros of that truncated problem look perfectly converged. Require the
    # last eight normalized coefficients to have negligible norm. Explicit
    # nleg is a fixed resolution: reject insufficient values rather than
    # silently changing the requested resolution.
    tail_tolerance = 1e-13

    def tail_size(vectors):
        width = min(8, max(1, ang.nleg-J))
        return float(np.max(np.linalg.norm(vectors[..., -width:, :], axis=-2)))

    while tail_size(ang.lowest(c_est, J)) > tail_tolerance:
        if not automatic_basis:
            raise ValueError(f"nleg={ang.nleg} leaves an unresolved angular tail; increase nleg")
        ang = _Angular(max(ang.nleg+20, int(np.ceil(1.2*ang.nleg))))

    # --- scan ---------------------------------------------------------------
    roots = [[] for _ in range(J)]
    c0 = dc
    scan_limit = c_est
    last = [None] * J
    while min(len(r) for r in roots) < pmax:
        if c0 > scan_limit:
            scan_limit *= 1.5
        count = min(block, max(1, int(np.floor((scan_limit-c0)/dc))+1))
        cg = c0 + dc * np.arange(count)
        vectors = np.stack([ang.lowest(cc, J) for cc in cg])
        if tail_size(vectors) > tail_tolerance:
            if not automatic_basis:
                raise ValueError(f"nleg={ang.nleg} leaves an unresolved angular tail during the root scan; increase nleg")
            ang = _Angular(max(ang.nleg+20, int(np.ceil(1.2*ang.nleg))))
            roots = [[] for _ in range(J)]
            last = [None]*J
            c0 = dc
            continue
        d = vectors * ang.w[None, :, None]
        Jb = sph_jn_all(ang.kmax, cg * x)[::2]                    # (nleg, nc)
        g = np.einsum("kc,ckj->jc", ang.sgn[:, None] * Jb, d)      # (J, nc)
        Rg = g / d.sum(1).T
        # Conditioning: at small c, R_n ~ (c x)^n is far below the rounding
        # error of the eigenvector components (~eps each, absolute, for a unit
        # vector), so the computed sum is noise of size ~eps * ||j_k(c x)||_2
        # and its sign changes are not roots.  Trust only |sum| well above it.
        ok = np.abs(g) > 1e-12 * np.sqrt(np.sum((Jb * ang.w[:, None])**2, 0))
        for j in range(J):
            if len(roots[j]) == pmax:
                continue
            # A genuine zero at a scan node is also ill-conditioned. Retain
            # the last reliable sign instead of discarding BOTH adjoining
            # intervals. Bridge at most one uncertain node; this span is less
            # than the minimum root spacing because scan_per_spacing >= 4.
            for i in np.flatnonzero(ok[j]):
                point = (cg[i], Rg[j, i])
                if last[j] is not None:
                    a0, fa0 = last[j]
                    gap = point[0] - a0
                    if gap > 2.01 * dc:
                        # Resolve longer gaps on a finer grid before trusting
                        # a sign change (or assuming there was no root).
                        trial = np.linspace(a0, point[0],
                                            int(np.ceil(gap / (dc / 4))) + 1)
                        vv = ang.vectors(trial, j)
                        dd = vv * ang.w
                        jb = sph_jn_all(ang.kmax, trial * x)[::2].T
                        gg = np.sum(ang.sgn * dd * jb, axis=1)
                        good = np.abs(gg) > 1e-12 * np.linalg.norm(jb * ang.w, axis=1)
                        ii = np.flatnonzero(good)
                        if (ii.size < 2 or
                                np.max(np.diff(trial[ii])) > 2.01 * dc):
                            raise RuntimeError(
                                f"Unresolved radial scan for n={2*j} near c={a0:g}; "
                                "increase nleg or scan_per_spacing")
                        rr = gg / dd.sum(1)
                        for ia, ib in zip(ii[:-1], ii[1:]):
                            if np.signbit(rr[ia]) != np.signbit(rr[ib]):
                                roots[j].append((trial[ia], trial[ib], rr[ia], rr[ib]))
                                if len(roots[j]) == pmax:
                                    break
                    elif np.signbit(fa0) != np.signbit(point[1]):
                        roots[j].append((a0, point[0], fa0, point[1]))
                last[j] = point
                if len(roots[j]) == pmax:
                    break
        c0 = cg[-1] + dc
        if c0 > 50 * c_est:
            raise RuntimeError("oblate_eigenvalues: zero scan ran away")

    # --- Anderson-Bjorck refinement -----------------------------------------
    jj = np.repeat(np.arange(J), pmax)
    a, b, fa, fb = np.array([r[:pmax] for r in roots]).reshape(-1, 4).T
    c = b.copy()
    active = np.ones(c.size, bool)
    for _ in range(maxiter):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        ai, bi, fai, fbi = a[idx], b[idx], fa[idx], fb[idx]
        ci = bi - fbi * (bi - ai) / (fbi - fai)
        lo, hi = np.minimum(ai, bi), np.maximum(ai, bi)
        bad = ~np.isfinite(ci) | (ci <= lo) | (ci >= hi)
        # Adjacent floating-point endpoints already locate the root.
        narrow = np.abs(bi - ai) <= 4e-16 * np.maximum(np.abs(bi), np.abs(ai))
        ci = np.where(bad & ~narrow, 0.5 * (ai + bi), ci)
        fc = _R_from_vectors(ang, ang.vectors(ci, jj[idx]), ci, x)
        if not np.all(np.isfinite(fc)):
            raise RuntimeError("Non-finite radial function during root refinement")
        same = np.signbit(fc) == np.signbit(fbi)
        m = np.where(same, 1.0 - fc / np.where(fbi != 0, fbi, 1.0), 1.0)
        m = np.where(m > 0, m, 0.5)
        # sign change between b and c: (a, b) <- (b, c); else shrink fa
        a[idx] = np.where(same, ai, bi)
        fa[idx] = np.where(same, fai * m, fbi)
        b[idx], fb[idx] = ci, fc
        conv = (np.abs(ci - c[idx]) <= 1e-14 * ci) | (fc == 0.0) | \
               (np.abs(bi - ai) <= 4e-16 * ci)
        c[idx] = ci
        active[idx[conv]] = False
    if active.any():
        raise RuntimeError(f"{active.sum()} oblate roots did not converge in {maxiter} iterations")
    table = c.reshape(J, pmax)
    final_vectors = ang.vectors(c, jj)
    width = min(8, max(1, ang.nleg-J))
    angular_tail = float(np.max(np.linalg.norm(final_vectors[:, -width:], axis=1)))
    if angular_tail > tail_tolerance:
        raise RuntimeError("Unresolved angular tail at an oblate eigenvalue")
    ang.max_angular_tail = angular_tail
    if np.any(~np.isfinite(table)) or np.any(table <= 0) or np.any(np.diff(table, axis=1) <= 0):
        raise RuntimeError("Invalid or duplicated oblate roots")
    # A tiny interpolation step alone can also indicate stagnation. Verify
    # every returned root against a locally measured radial slope, including
    # the c-dependence of the angular eigenvector. The inferred correction
    # must be small relative to c; this is a residual check, not a new root.
    delta = np.minimum(.1*c, np.maximum(dc*1e-5, np.sqrt(np.finfo(float).eps)*c))
    rm = _R_from_vectors(ang, ang.vectors(c-delta, jj), c-delta, x)
    rp = _R_from_vectors(ang, ang.vectors(c+delta, jj), c+delta, x)
    r0 = _R_from_vectors(ang, ang.vectors(c, jj), c, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = np.abs(2*delta*r0/(rp-rm))
    if np.any(~np.isfinite(correction)) or np.any(correction > 2e-12*c):
        raise RuntimeError("Oblate root refinement stagnated with a nonzero radial residual")
    ang.max_relative_root_residual = float(np.max(correction/c))
    # The fundamental Dirichlet root is bounded by the enclosing and
    # inscribed spheres. This also detects skipped roots near the sphere limit.
    if not (np.pi / np.cosh(xi_s) * (1-1e-12) <= table[0, 0] <= np.pi / x * (1+1e-12)):
        raise RuntimeError("Fundamental oblate root violates the sphere bounds")
    return table, ang


class OblateSpheroids:
    """Reusable spatial modes for oblate ejecta and a centred oblate source.

    xi_s>0 specifies the ejecta axis ratio tanh(xi_s); xi1>0 specifies
    the source axis ratio tanh(xi1); f>0 is the focal-length ratio.
    Both source semi-axes must fit within the ejecta. nmax is the largest
    even angular index and pmax is the radial root count per index.
    nleg=None grows the Legendre basis automatically; a fixed undersized
    basis raises an error. n_src/n_norm set source-surface/norm quadrature.
    """

    def __init__(self, xi_s, xi1, f=1.0, nmax=100, pmax=200, nleg=None,
                 n_src=400, n_norm=800, verbose=False, scan_per_spacing=8):
        if not source_geometry_is_valid(xi_s, xi1, f):
            raise ValueError("source spheroid sticks out of the ejecta")
        import time
        t0 = time.time()
        self.xi_s, self.xi1, self.f = xi_s, xi1, f
        for name, value in [("n_src", n_src), ("n_norm", n_norm)]:
            if not isinstance(value, (int, np.integer)) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        c, ang = oblate_eigenvalues(xi_s, nmax, pmax, nleg=nleg,
                                   scan_per_spacing=scan_per_spacing)
        self.ang = ang
        self.nmax, self.pmax = nmax, pmax
        self.numerical_settings = dict(nmax=nmax, pmax=pmax, nleg=ang.nleg,
                                       n_src=n_src, n_norm=n_norm,
                                       scan_per_spacing=scan_per_spacing)
        self.mode_diagnostics = dict(max_relative_root_residual=ang.max_relative_root_residual,
                                     max_angular_tail=ang.max_angular_tail)
        J = c.shape[0]
        self.n_ang = np.repeat(2 * np.arange(J), pmax)
        self.p = np.tile(np.arange(pmax), J)
        cf = c.ravel()
        self.c = cf
        self.beta = cf**2
        if verbose:
            print(f"  eigenvalues: {cf.size} modes, nleg = {ang.nleg}, "
                  f"c <= {cf.max():.1f}  ({time.time() - t0:.1f} s)")

        # angular functions at the roots
        v = ang.vectors(cf, self.n_ang // 2)
        d = v * ang.w                                    # P_k coefficients, even k
        S1 = d.sum(1)
        d = d * np.sign(S1)[:, None]
        S1 = np.abs(S1)
        b = ang.sgn * d                                  # spherical-wave coeffs
        self.d, self.b = d, b
        kmax = ang.kmax

        sh, ch = np.sinh(xi_s), np.cosh(xi_s)
        # R'(xi_s) at the boundary: d/dxi = cosh(xi) d/d(sinh(xi)).
        _, dJs = sph_jn_all(kmax, cf * sh, derivative=True)
        Rp = ch * cf * np.sum(b * dJs[::2].T, axis=1) / S1
        self.Rp = Rp

        # L_total weight: oint -d_n psi dA = -2 pi cosh(xi_s) R' int S sin(eta) deta
        self.B = -2.0 * np.pi * ch * Rp * 2.0 * d[:, 0]

        # norm (Rellich)
        u, wu = gauss_legendre(-1.0, 1.0, n_norm)
        Pu = legendre_all(kmax, u)[::2]                  # (nleg, n_norm)
        S_u = d @ Pu
        I = (S_u**2 / (sh**2 + u**2)) @ wu
        norm = np.pi * Rp**2 * sh * ch**2 / cf**2 * I

        # source projection (divergence theorem on the source spheroid)
        A1, C1 = f * np.cosh(xi1), f * np.sinh(xi1)
        V_src = 4.0 * np.pi / 3.0 * A1**2 * C1
        flux_src = self._source_flux(cf, b, kmax, A1, C1, n_src)
        num = -flux_src / cf**2
        self.C = num / V_src / norm
        if np.any(~np.isfinite(norm)) or np.any(norm <= 0) or np.any(~np.isfinite(self.C)):
            raise RuntimeError("Invalid mode norm/source projection; check geometry and resolution")
        self.norm = norm
        self.n_src = n_src
        if verbose:
            print(f"  mode table done ({time.time() - t0:.1f} s), "
                  f"sum rule = {self.energy_sum_rule:.6f}")

    @staticmethod
    def _source_flux(c, b, kmax, A1, C1, n_src, chunk=48):
        """oint_src grad psi . n dA for every mode."""
        s, ws = gauss_legendre(0.0, np.pi, n_src)
        rho, z = A1 * np.sin(s), C1 * np.cos(s)
        r = np.hypot(rho, z)
        u = z / r
        nrho, nz = C1 * np.sin(s), A1 * np.cos(s)       # n dA / (2 pi rho ds)
        P, dP = legendre_all(kmax, u, derivative=True)
        P, dP = P[::2], dP[::2]                          # (nleg, n_src)
        out = np.empty(c.size)
        wgt = 2.0 * np.pi * rho * ws
        for i0 in range(0, c.size, chunk):
            cc = c[i0:i0 + chunk]
            Jm, dJm = sph_jn_all(kmax, np.outer(cc, r), derivative=True)
            Jm, dJm = Jm[::2], dJm[::2]                  # (nleg, nc, n_src)
            bb = b[i0:i0 + chunk].T                      # (nleg, nc)
            psi_r = cc[:, None] * np.einsum("kc,kcq,kq->cq", bb, dJm, P)
            psi_u = np.einsum("kc,kcq,kq->cq", bb, Jm, dP)
            # grad psi in (rho, z): u = z/r
            d_rho = psi_r * rho / r - psi_u * z * rho / r**3
            d_z = psi_r * z / r + psi_u * rho**2 / r**3
            out[i0:i0 + chunk] = (d_rho * nrho + d_z * nz) @ wgt
        return out

    @property
    def energy_sum_rule(self):
        """sum_m B_m C_m / beta_m: the late-time L/Q; 1 for a complete mode set."""
        return float(np.sum(self.B * self.C / self.beta))

    def observer_matrix(self, obs_cos_edges=DEFAULT_EDGES, observer="centre",
                        n_split=160, n_bin=8):
        """Surface flux per mode, splitting the visibility-transition latitudes."""
        if not isinstance(n_split, (int, np.integer)) or n_split < 2:
            raise ValueError("n_split must be an integer >= 2")
        sh, ch = np.sinh(self.xi_s), np.cosh(self.xi_s)
        bins, centres = observer_cosines(obs_cos_edges, observer, n_bin=n_bin)
        Om = np.zeros((len(bins), self.ang.nleg))
        for j, (cx, cw) in enumerate(bins):
            for co, weight in zip(cx, cw):
                so = np.sqrt(max(1.0-co*co, 0.0))
                es = np.arctan2(abs(co)*ch, so*sh)
                pieces = [gauss_legendre(lo, hi, n_split)
                          for lo, hi in [(0.0, es), (es, np.pi-es), (np.pi-es, np.pi)]
                          if hi > lo]
                eta = np.concatenate([p[0] for p in pieces])
                we = np.concatenate([p[1] for p in pieces])
                N = np.sqrt(sh*sh+np.cos(eta)**2)
                W = phi_integral(so*sh*np.sin(eta)/N, co*ch*np.cos(eta)/N)
                P = legendre_all(self.ang.kmax, np.cos(eta))[::2]
                Om[j] += weight*(P @ (we*np.sin(eta)*W))
        return 3.0*ch*(-self.Rp)[None, :]*(Om @ self.d.T), centres

    def light_curve(self, t_unit, Mni56, tdep=None, fdep_file=None,
                    tmax=80*DAY, Nt=8000, obs_cos_edges=DEFAULT_EDGES,
                    observer="centre", oversample=32, n_split=160, n_bin=8):
        """Calculate (t, L_total, L_view, cos_view) with this spatial mode table."""
        return solve(self, t_unit, Mni56, tdep=tdep, fdep_file=fdep_file,
                     tmax=tmax, Nt=Nt, oversample=oversample,
                     obs_cos_edges=obs_cos_edges, observer=observer,
                     observer_options={"n_split": n_split, "n_bin": n_bin})


def light_curve_oblate_viewing_angle_rm_degeneracy_1a(
        t_unit, xi_s, xi1, Mni56, tdep=None, tmax=80*DAY, Nt=8000,
        nmax=100, pmax=200, obs_cos_edges=DEFAULT_EDGES, f=1.0,
        fdep_file=None, observer="centre", oversample=32, nleg=None,
        n_src=400, n_norm=800, n_split=160, n_bin=8,
        scan_per_spacing=8, verbose=False):
    """Bolometric 56Ni/56Co light curves using a supplied diffusion scale.

    Parameters
    ----------
    t_unit : float
        Characteristic diffusion scale a^2/D [s^2], with ejecta focal length a.
    xi_s, xi1 : float
        Positive oblate coordinates; ejecta/source axis ratios are tanh(xi_s)/tanh(xi1).
    f : float
        Source/ejecta focal-length ratio. Source semi-axes are
        f*a*cosh(xi1), f*a*sinh(xi1), and must fit within the ejecta.
        f=1 is confocal; f>1 is allowed when the source fits.
    Mni56 : float
        Initial 56Ni mass [g].
    tdep : float or None
        Gamma-ray escape time [s]; None means full trapping unless a table is supplied.
    fdep_file : path or None
        Whitespace-separated table (time [s], gamma deposition fraction).
        Interpolation is linear in t^2 and held constant outside the table.
        Supply either tdep or fdep_file.
    tmax, Nt : float, int
        Maximum time [s] and number of nodes uniform in t^2; omit the last node.
    nmax, pmax : int
        Largest even angular index and radial roots per index.
    obs_cos_edges : array-like or None
        Increasing edges in cos(theta). None requests the total curve only.
    observer : {'centre', 'bin'}
        Evaluate bin centres or average over each bin using n_bin Gauss nodes.
    oversample : int
        Internal time intervals per output interval; returned samples are exact grid nodes.
    nleg : int or None
        Even Legendre terms; None selects the basis adaptively.
    n_src, n_norm, n_split, n_bin : int
        Quadrature orders for source surface, mode norm, each visible-surface
        interval, and observer-bin average.
    scan_per_spacing : float
        Radial scan points per pi/cosh(xi_s); must be at least four.
    verbose : bool
        Print mode-table progress and elapsed time.

    Returns
    -------
    t : ndarray, shape (Nt-1,)
        Time [s], starting at zero.
    L_total : ndarray, shape (Nt-1,)
        Angle-integrated luminosity [erg/s].
    L_view : ndarray, shape (number of bins, Nt-1), or None
        Isotropic-equivalent luminosities [erg/s].
    cos_view : ndarray or None
        Centres of the observer bins.
    """
    geom = OblateSpheroids(xi_s, xi1, f=f, nmax=nmax, pmax=pmax,
                           nleg=nleg, n_src=n_src, n_norm=n_norm,
                           scan_per_spacing=scan_per_spacing, verbose=verbose)
    return geom.light_curve(t_unit, Mni56, tdep=tdep, fdep_file=fdep_file,
                            tmax=tmax, Nt=Nt, obs_cos_edges=obs_cos_edges,
                            observer=observer, oversample=oversample,
                            n_split=n_split, n_bin=n_bin)
