"""Diffusion from an off-centre spherical 56Ni source in spherical ejecta.

The ejecta radius is the length unit. Modes are
    psi_ln = j_l(alpha_ln*r) P_l(cos(theta)),  j_l(alpha_ln)=0.
Uniform-source projections follow the spherical-mean identity; observer
weights follow the Funk-Hecke theorem for the Eddington intensity kernel.
The characteristic diffusion scale is t_unit=v_o^2/D in s^2, where v_o is
the outer ejecta velocity and D is the diffusion coefficient in velocity space.
"""

import warnings

import numpy as np

from _diffusion_common import (DAY, MSUN, gauss_legendre, legendre_all,
                                observer_cosines, solve, sph_jn_all)

DEFAULT_EDGES = np.linspace(-1.0, 1.0, 11)


def bessel_zeros(lmax, nmax):
    """alpha[l, n] = (n+1)-th positive zero of j_l, for l <= lmax, n < nmax."""
    L = np.arange(lmax + 1)
    xmax = (nmax + 0.5 * lmax + 3.0) * np.pi
    while True:
        xg = np.arange(0.5, xmax, 0.25)      # zero spacing of j_l is > pi
        Jg = sph_jn_all(lmax, xg)
        s = np.signbit(Jg)
        lo_idx = [np.nonzero(s[l, 1:] != s[l, :-1])[0][:nmax] for l in L]
        if all(len(ix) == nmax for ix in lo_idx):
            break
        xmax *= 1.3
    ll = np.repeat(L, nmax)
    ix = np.concatenate(lo_idx)
    a, b = xg[ix], xg[ix + 1]
    fa = Jg[ll, ix]
    fb = Jg[ll, ix + 1]
    x = a - fa * (b - a) / (fb - fa)         # regula falsi start
    cols = np.arange(ll.size)
    for _ in range(60):
        J, dJ = sph_jn_all(lmax, x, derivative=True)
        f, df = J[ll, cols], dJ[ll, cols]
        # keep the bracket
        left = np.signbit(f) == np.signbit(fa)
        a = np.where(left, x, a)
        fa = np.where(left, f, fa)
        b = np.where(left, b, x)
        xn = x - f / df
        bad = ~((xn > a) & (xn < b))
        xn = np.where(bad, 0.5 * (a + b), xn)
        done = np.abs(xn - x) < 1e-14 * x
        x = xn
        if done.all():
            break
    return x.reshape(lmax + 1, nmax)


class OffCenterSphere:
    """Reusable spatial modes for a uniform spherical source.

    xs is its radius and z_off its axial displacement, in ejecta-radius units.
    Require xs>0 and xs+abs(z_off)<=1. lmax is the largest angular degree;
    nmax is the number of positive radial roots per degree.

    allow_outside=True accepts xs+abs(z_off)>1 with a warning instead of an
    error: the closed-form projection is then
    evaluated for the full source ball and the heating is still normalised
    by its full volume, so the energy deposited in the part outside the
    ejecta is missing from L_total.
    """

    def __init__(self, xs, z_off, lmax=100, nmax=100, allow_outside=False):
        if not np.all(np.isfinite([xs, z_off])) or xs <= 0:
            raise ValueError("xs must be positive and geometry parameters finite")
        for name, value, minimum in [("lmax", lmax, 0), ("nmax", nmax, 1)]:
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if xs + abs(z_off) > 1.0 + 1e-12:
            if not allow_outside:
                raise ValueError(f"source sticks out of the ejecta: xs + |z_off| = "
                                 f"{xs + abs(z_off):.4f} > 1 "
                                 f"(pass allow_outside=True to accept it)")
            warnings.warn(
                f"source geometry sticks out of the ejecta: xs + |z_off| = "
                f"{xs + abs(z_off):.4f} > 1 (xs={xs}, z_off={z_off}).  The part "
                f"outside the ejecta is still counted in the source normalisation, "
                f"so a fraction of the deposited energy will go missing from L_total.",
                RuntimeWarning, stacklevel=2)
        self.xs, self.z_off, self.lmax, self.nmax = xs, z_off, lmax, nmax
        self.numerical_settings = dict(lmax=lmax, nmax=nmax)
        if allow_outside:
            self.numerical_settings["allow_outside"] = True
        alpha = bessel_zeros(lmax, nmax)                 # (L, N)
        l = np.arange(lmax + 1)[:, None] * np.ones((1, nmax), dtype=int)
        a = alpha.ravel()
        lf = l.ravel()
        cols = np.arange(a.size)

        J_a = sph_jn_all(lmax + 1, a)                    # j_k(alpha)
        jl1 = J_a[lf + 1, cols]                          # j_{l+1}(alpha)
        z = abs(z_off)
        jz = sph_jn_all(lmax, a * z)[lf, cols]           # j_l(alpha |z|)
        if z_off < 0:
            jz = jz * (-1.0)**lf                         # parity
        xa = a * xs
        with np.errstate(divide="ignore", invalid="ignore"):
            Phi = np.where(xa < 1e-8, 1.0 - xa**2 / 10.0,
                           3.0 * sph_jn_all(1, xa)[1] / np.where(xa < 1e-8, 1, xa))
        self.l = lf
        self.alpha = a
        self.beta = a**2
        self.C = (2 * lf + 1) * jz * Phi / (2.0 * np.pi * jl1**2)
        self.f = a * jl1
        self.B = np.where(lf == 0, 4.0 * np.pi * self.f, 0.0)

        # Funk-Hecke weights kappa_l = int_0^1 mu (mu + 2/3) P_l(mu) dmu (exact)
        mu, w = gauss_legendre(0.0, 1.0, lmax // 2 + 4)
        P = legendre_all(lmax, mu)
        self.kappa = P @ (w * mu * (mu + 2.0 / 3.0))

    @property
    def energy_sum_rule(self):
        """sum_m B_m C_m / beta_m: the late-time L/Q; 1 for a complete mode set."""
        return float(np.sum(self.B * self.C / self.beta))

    def observer_matrix(self, obs_cos_edges=DEFAULT_EDGES, observer="centre", n_bin=8):
        """Isotropic-equivalent surface flux per mode, using Funk-Hecke moments."""
        bins, centres = observer_cosines(obs_cos_edges, observer, n_bin=n_bin)
        G = np.zeros((len(bins), self.alpha.size))
        for j, (cx, cw) in enumerate(bins):
            Pl = legendre_all(self.lmax, cx) @ cw
            G[j] = 6.0*np.pi*self.kappa[self.l]*Pl[self.l]*self.f
        return G, centres

    def light_curve(self, t_unit, Mni56, tdep=None, fdep_file=None,
                    tmax=80*DAY, Nt=8000, obs_cos_edges=DEFAULT_EDGES,
                    observer="centre", oversample=32, n_bin=8):
        """Calculate (t, L_total, L_view, cos_view) with this spatial mode table."""
        return solve(self, t_unit, Mni56, tdep=tdep, fdep_file=fdep_file,
                     tmax=tmax, Nt=Nt, oversample=oversample,
                     obs_cos_edges=obs_cos_edges, observer=observer,
                     observer_options={"n_bin": n_bin})


def light_curve_offcenter_viewing_angle_rm_degeneracy_1a(
        t_unit, xs, z_off, Mni56, tdep=None, tmax=80*DAY, Nt=8000,
        lmax=100, nmax=100, obs_cos_edges=DEFAULT_EDGES, fdep_file=None,
        observer="centre", oversample=32, n_bin=8, verbose=False,
        allow_outside=False):
    """Bolometric 56Ni/56Co light curves using a supplied diffusion scale.

    Parameters
    ----------
    t_unit : float
        Characteristic diffusion scale v_o^2/D [s^2].
    xs, z_off : float
        Source radius and axial offset, divided by the ejecta radius.
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
    lmax, nmax : int
        Largest angular degree and radial roots per degree.
    obs_cos_edges : array-like or None
        Increasing edges in cos(theta). None requests the total curve only.
    observer : {'centre', 'bin'}
        Evaluate bin centres or average over each bin using n_bin Gauss nodes.
    oversample : int
        Internal time intervals per output interval; returned samples are exact grid nodes.
    n_bin : int
        Observer-bin integration order when observer='bin'.
    verbose : bool
        Print the spatial mode count.
    allow_outside : bool
        Accept xs + |z_off| > 1 with a warning (see OffCenterSphere).

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
    geom = OffCenterSphere(xs, z_off, lmax=lmax, nmax=nmax,
                           allow_outside=allow_outside)
    if verbose:
        print(f"Spherical geometry: {geom.alpha.size} spatial modes")
    return geom.light_curve(t_unit, Mni56, tdep=tdep, fdep_file=fdep_file,
                            tmax=tmax, Nt=Nt, obs_cos_edges=obs_cos_edges,
                            observer=observer, oversample=oversample, n_bin=n_bin)
