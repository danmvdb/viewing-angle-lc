"""Full and first-order oblate diffusion with a centred 56Ni source."""

from .analytic_oblate_viewing_angle import (
    DAY, MSUN, OblateSpheroids, source_geometry_is_valid,
    light_curve_oblate_viewing_angle_rm_degeneracy_1a,
)
from .analytic_slightly_oblate_viewing_angle import (
    SlightlyOblate, eps_from_delta, eps_s_from_f_q, quasi_static_multipole,
    light_curve_slightly_oblate_viewing_angle_rm_degeneracy_1a,
)

__all__ = ["DAY", "MSUN", "OblateSpheroids", "source_geometry_is_valid",
           "light_curve_oblate_viewing_angle_rm_degeneracy_1a",
           "SlightlyOblate", "eps_from_delta", "eps_s_from_f_q",
           "quasi_static_multipole",
           "light_curve_slightly_oblate_viewing_angle_rm_degeneracy_1a"]
