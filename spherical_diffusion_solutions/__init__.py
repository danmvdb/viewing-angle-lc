"""Spherical ejecta heated by an off-centre spherical 56Ni source."""

from .analytic_spherical_off_center_viewing_angle import (
    DAY, MSUN, OffCenterSphere,
    light_curve_offcenter_viewing_angle_rm_degeneracy_1a,
)

__all__ = ["DAY", "MSUN", "OffCenterSphere",
           "light_curve_offcenter_viewing_angle_rm_degeneracy_1a"]
