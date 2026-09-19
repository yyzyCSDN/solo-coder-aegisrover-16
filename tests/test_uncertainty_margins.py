"""Tests for uncertainty-aware obstacle safety margins."""
import math

import pytest

from aegisrover.planning.predict import (
    CircleObstacle, minkowski_clearance, swept_path_conflict,
)
from aegisrover.planning.uncertainty import (
    UncertaintyMargin, bearing_sigma, chi2_quantile_2dof, inflate_uncertain,
    inverse_normal_cdf, isotropic_sigma, relative_covariance,
    uncertain_clearance, uncertainty_margin,
)

Z95 = 1.6448536269514722


# -------------------------------------------------------------------------- quantiles
def test_normal_quantiles_match_known_values():
    assert inverse_normal_cdf(0.95) == pytest.approx(1.64485363, abs=1e-7)
    assert inverse_normal_cdf(0.5) == pytest.approx(0.0, abs=1e-12)
    assert inverse_normal_cdf(0.975) == pytest.approx(1.95996398, abs=1e-7)
    assert inverse_normal_cdf(0.999) == pytest.approx(3.09023231, abs=1e-7)
    # symmetry of the tails
    assert inverse_normal_cdf(0.1) == pytest.approx(-inverse_normal_cdf(0.9), abs=1e-9)
    with pytest.raises(ValueError):
        inverse_normal_cdf(0.0)
    with pytest.raises(ValueError):
        inverse_normal_cdf(1.0)


def test_chi2_two_dof_quantile():
    assert chi2_quantile_2dof(0.95) == pytest.approx(-2.0 * math.log(0.05))


# --------------------------------------------------------------------- covariance math
def test_relative_covariance_adds_independent_errors():
    cov = relative_covariance(((1.0, 0.2), (0.2, 1.0)), ((3.0, 0.0), (0.0, 4.0)))
    assert cov == pytest.approx(((4.0, 0.2), (0.2, 5.0)))
    assert relative_covariance(None, None) == ((0.0, 0.0), (0.0, 0.0))


def test_bearing_sigma_projects_anisotropic_covariance():
    cov = ((0.25, 0.0), (0.0, 1.0))  # sigma 0.5 along x, 1.0 along y
    assert bearing_sigma(cov, (1.0, 0.0)) == pytest.approx(0.5)
    assert bearing_sigma(cov, (0.0, 1.0)) == pytest.approx(1.0)
    assert bearing_sigma(cov, (1.0, 1.0)) == pytest.approx(math.sqrt(0.625))
    assert bearing_sigma(None, (1.0, 0.0)) == 0.0


def test_isotropic_sigma_is_largest_eigenvalue_root():
    cov = ((1.0, 0.0), (0.0, 0.25))
    assert isotropic_sigma(cov) == pytest.approx(1.0)
    rotated = ((0.625, 0.375), (0.375, 0.625))  # eigenvalues 1.0 and 0.25
    assert isotropic_sigma(rotated) == pytest.approx(1.0)
    # coincident LOS degenerates to the conservative worst-case bearing
    assert bearing_sigma(rotated, (0.0, 0.0)) == pytest.approx(1.0)


# ----------------------------------------------------------------------------- margins
def test_zero_uncertainty_adds_no_margin():
    obstacle = CircleObstacle(3.0, 0.0, 0.5)
    result = uncertain_clearance(obstacle, (0.0, 0.0), robot_radius=0.4)
    assert result.margin == pytest.approx(0.0)
    assert result.safe_distance == pytest.approx(0.9)
    assert result.total_sigma == 0.0 and result.alarm is False


def test_larger_error_gives_larger_margin_and_alarms_earlier():
    # Object straight ahead along +x; the range error is exactly sigma_x.
    tight = CircleObstacle(3.0, 0.0, 0.5, covariance=((0.01, 0.0), (0.0, 4.0)))
    loose = CircleObstacle(3.0, 0.0, 0.5, covariance=((0.16, 0.0), (0.0, 4.0)))
    r_tight = uncertain_clearance(tight, (0.0, 0.0))
    r_loose = uncertain_clearance(loose, (0.0, 0.0))
    assert r_tight.position_sigma == pytest.approx(0.1)
    assert r_loose.position_sigma == pytest.approx(0.4)
    assert r_loose.margin > r_tight.margin
    assert r_loose.margin == pytest.approx(Z95 * 0.4)
    assert r_tight.safe_distance == pytest.approx(0.5 + Z95 * 0.1)

    # The same measured range can be "clear" with a good sensor but already alarm
    # with a noisy one.
    near = CircleObstacle(0.72, 0.0, 0.5)
    assert uncertain_clearance(near, (0.0, 0.0)).alarm is False
    near_noisy = CircleObstacle(0.72, 0.0, 0.5,
                                covariance=((0.04, 0.0), (0.0, 0.04)))
    assert uncertain_clearance(near_noisy, (0.0, 0.0)).alarm is True


def test_cross_axis_uncertainty_is_ignored_for_a_head_on_obstacle():
    # Huge bearing uncertainty perpendicular to the LOS must not inflate the
    # longitudinal range bound.
    cov = ((0.09, 0.0), (0.0, 100.0))
    result = uncertain_clearance(CircleObstacle(5.0, 0.0, 0.5, covariance=cov),
                                 (0.0, 0.0))
    assert result.position_sigma == pytest.approx(0.3)
    assert result.margin == pytest.approx(Z95 * 0.3)


def test_robot_localisation_covariance_is_included():
    obstacle = CircleObstacle(3.0, 0.0, 0.5,
                              covariance=((0.09, 0.0), (0.0, 0.0)))
    alone = uncertain_clearance(obstacle, (0.0, 0.0))
    with_localisation = uncertain_clearance(
        obstacle, (0.0, 0.0), robot_covariance=((0.16, 0.0), (0.0, 0.0)))
    assert alone.position_sigma == pytest.approx(0.3)
    assert with_localisation.position_sigma == pytest.approx(0.5)


def test_radius_sigma_combines_in_quadrature():
    obstacle = CircleObstacle(3.0, 0.0, 0.5,
                              covariance=((0.09, 0.0), (0.0, 0.0)),
                              radius_sigma=0.4)
    result = uncertain_clearance(obstacle, (0.0, 0.0))
    assert result.total_sigma == pytest.approx(0.5)
    assert result.margin == pytest.approx(Z95 * 0.5)


def test_hard_buffer_is_additive_and_zero_error_still_reserves_it():
    obstacle = CircleObstacle(3.0, 0.0, 0.5)
    result = uncertain_clearance(obstacle, (0.0, 0.0), hard_buffer=0.2)
    assert result.margin == pytest.approx(0.2)
    assert result.safe_distance == pytest.approx(0.7)


def test_confidence_controls_the_quantile():
    obstacle = CircleObstacle(3.0, 0.0, 0.5,
                              covariance=((0.04, 0.0), (0.0, 0.0)))
    p90 = uncertain_clearance(obstacle, (0.0, 0.0), confidence=0.90)
    p99 = uncertain_clearance(obstacle, (0.0, 0.0), confidence=0.99)
    assert p90.margin == pytest.approx(1.28155157 * 0.2)
    assert p99.margin > p90.margin


def test_explain_and_dict_report_every_term():
    obstacle = CircleObstacle(3.0, 0.0, 0.5,
                              covariance=((0.04, 0.0), (0.0, 9.0)),
                              radius_sigma=0.0)
    result = uncertain_clearance(obstacle, (0.0, 0.0))
    text = result.explain()
    assert '0.2000 m' in text and 'z=1.645' in text and 'clear' in text
    data = result.to_dict()
    assert data['position_sigma'] == pytest.approx(0.2)
    assert data['safe_distance'] == pytest.approx(0.5 + Z95 * 0.2)
    assert data['alarm'] is False and data['isotropic'] is False


# ---------------------------------------------------------------------- map inflation
def test_inflate_uncertain_uses_worst_case_bearing():
    obstacle = CircleObstacle(2.0, 0.0, 0.5,
                              covariance=((0.25, 0.0), (0.0, 1.0)))
    inflated = inflate_uncertain([obstacle])
    # worst-case LOS sigma = 1.0 (largest eigenvalue)
    assert inflated[0].radius == pytest.approx(0.5 + Z95 * 1.0)
    assert inflated[0].covariance is None and inflated[0].radius_sigma == 0.0


def test_chi2_inflation_is_more_conservative_than_normal():
    obstacle = CircleObstacle(2.0, 0.0, 0.5,
                              covariance=((1.0, 0.0), (0.0, 1.0)))
    normal = inflate_uncertain([obstacle], bound='normal')[0].radius
    chi2 = inflate_uncertain([obstacle], bound='chi2')[0].radius
    assert chi2 > normal
    assert chi2 == pytest.approx(0.5 + math.sqrt(-2.0 * math.log(0.05)))


def test_inflated_obstacles_work_with_existing_collision_checks():
    path = [(0.0, 1.2), (4.0, 1.2)]
    obstacle = CircleObstacle(2.0, 0.0, 0.5,
                              covariance=((0.25, 0.0), (0.0, 1.0)))
    # Nominal gap is 1.2 - (0.5 + 0.4) = 0.3 m: raw obstacle lets the path
    # through with a 0.4 robot...
    assert swept_path_conflict(path, [obstacle], robot_radius=0.4) is None
    # ...but the uncertainty-inflated circle rejects it, and the reported
    # clearance becomes negative.
    inflated = inflate_uncertain([obstacle])
    conflict = swept_path_conflict(path, inflated, robot_radius=0.4)
    assert conflict is not None
    assert minkowski_clearance(path, inflated) < 0.0


# --------------------------------------------------------------------------- validation
def test_invalid_arguments_are_rejected():
    obstacle = CircleObstacle(3.0, 0.0, 0.5)
    with pytest.raises(ValueError):
        uncertain_clearance(obstacle, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        uncertainty_margin(obstacle, (0.0, 0.0), confidence=1.0)
    with pytest.raises(ValueError):
        uncertainty_margin(obstacle, (0.0, 0.0), isotropic=False, bound='chi2')
    with pytest.raises(ValueError):
        uncertainty_margin(obstacle, (0.0, 0.0), bound='banana')
    with pytest.raises(ValueError):
        uncertainty_margin(obstacle, (0.0, 0.0), hard_buffer=-0.1)
    assert uncertainty_margin(obstacle, None, isotropic=True).alarm is False
