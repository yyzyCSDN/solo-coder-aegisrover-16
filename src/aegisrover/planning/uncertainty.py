"""Uncertainty-aware safety margins for observed circle obstacles.

A sensor does not return the true obstacle centre: it returns an estimate with a
2x2 covariance (the tracker/EKF posterior), and the physical radius itself may be
mis-estimated by the detector. Collision tests on the raw numbers only are then
over-optimistic — with large sensor error the robot can be commanded to drive to
within ``robot_radius + obstacle.radius`` of a centre that is in fact closer.

The safety margin below is derived as follows.  Let

    p = obstacle centre estimate - robot position estimate,
    e = error in the relative position, modelled e ~ N(0, C),
        C = C_obstacle + C_robot (+ C_predict for predicted contacts),
    r_nom = robot_radius + obstacle.radius,
    sigma_r = combined radius standard deviation.

The true range along the line of sight is ``u = u_hat - u^T e`` where
``u_hat = |p|``.  Gaussian propagation gives

    sigma_u = sqrt(u^T C u / u_hat^2)          (std-dev of the LOS range),
    sigma_tot = sqrt(sigma_u^2 + sigma_r^2)    (range and size combined).

The robot is safe when the true clearance is non-negative, i.e.

    u_hat >= r_nom + m,   m = z * sigma_tot,

where ``z`` is the one-sided standard-normal quantile ``z = Phi^{-1}(1 - alpha)``.
With this margin the probability of being inside the true disk at the alarm
boundary is at most ``alpha`` (95% confidence => ``z = 1.6449``).  The margin is
zero for zero covariance and grows linearly with the observation standard
deviation, which is exactly "bigger error, bigger buffer".

For map inflation / swept-path planning, where the bearing of the future
approach is unknown, an isotropic bound is used instead:

    sigma_iso = largest eigenvalue of C (worst-case LOS direction),
    m_iso = k * sqrt(sigma_iso^2 + sigma_r^2),

with ``k = z`` (normal one-sided bound) or ``k = sqrt(chi2_inv(1-alpha, dof=2))``
for a full disk that contains the true centre with probability ``1-alpha``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .predict import CircleObstacle

__all__ = ('UncertaintyMargin', 'relative_covariance', 'bearing_sigma',
           'isotropic_sigma', 'uncertainty_margin', 'uncertain_clearance',
           'inflate_uncertain', 'inverse_normal_cdf', 'chi2_quantile_2dof')

Matrix2 = tuple[tuple[float, float], tuple[float, float]]

# One-sided standard-normal quantiles for common confidence levels.
_Z_TABLE = {0.90: 1.2815515655446004,
            0.95: 1.6448536269514722,
            0.975: 1.959963984540054,
            0.99: 2.3263478740408408,
            0.999: 3.0902323061678135}


def inverse_normal_cdf(p: float) -> float:
    """Inverse standard-normal CDF via Acklam's rational approximation.

    One Newton refinement against ``Phi`` brings the relative error below ~1e-14.
    """
    if not 0.0 < p < 1.0:
        raise ValueError('probability must lie strictly between 0 and 1')
    rounded = round(p, 6)
    if rounded in _Z_TABLE:
        return _Z_TABLE[rounded]
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    # Newton step: x <- x - (Phi(x) - p) / phi(x)
    cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    return x - (cdf - p) / pdf


def chi2_quantile_2dof(probability: float) -> float:
    """Quantile of a chi-square distribution with 2 dof: ``-2 ln(1-p)``."""
    if not 0.0 < probability < 1.0:
        raise ValueError('probability must lie strictly between 0 and 1')
    return -2.0 * math.log(1.0 - probability)


def _as_matrix(cov: Matrix2 | None) -> Matrix2:
    if cov is None:
        return ((0.0, 0.0), (0.0, 0.0))
    (a, b), (c, d) = cov
    for v in (a, b, c, d):
        if not math.isfinite(v):
            raise ValueError('covariance contains non-finite entries')
    return ((float(a), float(b)), (float(c), float(d)))


def relative_covariance(*covariances: Matrix2 | None) -> Matrix2:
    """Covariance of a relative estimate when both ends are uncertain.

    Independent position errors add: ``Var(a - b) = Var(a) + Var(b)``.
    """
    total = [[0.0, 0.0], [0.0, 0.0]]
    for cov in covariances:
        matrix = _as_matrix(cov)
        for i in range(2):
            for j in range(2):
                total[i][j] += matrix[i][j]
    return ((total[0][0], total[0][1]), (total[1][0], total[1][1]))


def bearing_sigma(covariance: Matrix2 | None, bearing: tuple[float, float]) -> float:
    """Std-dev of the position error projected on the line-of-sight unit vector."""
    ux, uy = bearing
    norm = math.hypot(ux, uy)
    if norm <= 1e-15:
        # Coincident estimates: the LOS is undefined, fall back to the worst
        # direction so the result stays conservative.
        return isotropic_sigma(covariance)
    ux, uy = ux / norm, uy / norm
    (a, b), (c, d) = _as_matrix(covariance)
    variance = ux * ux * a + ux * uy * (b + c) + uy * uy * d
    return math.sqrt(max(0.0, variance))


def isotropic_sigma(covariance: Matrix2 | None) -> float:
    """Worst-case line-of-sight std-dev, i.e. the largest std-dev of the 2x2
    covariance (sqrt of its largest eigenvalue)."""
    (a, b), (c, d) = _as_matrix(covariance)
    trace, det = a + d, a * d - b * c
    discriminant = max(0.0, 0.25 * trace * trace - det)
    largest = 0.5 * trace + math.sqrt(discriminant)
    return math.sqrt(max(0.0, largest))


@dataclass(frozen=True)
class UncertaintyMargin:
    """One computed conservative margin plus every term used to build it.

    ``margin`` is the distance buffer; ``safe_distance`` is the threshold centre
    separation (nominal contact radius plus margin).  ``alarm`` is set when the
    measured centre distance is already at or below that threshold.
    """

    margin: float
    safe_distance: float
    measured_distance: float
    nominal_radius: float
    position_sigma: float
    radius_sigma: float
    total_sigma: float
    quantile: float
    confidence: float
    isotropic: bool
    alarm: bool

    def to_dict(self) -> dict:
        return {'margin': round(self.margin, 6),
                'safe_distance': round(self.safe_distance, 6),
                'measured_distance': round(self.measured_distance, 6),
                'nominal_radius': round(self.nominal_radius, 6),
                'position_sigma': round(self.position_sigma, 6),
                'radius_sigma': round(self.radius_sigma, 6),
                'total_sigma': round(self.total_sigma, 6),
                'quantile': round(self.quantile, 6),
                'confidence': round(self.confidence, 6),
                'isotropic': self.isotropic,
                'alarm': self.alarm}

    def explain(self) -> str:
        """Human-readable derivation of the numeric margin."""
        mode = 'isotropic (worst-case bearing)' if self.isotropic else 'line-of-sight'
        return (
            f'range uncertainty {self.position_sigma:.4f} m + size uncertainty '
            f'{self.radius_sigma:.4f} m -> combined sigma '
            f'{self.total_sigma:.4f} m; {self.confidence:.0%} one-sided quantile '
            f'z={self.quantile:.3f}; margin = z*sigma = {self.margin:.4f} m '
            f'({mode}); alarm threshold = contact {self.nominal_radius:.4f} m + '
            f'margin {self.margin:.4f} m = {self.safe_distance:.4f} m; measured '
            f'{self.measured_distance:.4f} m -> '
            f'{"ALARM" if self.alarm else "clear"}')


def uncertainty_margin(obstacle: CircleObstacle, robot_position: tuple[float, float] | None,
                       *, robot_radius: float = 0.0,
                       robot_covariance: Matrix2 | None = None,
                       confidence: float = 0.95, isotropic: bool = False,
                       bound: str = 'normal',
                       hard_buffer: float = 0.0) -> UncertaintyMargin:
    """Conservative safety distance for one observed obstacle.

    Parameters
    ----------
    obstacle:
        Observed circle; its ``covariance`` is the tracker posterior on the centre
        and ``radius_sigma`` the detector's size uncertainty.
    robot_position:
        Current/closest robot point. Required for the bearing-aware (non
        isotropic) bound; ignored when ``isotropic`` is true.
    robot_covariance:
        Optional localisation covariance; added to the obstacle covariance because
        the robot position is itself an estimate.
    confidence:
        Probability ``1 - alpha`` that the true geometry stays outside the margin.
    isotropic:
        When true the worst-case bearing (largest covariance eigenvalue) is used.
        Needed for map inflation where the future approach direction is unknown.
    bound:
        ``'normal'`` uses the one-sided Gaussian quantile ``Phi^{-1}(confidence)``
        (probability bound along the line of sight).  ``'chi2'`` is only valid with
        ``isotropic=True`` and inflates a disk that contains the true centre with
        the given joint probability (more conservative).
    hard_buffer:
        Extra deterministic clearance, added on top of the statistical margin.
    """
    if robot_radius < 0:
        raise ValueError('robot_radius must not be negative')
    if hard_buffer < 0:
        raise ValueError('hard_buffer must not be negative')
    if obstacle.radius < 0:
        raise ValueError('obstacle radius must not be negative')
    if obstacle.radius_sigma < 0:
        raise ValueError('radius_sigma must not be negative')
    if not 0.0 < confidence < 1.0:
        raise ValueError('confidence must lie strictly between 0 and 1')
    if bound == 'chi2' and not isotropic:
        raise ValueError("bound='chi2' requires isotropic=True")
    if bound not in ('normal', 'chi2'):
        raise ValueError("bound must be 'normal' or 'chi2'")

    cov = relative_covariance(obstacle.covariance, robot_covariance)
    if isotropic:
        position_sigma = isotropic_sigma(cov)
        if robot_position is None:
            # Map-inflation call: no specific viewpoint, so no live alarm check.
            measured = math.inf
        else:
            measured = math.hypot(obstacle.x - robot_position[0],
                                  obstacle.y - robot_position[1])
    else:
        if robot_position is None:
            raise ValueError('robot_position is required for the bearing-aware bound')
        measured = math.hypot(obstacle.x - robot_position[0],
                              obstacle.y - robot_position[1])
        position_sigma = bearing_sigma(cov, (obstacle.x - robot_position[0],
                                             obstacle.y - robot_position[1]))

    radius_sigma = float(obstacle.radius_sigma)
    total_sigma = math.hypot(position_sigma, radius_sigma)
    if isotropic and bound == 'chi2':
        quantile = math.sqrt(chi2_quantile_2dof(confidence))
    else:
        quantile = inverse_normal_cdf(confidence)
    nominal = robot_radius + obstacle.radius
    margin = hard_buffer + quantile * total_sigma
    safe_distance = nominal + margin
    return UncertaintyMargin(
        margin=margin, safe_distance=safe_distance, measured_distance=measured,
        nominal_radius=nominal, position_sigma=position_sigma,
        radius_sigma=radius_sigma, total_sigma=total_sigma, quantile=quantile,
        confidence=confidence, isotropic=isotropic,
        alarm=measured <= safe_distance + 1e-12)


def uncertain_clearance(obstacle: CircleObstacle, robot_position: tuple[float, float],
                        *, robot_radius: float = 0.0,
                        robot_covariance: Matrix2 | None = None,
                        confidence: float = 0.95,
                        hard_buffer: float = 0.0) -> UncertaintyMargin:
    """Bearing-aware alarm check at the robot's current position."""
    return uncertainty_margin(
        obstacle, robot_position, robot_radius=robot_radius,
        robot_covariance=robot_covariance, confidence=confidence,
        isotropic=False, bound='normal', hard_buffer=hard_buffer)


def inflate_uncertain(obstacles: Iterable[CircleObstacle], *,
                      robot_covariance: Matrix2 | None = None,
                      confidence: float = 0.95, bound: str = 'normal',
                      hard_buffer: float = 0.0) -> list[CircleObstacle]:
    """Inflate every obstacle by its uncertainty margin for swept-path planning.

    The inflation is isotropic because the planner does not know from which
    bearing the path will approach.  The robot's own (runtime) localisation
    covariance can be passed as ``robot_covariance``; the robot footprint radius
    is deliberately not added here — ``swept_path_conflict`` applies it the same
    way it does for the plain :func:`inflate`.  The returned circles carry no
    covariance: the uncertainty is now *inside* the radius, so feeding them back
    into this function never double-counts the margin.
    """
    inflated: list[CircleObstacle] = []
    for obstacle in obstacles:
        result = uncertainty_margin(
            obstacle, None, robot_covariance=robot_covariance,
            confidence=confidence, isotropic=True, bound=bound,
            hard_buffer=hard_buffer)
        inflated.append(CircleObstacle(
            obstacle.x, obstacle.y, obstacle.radius + result.margin,
            obstacle.vx, obstacle.vy, None, 0.0))
    return inflated
