"""Collision checking for swept motion and constant-velocity obstacles.

Checking only the sampled waypoints misses collisions that happen *between* two
waypoints, so every check works on the swept segment. Dynamic obstacles are checked
against the robot's relative motion, which is the only way to notice that two agents
converging head-on are dangerous even though each one alone moves predictably.

Observed obstacle positions are never exact: the noisier the sensor, the further the
true obstacle can be from its estimate, and treating the estimate as exact is what
makes an alarm fire only once the robot is already too close. Obstacles can therefore
carry an :class:`ObstacleUncertainty`, and :func:`inflate_uncertain` grows each radius
by a confidence-scaled margin *before* any of the checks above run, so a worse sensor
automatically means an earlier alarm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = ('CircleObstacle', 'Conflict', 'segment_point_distance', 'segment_circles_conflict',
           'swept_path_conflict', 'time_to_collision', 'inflate', 'minkowski_clearance',
           'ObstacleUncertainty', 'UncertainObstacle', 'confidence_scale', 'safety_margin',
           'inflate_uncertain', 'conservative_clearance')

Point = tuple[float, float]


@dataclass(frozen=True)
class CircleObstacle:
    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0

    def at(self, t: float) -> Point:
        return (self.x + self.vx * t, self.y + self.vy * t)


@dataclass(frozen=True)
class Conflict:
    segment: int
    obstacle: int
    distance: float
    point: Point

    def to_dict(self) -> dict:
        return {'segment': self.segment, 'obstacle': self.obstacle,
                'distance': round(self.distance, 6), 'point': [round(v, 6) for v in self.point]}


def segment_point_distance(a: Point, b: Point, p: Point) -> tuple[float, Point]:
    vx, vy = b[0] - a[0], b[1] - a[1]
    wx, wy = p[0] - a[0], p[1] - a[1]
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-18:
        return math.hypot(p[0] - a[0], p[1] - a[1]), a
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / length_sq))
    closest = (a[0] + t * vx, a[1] + t * vy)
    return math.hypot(p[0] - closest[0], p[1] - closest[1]), closest


def segment_circles_conflict(a: Point, b: Point, obstacles: Sequence[CircleObstacle],
                             robot_radius: float) -> Conflict | None:
    for index, obstacle in enumerate(obstacles):
        distance, closest = segment_point_distance(a, b, (obstacle.x, obstacle.y))
        if distance <= robot_radius + obstacle.radius + 1e-12:
            return Conflict(-1, index, distance, closest)
    return None


def swept_path_conflict(path: Sequence[Point], obstacles: Sequence[CircleObstacle],
                        robot_radius: float) -> Conflict | None:
    if robot_radius < 0:
        raise ValueError('robot_radius must not be negative')
    for segment, (a, b) in enumerate(zip(path, path[1:])):
        found = segment_circles_conflict(a, b, obstacles, robot_radius)
        if found is not None:
            return Conflict(segment, found.obstacle, found.distance, found.point)
    return None


def time_to_collision(robot: Point, robot_velocity: Point, obstacle: CircleObstacle, *,
                      horizon: float, robot_radius: float = 0.0) -> tuple[float, float]:
    """Earliest contact time inside ``horizon`` plus the separation at that moment.

    ``-1.0`` as the first element means "no contact inside the window"; the second
    element is then the closest approach distance.
    """
    if horizon <= 0:
        raise ValueError('horizon must be positive')
    dx = robot[0] - obstacle.x
    dy = robot[1] - obstacle.y
    dvx = robot_velocity[0] - obstacle.vx
    dvy = robot_velocity[1] - obstacle.vy
    radius = robot_radius + obstacle.radius
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (dx * dvx + dy * dvy)
    c = dx * dx + dy * dy - radius * radius
    if a <= 1e-18:
        closest = math.hypot(dx, dy)
        return (0.0 if closest <= radius else -1.0, closest)
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        t_closest = max(0.0, min(horizon, -b / (2 * a)))
        return (-1.0, math.hypot(dx + dvx * t_closest, dy + dvy * t_closest))
    root = math.sqrt(discriminant)
    for candidate in ((-b - root) / (2 * a), (-b + root) / (2 * a)):
        if 0.0 <= candidate <= horizon:
            return (candidate, math.hypot(dx + dvx * candidate, dy + dvy * candidate))
    t_closest = max(0.0, min(horizon, -b / (2 * a)))
    return (-1.0, math.hypot(dx + dvx * t_closest, dy + dvy * t_closest))


def inflate(obstacles: Iterable[CircleObstacle], margin: float) -> list[CircleObstacle]:
    if margin < 0:
        raise ValueError('margin must not be negative')
    return [CircleObstacle(o.x, o.y, o.radius + margin, o.vx, o.vy) for o in obstacles]


def minkowski_clearance(path: Sequence[Point], obstacles: Sequence[CircleObstacle]) -> float:
    """Smallest clearance along the path (negative when the path intersects)."""
    best = math.inf
    for a, b in zip(path, path[1:]):
        for obstacle in obstacles:
            distance, _ = segment_point_distance(a, b, (obstacle.x, obstacle.y))
            best = min(best, distance - obstacle.radius)
    return best


@dataclass(frozen=True)
class ObstacleUncertainty:
    """Observation uncertainty of a tracked obstacle.

    ``var_x``/``var_y``/``cov_xy`` form the 2D position covariance of the observed
    centre, ``radius_std`` is the standard deviation of the radius estimate and
    ``velocity_std`` the (isotropic) standard deviation of the velocity estimate,
    which is what lets the margin grow over a prediction horizon.
    """
    var_x: float
    var_y: float
    cov_xy: float = 0.0
    radius_std: float = 0.0
    velocity_std: float = 0.0

    def __post_init__(self) -> None:
        if self.var_x < 0 or self.var_y < 0:
            raise ValueError('variances must not be negative')
        if self.cov_xy * self.cov_xy > self.var_x * self.var_y + 1e-12:
            raise ValueError('covariance must be positive semi-definite')
        if self.radius_std < 0:
            raise ValueError('radius_std must not be negative')
        if self.velocity_std < 0:
            raise ValueError('velocity_std must not be negative')

    @classmethod
    def isotropic(cls, sigma: float, *, radius_std: float = 0.0,
                  velocity_std: float = 0.0) -> 'ObstacleUncertainty':
        """Equal uncertainty in every direction, the common single-sensor case."""
        return cls(sigma * sigma, sigma * sigma, 0.0, radius_std, velocity_std)

    def max_position_std(self, horizon: float = 0.0) -> float:
        """Worst-direction position standard deviation after ``horizon`` seconds.

        The worst direction is the largest eigenvalue of the 2x2 covariance
        (closed form for 2D), and velocity uncertainty adds ``velocity_std * t``
        in quadrature, the constant-velocity growth of the position error.
        """
        mean = (self.var_x + self.var_y) / 2.0
        spread = math.hypot((self.var_x - self.var_y) / 2.0, self.cov_xy)
        sigma_max = math.sqrt(max(0.0, mean + spread))
        return math.hypot(sigma_max, self.velocity_std * horizon)


def confidence_scale(confidence: float) -> float:
    """Radius multiplier containing ``confidence`` of a 2D Gaussian.

    The squared Mahalanobis distance of a 2D Gaussian follows a chi-square
    distribution with two degrees of freedom, whose quantile has the closed form
    ``sqrt(-2 ln(1 - p))``: 2.15 for p=0.90, 3.03 for p=0.99, 3.72 for p=0.999.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError('confidence must be in (0, 1)')
    return math.sqrt(-2.0 * math.log(1.0 - confidence))


def safety_margin(uncertainty: ObstacleUncertainty, *, confidence: float = 0.99,
                  horizon: float = 0.0) -> float:
    """Conservative inflation of an obstacle radius, in the obstacle's units.

    ``margin = k(p) * (sigma_pos(horizon) + radius_std)`` where
    ``sigma_pos(horizon) = sqrt(lambda_max(P) + (velocity_std * horizon)**2)``
    is the worst-direction position standard deviation and ``k(p)`` is the 2D
    confidence scale above: with probability ``p`` the true obstacle surface is
    inside the inflated circle, so checks against it alarm early rather than late.
    """
    if horizon < 0:
        raise ValueError('horizon must not be negative')
    k = confidence_scale(confidence)
    return k * (uncertainty.max_position_std(horizon) + uncertainty.radius_std)


@dataclass(frozen=True)
class UncertainObstacle:
    """A ``CircleObstacle`` together with the uncertainty of that observation."""
    obstacle: CircleObstacle
    uncertainty: ObstacleUncertainty

    def margin(self, *, confidence: float = 0.99, horizon: float = 0.0) -> float:
        return safety_margin(self.uncertainty, confidence=confidence, horizon=horizon)

    def inflated(self, *, confidence: float = 0.99, horizon: float = 0.0) -> CircleObstacle:
        o = self.obstacle
        return CircleObstacle(o.x, o.y, o.radius + self.margin(confidence=confidence, horizon=horizon),
                              o.vx, o.vy)


def inflate_uncertain(obstacles: Iterable[UncertainObstacle], *, confidence: float = 0.99,
                      horizon: float = 0.0) -> list[CircleObstacle]:
    """Plain circles inflated by their observation uncertainty, ready for any check."""
    return [o.inflated(confidence=confidence, horizon=horizon) for o in obstacles]


def conservative_clearance(path: Sequence[Point], obstacles: Sequence[UncertainObstacle], *,
                           confidence: float = 0.99, horizon: float = 0.0) -> float:
    """Clearance against uncertainty-inflated obstacles (negative inside the margin)."""
    return minkowski_clearance(path, inflate_uncertain(obstacles, confidence=confidence,
                                                       horizon=horizon))
