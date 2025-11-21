"""
truncated_hypervolume.py

2-D and 3-D hypervolume for minimization and maximization, with optional
truncation via aspiration (a) and reservation (w) levels.

Includes simple Monte Carlo estimators to verify the exact values
on small examples.
"""

from collections import defaultdict
import random
from typing import Iterable, List, Tuple, Optional

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


# ======================================================================
# 2-D hypervolume (minimization core)
# ======================================================================

def _hv2d_union_origin(extents: Iterable[Point2D]) -> float:
    """
    Given rectangles [0, X] x [0, Y] for each (X, Y) in extents (X>0, Y>0),
    return the area of their union.
    """
    pts = sorted(extents)  # sort by X ascending
    n = len(pts)
    if n == 0:
        return 0.0

    y_star = 0.0
    hv = 0.0
    x_prev = pts[-1][0]

    # Sweep from right (largest X) to left
    for i in range(n - 1, -1, -1):
        x_i, y_i = pts[i]
        if y_i > y_star:
            hv += (x_prev - x_i) * y_star
            y_star = y_i
            x_prev = x_i

    hv += x_prev * y_star
    return hv


def _hypervolume_2d_min_core(points: Iterable[Point2D],
                             ref: Point2D) -> float:
    """
    2-D hypervolume for minimization w.r.t. ref = (rx, ry).

    Each point p = (x, y) contributes rectangle [x, rx] x [y, ry].
    """
    rx, ry = map(float, ref)

    # Filter to ref-box
    P: List[Point2D] = [
        (float(x), float(y)) for (x, y) in points if x <= rx and y <= ry
    ]
    if not P:
        return 0.0

    # Transform [x, rx] x [y, ry] -> [0, X] x [0, Y] with X = rx - x, Y = ry - y
    trans: List[Point2D] = [(rx - x, ry - y) for (x, y) in P]
    trans = [(X, Y) for (X, Y) in trans if X > 0.0 and Y > 0.0]
    if not trans:
        return 0.0

    return _hv2d_union_origin(trans)


def hypervolume_2d_min(points: Iterable[Point2D],
                       ref: Point2D,
                       a: Optional[Point2D] = None,
                       w: Optional[Point2D] = None) -> float:
    """
    2-D hypervolume for minimization.

    If a and w are given, we project each point p into [a, w] componentwise:
        p'_j = min(max(p_j, a_j), w_j),
    and compute HV_min(P', ref=w).
    """
    if a is None or w is None:
        return _hypervolume_2d_min_core(points, ref)

    ax, ay = map(float, a)
    wx, wy = map(float, w)

    # Ensure a_j <= w_j
    ax, wx = min(ax, wx), max(ax, wx)
    ay, wy = min(ay, wy), max(ay, wy)

    trunc_points: List[Point2D] = []
    for x, y in points:
        x = float(x)
        y = float(y)
        tx = min(max(x, ax), wx)
        ty = min(max(y, ay), wy)
        trunc_points.append((tx, ty))

    trunc_ref = (wx, wy)
    return _hypervolume_2d_min_core(trunc_points, trunc_ref)


def hypervolume_2d(points: Iterable[Point2D],
                   ref: Point2D,
                   maximize: bool = False,
                   a: Optional[Point2D] = None,
                   w: Optional[Point2D] = None) -> float:
    """
    Unified 2-D hypervolume interface.

    Parameters
    ----------
    points : iterable of (x, y)
    ref    : reference point (rx, ry)
    maximize : if False (default), minimization; if True, maximization
    a, w   : optional aspiration and reservation levels (same orientation
             as ref and points). If omitted, standard hypervolume is used.

    For maximization, we reduce to minimization via sign flip.
    """
    if not maximize:
        return hypervolume_2d_min(points, ref, a=a, w=w)

    # Maximization via negation
    neg_points = [(-float(x), -float(y)) for (x, y) in points]
    rx, ry = ref
    neg_ref: Point2D = (-float(rx), -float(ry))

    if a is not None and w is not None:
        a_min: Point2D = tuple(-float(v) for v in a)  # type: ignore
        w_min: Point2D = tuple(-float(v) for v in w)  # type: ignore
    else:
        a_min = w_min = None

    return hypervolume_2d_min(neg_points, neg_ref, a=a_min, w=w_min)


def hypervolume_2d_max(points: Iterable[Point2D],
                       ref: Point2D = (0.0, 0.0),
                       a: Optional[Point2D] = None,
                       w: Optional[Point2D] = None) -> float:
    """
    Convenience wrapper for 2-D maximization hypervolume.
    """
    return hypervolume_2d(points, ref, maximize=True, a=a, w=w)


# ======================================================================
# 3-D hypervolume (minimization via z-sweep + 2-D HV)
# ======================================================================

def _hypervolume_3d_min_core(points: Iterable[Point3D],
                             ref: Point3D) -> float:
    """
    3-D hypervolume for minimization w.r.t. ref = (rx, ry, rz).

    Each point p = (x, y, z) contributes box [x, rx] x [y, ry] x [z, rz].

    Algorithm:
        - Group points by their z-coordinate.
        - Sweep z from smallest to largest.
        - At each z-level, add the corresponding (x, y) points and
          compute 2-D hypervolume HV2D of the active set w.r.t. (rx, ry).
        - The contribution of the slab [z_i, z_{i+1}] is
              HV2D(active) * (z_{i+1} - z_i).
    """
    rx, ry, rz = map(float, ref)

    # Filter to ref-box
    P: List[Point3D] = [
        (float(x), float(y), float(z))
        for (x, y, z) in points
        if x <= rx and y <= ry and z <= rz
    ]
    if not P:
        return 0.0

    by_z: defaultdict = defaultdict(list)
    for (x, y, z) in P:
        by_z[z].append((x, y))

    z_levels = sorted(by_z.keys())
    active_2d: List[Point2D] = []
    hv = 0.0

    for idx, z in enumerate(z_levels):
        # Add all (x, y) at this z-level
        active_2d.extend(by_z[z])
        hv2 = _hypervolume_2d_min_core(active_2d, (rx, ry))

        z_next = rz if idx == len(z_levels) - 1 else z_levels[idx + 1]
        hv += hv2 * (z_next - z)

    return hv


def hypervolume_3d_min(points: Iterable[Point3D],
                       ref: Point3D,
                       a: Optional[Point3D] = None,
                       w: Optional[Point3D] = None) -> float:
    """
    3-D hypervolume for minimization.

    If a and w are given, we project each point p into [a, w] componentwise:
        p'_j = min(max(p_j, a_j), w_j),
    and compute HV_min(P', ref=w).
    """
    if a is None or w is None:
        return _hypervolume_3d_min_core(points, ref)

    ax, ay, az = map(float, a)
    wx, wy, wz = map(float, w)

    # Ensure a_j <= w_j (aspiration better than reservation)
    ax, wx = min(ax, wx), max(ax, wx)
    ay, wy = min(ay, wy), max(ay, wy)
    az, wz = min(az, wz), max(az, wz)

    trunc_points: List[Point3D] = []
    for x, y, z in points:
        x = float(x)
        y = float(y)
        z = float(z)
        tx = min(max(x, ax), wx)
        ty = min(max(y, ay), wy)
        tz = min(max(z, az), wz)
        trunc_points.append((tx, ty, tz))

    trunc_ref = (wx, wy, wz)
    return _hypervolume_3d_min_core(trunc_points, trunc_ref)


# Backwards-compatible alias (from original code)
hypervolume_3d_min_arrays = hypervolume_3d_min


def hypervolume_3d(points: Iterable[Point3D],
                   ref: Point3D,
                   maximize: bool = False,
                   a: Optional[Point3D] = None,
                   w: Optional[Point3D] = None) -> float:
    """
    Unified 3-D hypervolume interface.

    Parameters
    ----------
    points : iterable of (x, y, z)
    ref    : reference point (rx, ry, rz)
    maximize : if False (default), minimization; if True, maximization
    a, w   : optional aspiration and reservation levels.

    For maximization, we reduce to minimization via sign flip.
    """
    if not maximize:
        return hypervolume_3d_min(points, ref, a=a, w=w)

    # Maximization via negation
    neg_points = [(-float(x), -float(y), -float(z)) for (x, y, z) in points]
    rx, ry, rz = ref
    neg_ref: Point3D = (-float(rx), -float(ry), -float(rz))

    if a is not None and w is not None:
        a_min: Point3D = tuple(-float(v) for v in a)  # type: ignore
        w_min: Point3D = tuple(-float(v) for v in w)  # type: ignore
    else:
        a_min = w_min = None

    return hypervolume_3d_min(neg_points, neg_ref, a=a_min, w=w_min)


def hypervolume_3d_max(points: Iterable[Point3D],
                       ref: Point3D = (0.0, 0.0, 0.0),
                       a: Optional[Point3D] = None,
                       w: Optional[Point3D] = None) -> float:
    """
    Convenience wrapper for 3-D maximization hypervolume.
    """
    return hypervolume_3d(points, ref, maximize=True, a=a, w=w)


# ======================================================================
# Monte Carlo estimators for verification
# ======================================================================

def _prepare_min_geometry_2d(points: Iterable[Point2D],
                             ref: Point2D,
                             maximize: bool = False,
                             a: Optional[Point2D] = None,
                             w: Optional[Point2D] = None) -> Tuple[List[Point2D], Point2D]:
    """
    Return (P_eff, ref_eff) in minimization space after any truncation.
    Used internally by the Monte Carlo estimators.
    """
    if maximize:
        neg_points = [(-float(x), -float(y)) for (x, y) in points]
        rx, ry = ref
        ref_min: Point2D = (-float(rx), -float(ry))
        if a is not None and w is not None:
            a_min: Point2D = tuple(-float(v) for v in a)  # type: ignore
            w_min: Point2D = tuple(-float(v) for v in w)  # type: ignore
        else:
            a_min = w_min = None
        points_min = neg_points
    else:
        points_min = [(float(x), float(y)) for (x, y) in points]
        ref_min = tuple(map(float, ref))
        a_min = tuple(map(float, a)) if a is not None else None
        w_min = tuple(map(float, w)) if w is not None else None

    # Apply truncation
    if a_min is not None and w_min is not None:
        ax, ay = a_min
        wx, wy = w_min
        ax, wx = min(ax, wx), max(ax, wx)
        ay, wy = min(ay, wy), max(ay, wy)
        proj_points: List[Point2D] = []
        for x, y in points_min:
            tx = min(max(x, ax), wx)
            ty = min(max(y, ay), wy)
            proj_points.append((tx, ty))
        ref_eff: Point2D = (wx, wy)
        P_eff = proj_points
    else:
        ref_eff = ref_min  # type: ignore
        P_eff = points_min

    # Filter to contributing points
    P_eff = [(x, y) for (x, y) in P_eff if x <= ref_eff[0] and y <= ref_eff[1]]
    return P_eff, ref_eff


def _prepare_min_geometry_3d(points: Iterable[Point3D],
                             ref: Point3D,
                             maximize: bool = False,
                             a: Optional[Point3D] = None,
                             w: Optional[Point3D] = None) -> Tuple[List[Point3D], Point3D]:
    """
    Return (P_eff, ref_eff) in minimization space after any truncation.
    Used internally by the Monte Carlo estimators.
    """
    if maximize:
        neg_points = [(-float(x), -float(y), -float(z)) for (x, y, z) in points]
        rx, ry, rz = ref
        ref_min: Point3D = (-float(rx), -float(ry), -float(rz))
        if a is not None and w is not None:
            a_min: Point3D = tuple(-float(v) for v in a)  # type: ignore
            w_min: Point3D = tuple(-float(v) for v in w)  # type: ignore
        else:
            a_min = w_min = None
        points_min = neg_points
    else:
        points_min = [(float(x), float(y), float(z)) for (x, y, z) in points]
        ref_min = tuple(map(float, ref))
        a_min = tuple(map(float, a)) if a is not None else None
        w_min = tuple(map(float, w)) if w is not None else None

    # Apply truncation
    if a_min is not None and w_min is not None:
        ax, ay, az = a_min
        wx, wy, wz = w_min
        ax, wx = min(ax, wx), max(ax, wx)
        ay, wy = min(ay, wy), max(ay, wy)
        az, wz = min(az, wz), max(az, wz)
        proj_points: List[Point3D] = []
        for x, y, z in points_min:
            tx = min(max(x, ax), wx)
            ty = min(max(y, ay), wy)
            tz = min(max(z, az), wz)
            proj_points.append((tx, ty, tz))
        ref_eff: Point3D = (wx, wy, wz)
        P_eff = proj_points
    else:
        ref_eff = ref_min  # type: ignore
        P_eff = points_min

    # Filter to contributing points
    P_eff = [
        (x, y, z)
        for (x, y, z) in P_eff
        if x <= ref_eff[0] and y <= ref_eff[1] and z <= ref_eff[2]
    ]
    return P_eff, ref_eff


def mc_estimate_hv_2d(points: Iterable[Point2D],
                      ref: Point2D,
                      maximize: bool = False,
                      a: Optional[Point2D] = None,
                      w: Optional[Point2D] = None,
                      n_samples: int = 100_000,
                      seed: int = 0) -> Tuple[float, float]:
    """
    Monte Carlo estimate of the 2-D hypervolume.

    Returns
    -------
    (hv_exact, hv_mc)
      hv_exact : exact hypervolume from the deterministic algorithm
      hv_mc    : Monte Carlo estimate based on sampling
    """
    hv_exact = hypervolume_2d(points, ref, maximize=maximize, a=a, w=w)
    P_eff, ref_eff = _prepare_min_geometry_2d(points, ref, maximize=maximize, a=a, w=w)

    if hv_exact == 0.0 or not P_eff:
        return hv_exact, 0.0

    # Bounding box for sampling
    lows = [ref_eff[0], ref_eff[1]]
    highs = [ref_eff[0], ref_eff[1]]
    for x, y in P_eff:
        lows[0] = min(lows[0], x)
        lows[1] = min(lows[1], y)
        highs[0] = max(highs[0], x)
        highs[1] = max(highs[1], y)

    vol_box = (highs[0] - lows[0]) * (highs[1] - lows[1])
    if vol_box <= 0.0:
        return hv_exact, 0.0

    rng = random.Random(seed)
    inside = 0

    for _ in range(n_samples):
        x = lows[0] + (highs[0] - lows[0]) * rng.random()
        y = lows[1] + (highs[1] - lows[1]) * rng.random()
        # Check membership in union of rectangles [p, ref_eff]
        for px, py in P_eff:
            if px <= x <= ref_eff[0] and py <= y <= ref_eff[1]:
                inside += 1
                break

    hv_mc = vol_box * inside / float(n_samples)
    return hv_exact, hv_mc


def mc_estimate_hv_3d(points: Iterable[Point3D],
                      ref: Point3D,
                      maximize: bool = False,
                      a: Optional[Point3D] = None,
                      w: Optional[Point3D] = None,
                      n_samples: int = 200_000,
                      seed: int = 0) -> Tuple[float, float]:
    """
    Monte Carlo estimate of the 3-D hypervolume.

    Returns
    -------
    (hv_exact, hv_mc)
      hv_exact : exact hypervolume from the deterministic algorithm
      hv_mc    : Monte Carlo estimate based on sampling
    """
    hv_exact = hypervolume_3d(points, ref, maximize=maximize, a=a, w=w)
    P_eff, ref_eff = _prepare_min_geometry_3d(points, ref, maximize=maximize, a=a, w=w)

    if hv_exact == 0.0 or not P_eff:
        return hv_exact, 0.0

    # Bounding box for sampling
    lows = [ref_eff[0], ref_eff[1], ref_eff[2]]
    highs = [ref_eff[0], ref_eff[1], ref_eff[2]]
    for x, y, z in P_eff:
        lows[0] = min(lows[0], x)
        lows[1] = min(lows[1], y)
        lows[2] = min(lows[2], z)
        highs[0] = max(highs[0], x)
        highs[1] = max(highs[1], y)
        highs[2] = max(highs[2], z)

    vol_box = (highs[0] - lows[0]) * (highs[1] - lows[1]) * (highs[2] - lows[2])
    if vol_box <= 0.0:
        return hv_exact, 0.0

    rng = random.Random(seed)
    inside = 0

    for _ in range(n_samples):
        x = lows[0] + (highs[0] - lows[0]) * rng.random()
        y = lows[1] + (highs[1] - lows[1]) * rng.random()
        z = lows[2] + (highs[2] - lows[2]) * rng.random()
        # Check membership in union of boxes [p, ref_eff]
        for px, py, pz in P_eff:
            if (px <= x <= ref_eff[0] and
                py <= y <= ref_eff[1] and
                pz <= z <= ref_eff[2]):
                inside += 1
                break

    hv_mc = vol_box * inside / float(n_samples)
    return hv_exact, hv_mc


# ======================================================================
# Examples + Monte Carlo verification
# ======================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 3-D example: maximization, no truncation
    # ------------------------------------------------------------------
    P3: List[Point3D] = [
        (1, 6, 4),
        (3, 5, 1),
        (4, 4, 6),
        (5, 2, 3),
        (6, 1, 5),
        (1, 3, 7),
        (2, 2, 8),
    ]
    ref3: Point3D = (0, 0, 0)

    hv_exact, hv_mc = mc_estimate_hv_3d(P3, ref3, maximize=True,
                                        n_samples=200_000, seed=1)
    rel_err = abs(hv_mc - hv_exact) / hv_exact if hv_exact != 0.0 else 0.0
    print("=== 3D example: maximization, no truncation ===")
    print(f"Exact HV   : {hv_exact:.6f}")
    print(f"MC estimate: {hv_mc:.6f}")
    print(f"Rel. error : {rel_err:.6e}")
    print()

    # ------------------------------------------------------------------
    # 3-D example: minimization with and without truncation
    # ------------------------------------------------------------------
    P3_min: List[Point3D] = [(3, 3, 3)]
    r3: Point3D = (5, 5, 5)
    a3: Point3D = (2, 2, 2)
    w3: Point3D = (4, 4, 4)

    # No truncation
    hv_exact, hv_mc = mc_estimate_hv_3d(P3_min, r3, maximize=False,
                                        n_samples=200_000, seed=2)
    rel_err = abs(hv_mc - hv_exact) / hv_exact if hv_exact != 0.0 else 0.0
    print("=== 3D example: minimization, no truncation ===")
    print(f"Exact HV   : {hv_exact:.6f}  (should be 8.0)")
    print(f"MC estimate: {hv_mc:.6f}")
    print(f"Rel. error : {rel_err:.6e}")
    print()

    # With truncation [a3, w3]
    hv_exact, hv_mc = mc_estimate_hv_3d(P3_min, r3, maximize=False,
                                        a=a3, w=w3,
                                        n_samples=200_000, seed=3)
    rel_err = abs(hv_mc - hv_exact) / hv_exact if hv_exact != 0.0 else 0.0
    print("=== 3D example: minimization, truncated [a,w] ===")
    print(f"Exact HV   : {hv_exact:.6f}  (should be 1.0)")
    print(f"MC estimate: {hv_mc:.6f}")
    print(f"Rel. error : {rel_err:.6e}")
    print()

    # ------------------------------------------------------------------
    # 2-D example: minimization with and without truncation
    # ------------------------------------------------------------------
    P2: List[Point2D] = [(1, 4), (3, 3), (1.5, 3.5)]
    r2: Point2D = (5, 5)
    a2: Point2D = (2, 2)
    w2: Point2D = (4, 4)

    # No truncation
    hv_exact, hv_mc = mc_estimate_hv_2d(P2, r2, maximize=False,
                                        n_samples=200_000, seed=4)
    rel_err = abs(hv_mc - hv_exact) / hv_exact if hv_exact != 0.0 else 0.0
    print("=== 2D example: minimization, no truncation ===")
    print(f"Exact HV   : {hv_exact:.6f}  (should be 6.75)")
    print(f"MC estimate: {hv_mc:.6f}")
    print(f"Rel. error : {rel_err:.6e}")
    print()

    # With truncation [a2, w2]
    hv_exact, hv_mc = mc_estimate_hv_2d(P2, r2, maximize=False,
                                        a=a2, w=w2,
                                        n_samples=200_000, seed=5)
    rel_err = abs(hv_mc - hv_exact) / hv_exact if hv_exact != 0.0 else 0.0
    print("=== 2D example: minimization, truncated [a,w] ===")
    print(f"Exact HV   : {hv_exact:.6f}  (should be 1.5)")
    print(f"MC estimate: {hv_mc:.6f}")
    print(f"Rel. error : {rel_err:.6e}")
    print()
