"""
truncated_hypervolume.py

2-D and 3-D hypervolume for minimization and maximization, with optional
ART truncation via aspiration (a) and reservation (w) levels.

Mathematically, for minimization we consider the dominated region

    D(P) = { z in R^m : exists p in P with p <= z <= r },

and the clipping region

    C_min = H1 ∪ H2 ∪ H3

with
    H1 = (-∞, a_1] x ... x (-∞, a_m],
    H2 = [a_1, w_1] x ... x [a_m, w_m],
    H3 = [w_1, r_1] x ... x [w_m, r_m],

and define the ART-Hypervolume as

    HV_ART_min(P; r, a, w) = volume( C_min ∩ D(P) ).

For maximization we reduce to minimization via sign flip.

This file provides:
- Exact 2D / 3D hypervolume (standard and ART).
- Monte Carlo estimators for verification on toy examples.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Tuple, Optional

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


# ======================================================================
# 2-D hypervolume (minimization core, no truncation)
# ======================================================================

def _hv2d_union_origin(extents: Iterable[Point2D]) -> float:
    """
    Area of union of rectangles [0, X] x [0, Y] for (X, Y) in extents.
    Assumes X > 0, Y > 0.
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

    # Transform [x, rx] x [y, ry] -> [0, X] x [0, Y]
    trans: List[Point2D] = [(rx - x, ry - y) for (x, y) in P]
    trans = [(X, Y) for (X, Y) in trans if X > 0.0 and Y > 0.0]
    if not trans:
        return 0.0

    return _hv2d_union_origin(trans)


# ======================================================================
# 2-D ART Hypervolume (minimization)
# ======================================================================

def _hypervolume_2d_min_art(points: Iterable[Point2D],
                            ref: Point2D,
                            a: Point2D,
                            w: Point2D) -> float:
    """
    2-D ART-Hypervolume for minimization:

        HV_ART_min(P; r, a, w) = volume( C_min ∩ D(P) ),

    where
      D(P) = union_{p in P} [p, r],
      C_min = H1 ∪ H2 ∪ H3 with
        H1 = (-∞, a]^2,
        H2 = [a, w]^2,
        H3 = [w, r]^2.

    Implemented as:
      HV1 = HV_min( {p <= a}, ref=a )              from H1
      HV2 = HV_min( {q = max(p, a), p <= w}, ref=w ) from H2
      HV3 = HV_min( {r' = max(p, w)}, ref=r )        from H3
      HV_ART = HV1 + HV2 + HV3
    """
    rx, ry = map(float, ref)
    ax, ay = map(float, a)
    wx, wy = map(float, w)

    # Enforce a <= w <= r componentwise
    wx = min(wx, rx)
    wy = min(wy, ry)
    ax = min(ax, wx)
    ay = min(ay, wy)

    # Filter points to the overall ref-box
    P: List[Point2D] = [
        (float(x), float(y)) for (x, y) in points if x <= rx and y <= ry
    ]
    if not P:
        return 0.0

    # H1: (-∞, a]^2  → HV1 with ref=a, points p with p <= a
    P1: List[Point2D] = [(x, y) for (x, y) in P if x <= ax and y <= ay]
    hv1 = _hypervolume_2d_min_core(P1, (ax, ay)) if P1 else 0.0

    # H2: [a, w]^2 → HV2 with ref=w, points q = max(p, a) for p <= w
    P2_prime: List[Point2D] = []
    for x, y in P:
        if x <= wx and y <= wy:
            qx = max(x, ax)
            qy = max(y, ay)
            P2_prime.append((qx, qy))
    hv2 = _hypervolume_2d_min_core(P2_prime, (wx, wy)) if P2_prime else 0.0

    # H3: [w, r]^2 → HV3 with ref=r, points r' = max(p, w)
    P3_prime: List[Point2D] = [(max(x, wx), max(y, wy)) for (x, y) in P]
    hv3 = _hypervolume_2d_min_core(P3_prime, (rx, ry)) if P3_prime else 0.0

    return hv1 + hv2 + hv3


def hypervolume_2d_min(points: Iterable[Point2D],
                       ref: Point2D,
                       a: Optional[Point2D] = None,
                       w: Optional[Point2D] = None) -> float:
    """
    2-D hypervolume for minimization.

    If a and w are None:
        standard hypervolume HV_min(P; ref).

    If a and w are given:
        ART-Hypervolume HV_ART_min(P; ref, a, w) as C_min ∩ D(P).
    """
    if a is None or w is None:
        return _hypervolume_2d_min_core(points, ref)
    return _hypervolume_2d_min_art(points, ref, a, w)


def hypervolume_2d(points: Iterable[Point2D],
                   ref: Point2D,
                   maximize: bool = False,
                   a: Optional[Point2D] = None,
                   w: Optional[Point2D] = None) -> float:
    """
    Unified 2-D hypervolume interface.

    Parameters
    ----------
    points   : iterable of (x, y)
    ref      : reference point (rx, ry)
    maximize : if False (default), minimization; if True, maximization
    a, w     : optional aspiration and reservation levels.

    - If maximize == False:
        - (a, w) omitted → standard HV_min.
        - (a, w) given   → ART-HV_min (C_min ∩ D(P)).

    - If maximize == True:
        We map to minimization via sign flip and apply the same logic.
    """
    if not maximize:
        return hypervolume_2d_min(points, ref, a=a, w=w)

    # Maximization via negation
    neg_points = [(-float(x), -float(y)) for (x, y) in points]
    rx, ry = ref
    neg_ref: Point2D = (-float(rx), -float(ry))

    if a is not None and w is not None:
        a_min: Point2D = (-float(a[0]), -float(a[1]))
        w_min: Point2D = (-float(w[0]), -float(w[1]))
    else:
        a_min = w_min = None

    return hypervolume_2d_min(neg_points, neg_ref, a=a_min, w=w_min)


def hypervolume_2d_max(points: Iterable[Point2D],
                       ref: Point2D = (0.0, 0.0),
                       a: Optional[Point2D] = None,
                       w: Optional[Point2D] = None) -> float:
    """Convenience wrapper for 2-D maximization hypervolume."""
    return hypervolume_2d(points, ref, maximize=True, a=a, w=w)


# ======================================================================
# 3-D hypervolume (minimization core, no truncation)
# ======================================================================

def _hypervolume_3d_min_core(points: Iterable[Point3D],
                             ref: Point3D) -> float:
    """
    3-D hypervolume for minimization w.r.t. ref = (rx, ry, rz).

    Each point p = (x, y, z) contributes box [x, rx] x [y, ry] x [z, rz].

    Simple algorithm:
      - Group points by their z-coordinate.
      - Sweep z from smallest to largest.
      - At each z-level, maintain active 2-D set and compute HV2D
        with _hypervolume_2d_min_core.
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

    from collections import defaultdict
    by_z: dict[float, List[Point2D]] = defaultdict(list)
    for (x, y, z) in P:
        by_z[z].append((x, y))

    z_levels = sorted(by_z.keys())
    active_2d: List[Point2D] = []
    hv = 0.0

    for idx, z in enumerate(z_levels):
        active_2d.extend(by_z[z])
        hv2 = _hypervolume_2d_min_core(active_2d, (rx, ry))
        z_next = rz if idx == len(z_levels) - 1 else z_levels[idx + 1]
        hv += hv2 * (z_next - z)

    return hv


# ======================================================================
# 3-D ART Hypervolume (minimization)
# ======================================================================

def _hypervolume_3d_min_art(points: Iterable[Point3D],
                            ref: Point3D,
                            a: Point3D,
                            w: Point3D) -> float:
    """
    3-D ART-Hypervolume for minimization:

      HV_ART_min(P; r, a, w) = volume( C_min ∩ D(P) ),

    with
      C_min = H1 ∪ H2 ∪ H3,
      H1 = (-∞, a]^3,
      H2 = [a, w]^3,
      H3 = [w, r]^3.

    Implemented as:
      HV1 = HV_min( {p <= a}, ref=a )
      HV2 = HV_min( {q = max(p, a), p <= w}, ref=w )
      HV3 = HV_min( {r' = max(p, w)}, ref=r )
      HV_ART = HV1 + HV2 + HV3.
    """
    rx, ry, rz = map(float, ref)
    ax, ay, az = map(float, a)
    wx, wy, wz = map(float, w)

    # Enforce a <= w <= r componentwise
    wx = min(wx, rx)
    wy = min(wy, ry)
    wz = min(wz, rz)
    ax = min(ax, wx)
    ay = min(ay, wy)
    az = min(az, wz)

    # Filter points to the overall ref-box
    P: List[Point3D] = [
        (float(x), float(y), float(z))
        for (x, y, z) in points
        if x <= rx and y <= ry and z <= rz
    ]
    if not P:
        return 0.0

    # H1
    P1: List[Point3D] = [
        (x, y, z) for (x, y, z) in P if x <= ax and y <= ay and z <= az
    ]
    hv1 = _hypervolume_3d_min_core(P1, (ax, ay, az)) if P1 else 0.0

    # H2
    P2_prime: List[Point3D] = []
    for x, y, z in P:
        if x <= wx and y <= wy and z <= wz:
            qx = max(x, ax)
            qy = max(y, ay)
            qz = max(z, az)
            P2_prime.append((qx, qy, qz))
    hv2 = _hypervolume_3d_min_core(P2_prime, (wx, wy, wz)) if P2_prime else 0.0

    # H3
    P3_prime: List[Point3D] = [
        (max(x, wx), max(y, wy), max(z, wz)) for (x, y, z) in P
    ]
    hv3 = _hypervolume_3d_min_core(P3_prime, (rx, ry, rz)) if P3_prime else 0.0

    return hv1 + hv2 + hv3


def hypervolume_3d_min(points: Iterable[Point3D],
                       ref: Point3D,
                       a: Optional[Point3D] = None,
                       w: Optional[Point3D] = None) -> float:
    """
    3-D hypervolume for minimization.

    If a and w are None:
        standard HV_min(P; ref).

    If a and w are given:
        ART-Hypervolume HV_ART_min(P; ref, a, w) as C_min ∩ D(P).
    """
    if a is None or w is None:
        return _hypervolume_3d_min_core(points, ref)
    return _hypervolume_3d_min_art(points, ref, a, w)


# Backward-compatible alias
hypervolume_3d_min_arrays = hypervolume_3d_min


def hypervolume_3d(points: Iterable[Point3D],
                   ref: Point3D,
                   maximize: bool = False,
                   a: Optional[Point3D] = None,
                   w: Optional[Point3D] = None) -> float:
    """
    Unified 3-D hypervolume interface.

    - If maximize == False:
        (a, w) omitted → standard HV_min.
        (a, w) given   → ART-HV_min.

    - If maximize == True:
        Reduce to minimization by sign flip.
    """
    if not maximize:
        return hypervolume_3d_min(points, ref, a=a, w=w)

    # Maximization via negation
    neg_points = [(-float(x), -float(y), -float(z)) for (x, y, z) in points]
    rx, ry, rz = ref
    neg_ref: Point3D = (-float(rx), -float(ry), -float(rz))

    if a is not None and w is not None:
        a_min: Point3D = (-float(a[0]), -float(a[1]), -float(a[2]))
        w_min: Point3D = (-float(w[0]), -float(w[1]), -float(w[2]))
    else:
        a_min = w_min = None

    return hypervolume_3d_min(neg_points, neg_ref, a=a_min, w=w_min)


def hypervolume_3d_max(points: Iterable[Point3D],
                       ref: Point3D = (0.0, 0.0, 0.0),
                       a: Optional[Point3D] = None,
                       w: Optional[Point3D] = None) -> float:
    """Convenience wrapper for 3-D maximization hypervolume."""
    return hypervolume_3d(points, ref, maximize=True, a=a, w=w)


# ======================================================================
# Monte Carlo estimators (minimization core)
# ======================================================================

def _mc_box_2d(points: List[Point2D],
               ref: Point2D,
               low: Point2D,
               high: Point2D,
               n_samples: int,
               rng: random.Random) -> float:
    """
    Monte Carlo estimate of volume( D(P) ∩ [low, high] )
    in the minimization setting with reference ref.
    """
    if n_samples <= 0:
        return 0.0

    lx, ly = low
    ux, uy = high
    if ux <= lx or uy <= ly:
        return 0.0

    rx, ry = ref
    vol_box = (ux - lx) * (uy - ly)
    inside = 0

    for _ in range(n_samples):
        x = lx + (ux - lx) * rng.random()
        y = ly + (uy - ly) * rng.random()
        # Membership in dominated region D(P)
        for px, py in points:
            if px <= x <= rx and py <= y <= ry:
                inside += 1
                break

    return vol_box * inside / float(n_samples)


def _mc_estimate_hv_2d_min(points: Iterable[Point2D],
                           ref: Point2D,
                           a: Optional[Point2D],
                           w: Optional[Point2D],
                           n_samples: int,
                           seed: int) -> float:
    """
    Monte Carlo estimate in the **minimization** setting of:

      - HV_min(P; ref)             if a, w are None
      - HV_ART_min(P; ref, a, w)   if a, w given
    """
    P: List[Point2D] = [(float(x), float(y)) for (x, y) in points]
    if not P:
        return 0.0

    rx, ry = map(float, ref)
    rng = random.Random(seed)

    if a is None or w is None:
        # Standard HV: single bounding box
        min_x = min(x for x, _ in P)
        min_y = min(y for _, y in P)
        return _mc_box_2d(P, (rx, ry), (min_x, min_y), (rx, ry),
                          n_samples, rng)

    # ART: decompose into three regions
    ax, ay = map(float, a)
    wx, wy = map(float, w)

    wx = min(wx, rx)
    wy = min(wy, ry)
    ax = min(ax, wx)
    ay = min(ay, wy)

    min_x = min(x for x, _ in P)
    min_y = min(y for _, y in P)

    # Split samples roughly equally
    n1 = n_samples // 3
    n2 = n_samples // 3
    n3 = n_samples - n1 - n2

    hv_mc = 0.0

    # Region H1 ∩ D(P) subset of [minP, a]
    if min_x < ax and min_y < ay and n1 > 0:
        hv_mc += _mc_box_2d(P, (rx, ry), (min_x, min_y), (ax, ay), n1, rng)

    # Region H2 ∩ D(P) subset of [a, w]
    if ax < wx and ay < wy and n2 > 0:
        hv_mc += _mc_box_2d(P, (rx, ry), (ax, ay), (wx, wy), n2, rng)

    # Region H3 ∩ D(P) subset of [w, r]
    if wx < rx and wy < ry and n3 > 0:
        hv_mc += _mc_box_2d(P, (rx, ry), (wx, wy), (rx, ry), n3, rng)

    return hv_mc


def _mc_box_3d(points: List[Point3D],
               ref: Point3D,
               low: Point3D,
               high: Point3D,
               n_samples: int,
               rng: random.Random) -> float:
    """
    Monte Carlo estimate of volume( D(P) ∩ [low, high] )
    in the 3-D minimization setting with reference ref.
    """
    if n_samples <= 0:
        return 0.0

    lx, ly, lz = low
    ux, uy, uz = high
    if ux <= lx or uy <= ly or uz <= lz:
        return 0.0

    rx, ry, rz = ref
    vol_box = (ux - lx) * (uy - ly) * (uz - lz)
    inside = 0

    for _ in range(n_samples):
        x = lx + (ux - lx) * rng.random()
        y = ly + (uy - ly) * rng.random()
        z = lz + (uz - lz) * rng.random()
        for px, py, pz in points:
            if (px <= x <= rx and
                py <= y <= ry and
                pz <= z <= rz):
                inside += 1
                break

    return vol_box * inside / float(n_samples)


def _mc_estimate_hv_3d_min(points: Iterable[Point3D],
                           ref: Point3D,
                           a: Optional[Point3D],
                           w: Optional[Point3D],
                           n_samples: int,
                           seed: int) -> float:
    """
    Monte Carlo estimate in the **minimization** setting of:

      - HV_min(P; ref)             if a, w are None
      - HV_ART_min(P; ref, a, w)   if a, w given
    """
    P: List[Point3D] = [(float(x), float(y), float(z)) for (x, y, z) in points]
    if not P:
        return 0.0

    rx, ry, rz = map(float, ref)
    rng = random.Random(seed)

    if a is None or w is None:
        min_x = min(x for x, _, _ in P)
        min_y = min(y for _, y, _ in P)
        min_z = min(z for _, _, z in P)
        return _mc_box_3d(P, (rx, ry, rz),
                          (min_x, min_y, min_z),
                          (rx, ry, rz),
                          n_samples, rng)

    ax, ay, az = map(float, a)
    wx, wy, wz = map(float, w)

    wx = min(wx, rx)
    wy = min(wy, ry)
    wz = min(wz, rz)
    ax = min(ax, wx)
    ay = min(ay, wy)
    az = min(az, wz)

    min_x = min(x for x, _, _ in P)
    min_y = min(y for _, y, _ in P)
    min_z = min(z for _, _, z in P)

    n1 = n_samples // 3
    n2 = n_samples // 3
    n3 = n_samples - n1 - n2

    hv_mc = 0.0

    # H1 region
    if min_x < ax and min_y < ay and min_z < az and n1 > 0:
        hv_mc += _mc_box_3d(P, (rx, ry, rz),
                            (min_x, min_y, min_z),
                            (ax, ay, az),
                            n1, rng)

    # H2 region
    if ax < wx and ay < wy and az < wz and n2 > 0:
        hv_mc += _mc_box_3d(P, (rx, ry, rz),
                            (ax, ay, az),
                            (wx, wy, wz),
                            n2, rng)

    # H3 region
    if wx < rx and wy < ry and wz < rz and n3 > 0:
        hv_mc += _mc_box_3d(P, (rx, ry, rz),
                            (wx, wy, wz),
                            (rx, ry, rz),
                            n3, rng)

    return hv_mc


# ======================================================================
# Public Monte Carlo interfaces (min & max)
# ======================================================================

def mc_estimate_hv_2d(points: Iterable[Point2D],
                      ref: Point2D,
                      maximize: bool = False,
                      a: Optional[Point2D] = None,
                      w: Optional[Point2D] = None,
                      n_samples: int = 100_000,
                      seed: int = 0) -> Tuple[float, float]:
    """
    Monte Carlo verification of 2-D hypervolume.

    Returns
    -------
    (hv_exact, hv_mc)
    """
    hv_exact = hypervolume_2d(points, ref, maximize=maximize, a=a, w=w)

    if maximize:
        # Map to minimization
        neg_points = [(-float(x), -float(y)) for (x, y) in points]
        rx, ry = ref
        neg_ref: Point2D = (-float(rx), -float(ry))
        if a is not None and w is not None:
            a_min: Point2D = (-float(a[0]), -float(a[1]))
            w_min: Point2D = (-float(w[0]), -float(w[1]))
        else:
            a_min = w_min = None
        hv_mc = _mc_estimate_hv_2d_min(neg_points, neg_ref,
                                       a_min, w_min, n_samples, seed)
    else:
        hv_mc = _mc_estimate_hv_2d_min(points, ref, a, w, n_samples, seed)

    return hv_exact, hv_mc


def mc_estimate_hv_3d(points: Iterable[Point3D],
                      ref: Point3D,
                      maximize: bool = False,
                      a: Optional[Point3D] = None,
                      w: Optional[Point3D] = None,
                      n_samples: int = 200_000,
                      seed: int = 0) -> Tuple[float, float]:
    """
    Monte Carlo verification of 3-D hypervolume.

    Returns
    -------
    (hv_exact, hv_mc)
    """
    hv_exact = hypervolume_3d(points, ref, maximize=maximize, a=a, w=w)

    if maximize:
        neg_points = [(-float(x), -float(y), -float(z)) for (x, y, z) in points]
        rx, ry, rz = ref
        neg_ref: Point3D = (-float(rx), -float(ry), -float(rz))
        if a is not None and w is not None:
            a_min: Point3D = (-float(a[0]), -float(a[1]), -float(a[2]))
            w_min: Point3D = (-float(w[0]), -float(w[1]), -float(w[2]))
        else:
            a_min = w_min = None
        hv_mc = _mc_estimate_hv_3d_min(neg_points, neg_ref,
                                       a_min, w_min, n_samples, seed)
    else:
        hv_mc = _mc_estimate_hv_3d_min(points, ref, a, w, n_samples, seed)

    return hv_exact, hv_mc


# ======================================================================
# Examples + Monte Carlo verification
# ======================================================================

if __name__ == "__main__":
    # --------------------------------------------------------------
    # 3-D example: maximization, no truncation (standard HV)
    # --------------------------------------------------------------
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
    print(f"Exact HV   : {hv_exact:.6f}  (expected 128.0)")
    print(f"MC estimate: {hv_mc:.6f}")
    print(f"Rel. error : {rel_err:.6e}")
    print()

    # --------------------------------------------------------------
    # 3-D example: minimization with ART truncation
    # P3_min = {(3,3,3)}, r=(5,5,5), a=(2,2,2), w=(4,4,4)
    # Standard HV_min = 8
    # ART-HV_min = volume([3,4]^3) + volume([4,5]^3) = 1 + 1 = 2
    # --------------------------------------------------------------
    P3_min: List[Point3D] = [(3, 3, 3)]
    r3: Point3D = (5, 5, 5)
    a3: Point3D = (2, 2, 2)
    w3: Point3D = (4, 4, 4)

    hv_exact_std, hv_mc_std = mc_estimate_hv_3d(P3_min, r3, maximize=False,
                                                n_samples=200_000, seed=2)
    rel_err_std = (abs(hv_mc_std - hv_exact_std) / hv_exact_std
                   if hv_exact_std != 0.0 else 0.0)
    print("=== 3D example: minimization, standard HV ===")
    print(f"Exact HV   : {hv_exact_std:.6f}  (expected 8.0)")
    print(f"MC estimate: {hv_mc_std:.6f}")
    print(f"Rel. error : {rel_err_std:.6e}")
    print()

    hv_exact_art, hv_mc_art = mc_estimate_hv_3d(P3_min, r3, maximize=False,
                                                a=a3, w=w3,
                                                n_samples=200_000, seed=3)
    rel_err_art = (abs(hv_mc_art - hv_exact_art) / hv_exact_art
                   if hv_exact_art != 0.0 else 0.0)
    print("=== 3D example: minimization, ART-HV [a,w] ===")
    print(f"Exact ART-HV   : {hv_exact_art:.6f}  (expected 2.0)")
    print(f"MC estimate    : {hv_mc_art:.6f}")
    print(f"Rel. error     : {rel_err_art:.6e}")
    print()

    # --------------------------------------------------------------
    # 2-D example: minimization with ART truncation
    # P2 = {(1,4),(3,3),(1.5,3.5)}, r=(5,5), a=(2,2), w=(4,4)
    # Standard HV_min = 6.75
    # ART-HV_min = 2.5 (see LaTeX derivation)
    # --------------------------------------------------------------
    P2: List[Point2D] = [(1, 4), (3, 3), (1.5, 3.5)]
    r2: Point2D = (5, 5)
    a2: Point2D = (2, 2)
    w2: Point2D = (4, 4)

    hv_exact2_std, hv_mc2_std = mc_estimate_hv_2d(P2, r2, maximize=False,
                                                  n_samples=200_000, seed=4)
    rel_err2_std = (abs(hv_mc2_std - hv_exact2_std) / hv_exact2_std
                    if hv_exact2_std != 0.0 else 0.0)
    print("=== 2D example: minimization, standard HV ===")
    print(f"Exact HV   : {hv_exact2_std:.6f}  (expected 6.75)")
    print(f"MC estimate: {hv_mc2_std:.6f}")
    print(f"Rel. error : {rel_err2_std:.6e}")
    print()

    hv_exact2_art, hv_mc2_art = mc_estimate_hv_2d(P2, r2, maximize=False,
                                                  a=a2, w=w2,
                                                  n_samples=200_000, seed=5)
    rel_err2_art = (abs(hv_mc2_art - hv_exact2_art) / hv_exact2_art
                    if hv_exact2_art != 0.0 else 0.0)
    print("=== 2D example: minimization, ART-HV [a,w] ===")
    print(f"Exact ART-HV   : {hv_exact2_art:.6f}  (expected 2.5)")
    print(f"MC estimate    : {hv_mc2_art:.6f}")
    print(f"Rel. error     : {rel_err2_art:.6e}")
    print()
