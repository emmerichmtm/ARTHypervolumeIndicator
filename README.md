# ARC Hypervolume (2D & 3D)

This repository provides a small, self-contained Python module for computing

- standard **2-D / 3-D hypervolume**, and  
- the **ARC-Hypervolume** (Aspiration–Reservation **Clipping** Hypervolume)

for both **minimization** and **maximization** problems.

The code also includes simple **Monte Carlo estimators** to numerically verify
the exact values on toy examples.

---

## Files

- `truncated_hypervolume.py`  
  Core implementation of:
  - 2-D and 3-D **standard hypervolume**,  
  - 2-D and 3-D **ARC-Hypervolume**,  
  - Monte Carlo verification routines,  
  plus example calls in the `__main__` block that reproduce the values
  used in the LaTeX report.

---

## Installation & Quick Test

Clone or copy the file and run it with Python 3:

```bash
python truncated_hypervolume.py
```

You should see output similar to:

```text
=== 3D example: maximization, no clipping (standard HV) ===
Exact HV   : 128.000000  (expected 128.0)
MC estimate: 127.8...
Rel. error : ~1e-3

=== 3D example: minimization, standard HV ===
Exact HV   : 8.000000  (expected 8.0)
MC estimate: 8.0...
Rel. error : ~1e-3 or smaller

=== 3D example: minimization, ARC-HV [a,w] ===
Exact ARC-HV   : 2.000000  (expected 2.0)
MC estimate    : 2.0...
Rel. error     : ~1e-3 or smaller

=== 2D example: minimization, standard HV ===
Exact HV   : 6.750000  (expected 6.75)
MC estimate: 6.74...
Rel. error : ~1e-3 or smaller

=== 2D example: minimization, ARC-HV [a,w] ===
Exact ARC-HV   : 2.500000  (expected 2.5)
MC estimate    : 2.49...
Rel. error     : ~1e-3 or smaller
```

(Monte Carlo values vary slightly depending on the random seed and
number of samples.)

---

## API Overview

All user-facing functions are defined in `truncated_hypervolume.py`:

```python
from truncated_hypervolume import (
    # 2-D
    hypervolume_2d, hypervolume_2d_min, hypervolume_2d_max,
    # 3-D
    hypervolume_3d, hypervolume_3d_min, hypervolume_3d_max,
    # Monte Carlo checks
    mc_estimate_hv_2d, mc_estimate_hv_3d,
)
```

### 2-D hypervolume

```python
hv = hypervolume_2d(
    points,        # iterable of (x, y)
    ref,           # reference point (rx, ry)
    maximize=False,
    a=None,        # optional aspiration level (ax, ay)
    w=None,        # optional reservation level (wx, wy)
)
```

- If `maximize=False` (default), the problem is treated as **minimization**.  
- If `maximize=True`, the problem is treated as **maximization** via a sign flip.  
- If `a` and `w` are **not** provided, you get the **standard hypervolume**.  
- If `a` and `w` are provided, you get the **ARC-Hypervolume**:
  \(\lambda(C_{\min} \cap D(P))\) in minimization space.

Convenience wrappers:

```python
hv_min = hypervolume_2d_min(points, ref, a=a, w=w)
hv_max = hypervolume_2d_max(points, ref, a=a, w=w)
```

### 3-D hypervolume

```python
hv = hypervolume_3d(
    points,        # iterable of (x, y, z)
    ref,           # reference point (rx, ry, rz)
    maximize=False,
    a=None,
    w=None,
)
```

Convenience wrappers:

```python
hv_min = hypervolume_3d_min(points, ref, a=a, w=w)
hv_max = hypervolume_3d_max(points, ref, a=a, w=w)
```

---

## ARC-Hypervolume: Concept

### Standard hypervolume (minimization)

For a finite set \(P = \{p^1,\dots,p^n\} \subset \mathbb{R}^m\) and a reference
point \(r \in \mathbb{R}^m\) with \(p^i_j \le r_j\) for all components, the
dominated region is

\[
D(P) := \bigcup_{i=1}^n [p^i_1, r_1] \times \dots \times [p^i_m, r_m].
\]

The standard minimization hypervolume is

\[
\mathrm{HV}_{\min}(P; r) := \lambda^m(D(P)).
\]

### Clipping region \(C_{\min}\)

For **minimization**, we introduce:

- aspiration vector \(a\), reservation vector \(w\),
- with componentwise ordering \(a_j \le w_j \le r_j\).

We define three axis-aligned “preference boxes”:

- \(H_1 = (-\infty,a_1] \times \dots \times (-\infty,a_m]\)  
- \(H_2 = [a_1,w_1] \times \dots \times [a_m,w_m]\)  
- \(H_3 = [w_1,r_1] \times \dots \times [w_m,r_m]\)

and the **clipping region**

\[
C_{\min} := H_1 \cup H_2 \cup H_3.
\]

### ARC-Hypervolume (minimization)

The **ARC-Hypervolume** for minimization is

\[
\mathrm{HV}^{\mathrm{ARC}}_{\min}(P; r,a,w)
  := \lambda^m\bigl(C_{\min} \cap D(P)\bigr).
\]

Since \(H_1,H_2,H_3\) only intersect on lower-dimensional boundaries, we have

\[
\mathrm{HV}^{\mathrm{ARC}}_{\min}(P; r,a,w)
  = \lambda^m(H_1 \cap D(P))
  + \lambda^m(H_2 \cap D(P))
  + \lambda^m(H_3 \cap D(P)).
\]

Each term can be computed as an ordinary hypervolume of a modified point set:

- **Region \(H_1\)**  
  Use points with \(p \le a\) and reference \(a\).

- **Region \(H_2\)**  
  Use points with \(p \le w\), but clip them up to at least \(a\),
  i.e.\ replace \(p_j\) by \(\max(p_j,a_j)\), and use reference \(w\).

- **Region \(H_3\)**  
  Use all points, but clip them up to at least \(w\),
  i.e.\ replace \(p_j\) by \(\max(p_j,w_j)\), and use reference \(r\).

In code, this is exactly what the `_hypervolume_*_min_arc` functions implement.

### Maximization case

For **maximization**, we consider the dominated region

\[
D_{\max}(P) = \{z : \exists p \in P,\ r_j \le z_j \le p_j\ \forall j\}
\]

and define a clipping region

\[
C_{\max}
  := [r_1,w_1] \times \dots \times [r_m,w_m]
   \,\cup\,
     [w_1,a_1] \times \dots \times [w_m,a_m]
   \,\cup\,
     [a_1,\infty)^m,
\]

with \(r_j \le w_j \le a_j\).

The maximization ARC-HV is

\[
\mathrm{HV}^{\mathrm{ARC}}_{\max}(P; r,a,w)
  := \lambda^m(C_{\max} \cap D_{\max}(P)).
\]

In the implementation, this is obtained via a sign flip:

```text
p_min = -p,  r_min = -r,  a_min = -a,  w_min = -w
```

and then calling the minimization routines.

---

## Key Properties (minimization)

Assume \(a_j \le w_j \le r_j\) for all components.

- If **reservation** \(w\) is dominated (i.e.\ ∃ \(p \in P\) with \(p \le w\)),
  then the entire block \([w,r]^m\) is included in the clipped dominated region
  \(C_{\min} \cap D(P)\).  

- If **aspiration** \(a\) is dominated (∃ \(p \in P\) with \(p \le a\)),
  then the region from \(a\) up to \(r\) is included (via contributions in
  \(H_2\) and \(H_3\)).  

- Points that lie strictly outside the preference window in some components
  only contribute where their dominated boxes intersect \(H_1,H_2,H_3\);
  everything else is clipped away.

---

## Examples (matching the LaTeX report)

### 2-D example (minimization)

```python
from truncated_hypervolume import hypervolume_2d, mc_estimate_hv_2d

P2 = [(1, 4), (3, 3), (1.5, 3.5)]
r2 = (5, 5)
a2 = (2, 2)
w2 = (4, 4)

# Standard hypervolume
hv_full = hypervolume_2d(P2, r2, maximize=False)           # 6.75

# ARC-Hypervolume
hv_arc  = hypervolume_2d(P2, r2, maximize=False, a=a2, w=w2)  # 2.5

# Monte Carlo check
hv_exact, hv_mc = mc_estimate_hv_2d(P2, r2, maximize=False, a=a2, w=w2,
                                    n_samples=200_000, seed=5)
print(hv_exact, hv_mc)
```

Analytical results:

- Standard hypervolume:
  \[
    \mathrm{HV}_{\min}(P; r) = 6.75.
  \]
- ARC-Hypervolume:
  \[
    \mathrm{HV}^{\mathrm{ARC}}_{\min}(P; r,a,w) = 2.5.
  \]

### 3-D example (minimization)

```python
from truncated_hypervolume import hypervolume_3d, mc_estimate_hv_3d

P3 = [(3, 3, 3)]
r3 = (5, 5, 5)
a3 = (2, 2, 2)
w3 = (4, 4, 4)

# Standard hypervolume
hv3_full = hypervolume_3d(P3, r3, maximize=False)              # 8.0

# ARC-Hypervolume
hv3_arc  = hypervolume_3d(P3, r3, maximize=False, a=a3, w=w3)  # 2.0

# Monte Carlo check
hv_exact, hv_mc = mc_estimate_hv_3d(P3, r3, maximize=False, a=a3, w=w3,
                                    n_samples=200_000, seed=3)
print(hv_exact, hv_mc)
```

Analytical results:

- Standard hypervolume:
  \[
    \mathrm{HV}_{\min}(\{(3,3,3)\}; r) = (5-3)^3 = 8.
  \]
- ARC-Hypervolume:
  \[
    \mathrm{HV}^{\mathrm{ARC}}_{\min}(\{(3,3,3)\}; r,a,w)
    = |[3,4]^3| + |[4,5]^3| = 1 + 1 = 2.
  \]

### 3-D example (maximization, no clipping)

The original test (no aspiration / reservation) is:

```python
from truncated_hypervolume import hypervolume_3d

P3_max = [
    (1, 6, 4),
    (3, 5, 1),
    (4, 4, 6),
    (5, 2, 3),
    (6, 1, 5),
    (1, 3, 7),
    (2, 2, 8),
]
ref3 = (0, 0, 0)

hv_std = hypervolume_3d(P3_max, ref3, maximize=True)  # 128.0
```

This matches the expected value \(128\) (verified analytically and by Monte Carlo).

---

## Monte Carlo Verification

For sanity checks, the module provides Monte Carlo estimators:

```python
hv_exact_2d, hv_mc_2d = mc_estimate_hv_2d(
    points, ref,
    maximize=False,
    a=a, w=w,
    n_samples=100_000,
    seed=0,
)

hv_exact_3d, hv_mc_3d = mc_estimate_hv_3d(
    points3, ref3,
    maximize=True,
    a=a3, w=w3,
    n_samples=200_000,
    seed=1,
)
```

- `hv_exact_*` is the deterministic value from the exact algorithm
  (standard HV or ARC-HV depending on `a,w`).  
- `hv_mc_*` is a Monte Carlo estimate of the volume, obtained by sampling
  uniformly in a bounding box and checking membership in the dominated
  region (and, for ARC-HV, also in the clipping region).

On the examples in this repository, the relative error is typically
around \(10^{-3}\) for \(10^5\)–\(2\times10^5\) samples.

---

## License

Feel free to reuse and modify this code for research and teaching
purposes. If you build on it in a publication, a short citation or
acknowledgment is appreciated but not required.
