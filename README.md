# Truncated Hypervolume (2D & 3D)

This repository provides a small, self-contained Python module for computing
2-D and 3-D hypervolume indicators, with:

- **Minimization** and **maximization** support  
- Optional **truncation** using **aspiration** (`a`) and **reservation** (`w`) levels  
- Simple **Monte Carlo estimators** to numerically verify the exact results on toy examples  

The implementation is intended for research / teaching / prototyping, not for
large-scale production workloads.

---

## Files

- `truncated_hypervolume.py`  
  Core implementation of 2-D and 3-D hypervolume, truncation, and Monte Carlo
  verification, plus a few examples in the `__main__` block.

---

## Installation

Just drop the file into your project or clone this repository and run it with
a recent Python 3:

```bash
python truncated_hypervolume.py
