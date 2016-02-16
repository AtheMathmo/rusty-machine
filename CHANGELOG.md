# Change Log

This document will be used to keep track of changes made between release versions. I'll do my best to note any breaking changes!

---
## 0.1.8

### Breaking changes

- None

### Features

- Add Support Vector Machines.

### Minor Changes

- Minor code cleanup.
- Some micro optimization.

---

## 0.1.7

### Breaking changes

- None

### Features

- Added the stats module behind the optional feature flag `stats`.
- `stats` currently includes support for the Exponential and Gaussian distributions.

### Minor Changes

- Some rustfmt code cleanup.

---

## 0.1.6

### Breaking Changes

- Removed the `new` constructor for the `LinRegressor`. This has been replaced by the `default` function from the `Default` trait.

### Features

- Added a `select` method for cloning a block from a matrix.
- Implemented QR decomposition, and eigenvalue decomposition.
- Implemented eigendecomp (though only works definitely for real-symmetric matrices).

### Minor Changes

- Optimizations to matrix multiplication
