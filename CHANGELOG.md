# Change Log

This document will be used to keep track of changes made between release versions. I'll do my best to note any breaking changes!

---

## 0.1.6 (Unreleased)

### Breaking Changes

- Removed the `new` constructor for the `LinRegressor`. This has been replaced by the `default` function from the `Default` trait.

### Features

- Added a `select` method for cloning a block from a matrix.
- Implemented QR decomposition, and eigenvalue decomposition.

### Minor Changes

- Optimizations to matrix multiplication