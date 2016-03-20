# Change Log

This document will be used to keep track of changes made between release versions. I'll do my best to note any breaking changes!


## 0.2.3

### Breaking Changes

- `mut_data` method now returns a mutable slice `&mut [T]` instead of a `Vec<T>`.

### Features

- More vectorization and optimization of linear algebra.

### Minor Changes

- Copy and Clone now implemented where applicable.
- Added test coverage.

## 0.2.2

### New Contributors

- [zackmdavis](https://github.com/zackmdavis) (contributed all features for this version, thank you!)

### Breaking Changes

- None

### Features

- Can now debug print matrices and vectors.
- Can now pretty print matrices to given precision.

### Minor Changes

- Fixed the dependency versions used in Cargo.toml.
- Updated the library documentation with complete list of ML tools.

## 0.2.1

### Breaking Changes

- None

### Features

- Addition of Gaussian Mixture Models.
- Allow basic arithmetic to combine kernels.

### Minor Changes

- Added some missing documentation.
- Some code formatting.
- Minor improvements thanks to clippy.

## 0.2.0

### Breaking Changes

- Neural network instantiation `new` method now requires a training algorithm to be specified.

### Features

- Adding more kernels (for full list see API documentation).
- Generalized Linear Model.
- Updated model structures to allow more freedom in training algorithms.

### Minor Changes

- Some more documentation.
- Some minor code formatting.

---
## 0.1.8

### Breaking Changes

- None

### Features

- Add Support Vector Machines.

### Minor Changes

- Minor code cleanup.
- Some micro optimization.

---

## 0.1.7

### Breaking Changes

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
