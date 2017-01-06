# Change Log

This document will be used to keep track of changes made between release versions. I'll do my best to note any breaking changes!

## 0.5.4

### New Contributors

- [sinhrks](https://github.com/sinhrks)

### Breaking Changes

- None

### Features

- Add a new `datasets` module behind a `datasets` feature flag.
- Add new classification scores: `precision`, `recall`, and `f1`.
- Add a new `Transformer::fit` function to allow prefitting of a
`Transformer` before use.

### Bug Fixes

- None

### Minor Changes

- `LinRegressor` now uses `solve` instead of `inverse` for improved
accuracy and stability.

## 0.5.3

### Breaking Changes

- None

### Features

- Adding a new `confusion_matrix` module.

### Bug Fixes

- None

### Minor Changes

- Updated rulinalg dependency to `0.3.7`.

## 0.5.2

### New Contributors

- [scholtzan](https://github.com/scholtzan)

### Breaking Changes

- None

### Features

- None

### Bug Fixes

- Regularization constant for GMM is now only added to diagonal.

### Minor Changes

- Added some better `Result` handling to GMM.

## 0.5.1

This version includes no changes but is a bump due to a
[crates bug](https://github.com/rust-lang/crates.io/issues/448#issuecomment-251037439).

See the notes for 0.5.0 below.

## 0.5.0

This is another fairly large release. Thank you to everyone who contributed!

### New Contributors

- [NivenT](https://github.com/NivenT)
- [theotherphil](https://github.com/theotherphil)
- [andrewcsmith](https://github.com/andrewcsmith)

### Breaking Changes

- The `SupModel` and `UnSupModel` traits now return
`Result`s for the `train` and `predict` functions.
- Updated to [rulinalg](https://github.com/AtheMathmo/rulinalg) v0.3 ([see
rulinalg changelog for
details](https://github.com/AtheMathmo/rulinalg/blob/master/CHANGELOG.md#030)).

### Features

- Adding RMSProp gradient descent algorithm. #121
- Adding cross validation. #125
- Adding a new `Shuffler` transformer. #135

### Bug Fixes

- None

### Minor Changes

- Adding benchmarks
- Initiate GMM with sample covariance of data (instead of identity matrix).

## 0.4.4

### Breaking Changes

- None

### Features

- Adding new `Transformer` trait for data preprocessing.
- Adding a `MinMax` transformer.
- Adding a `Standardizer` transformer.

### Minor Changes

- None

## 0.4.3

### New Contributors

- [tafia](https://github.com/tafia) who is responsible for all changes in this release.

### Breaking Changes

- None

### Features

- None

### Minor Changes

- Made neural nets more efficient by reducing clones
and some restructuring.
- Removing unneeded copying in favour of slicing for performance.
- Using `iter_rows` in favour of manually row iterating by chunks.

## 0.4.2

### Breaking Changes

- None

### Features

- None

### Minor Changes

- Fixed a significant bug in the K-Means algorithm. Centroids
were not updating correctly during M-step.

## 0.4.1

### Breaking Changes

- None

### Features

- Added experimental implementation of DBSCAN clustering.

### Minor Changes

- Added new example for K-Means clustering in repo.

## 0.4.0

This is the biggest release so far. Primarily because the `linalg`
module has been pulled out into its own crate: [rulinalg](https://github.com/AtheMathmo/rulinalg).

In addition to this there have been a number of improvements to the `linalg`
and `learning` moduled in this release.

### Breaking Changes

- The `linalg` module pulled out and replaced by reexports of [rulinalg](https://github.com/AtheMathmo/rulinalg).
All structs are now imported at the `linalg` level, i.e. `linalg::matrix::Matrix` -> `linalg::Matrix`.
- Decomposition methods now return `Result` instead of panicking on fail.
- K-Means now has a trait for `Initializer` - which allows generic initialization algorithms.

### Features

- New error handling in both the `linalg` (now rulinalg) and `learning` modules.
- Bug fixed in eigendecomposition: it can now be used!
- K-means can now take a generic initialization algorithm.

### Minor Changes

- Optimization and code cleanup in the decomposition methods.
- Some optimization in the K-Means model.

## 0.3.3

### New Contributors

- [ic](https://github.com/ic) (Added examples to repo!)

### Breaking Changes

- Parameter methods now return `Option<&Type>` instead of `&Option<Type>`.

### Features

- `MatrixSlice` and `MatrixSliceMut` now have `IntoIterator` methods.

### Minor Changes

- Adding examples to the repository.

## 0.3.2

### New Contributors

- [DarkDrek](https://github.com/DarkDrek) (Who is responsible for almost all changes in this release. Thank you!)

### Breaking Changes

- `Matrix`: `mean` and `variance` methods now take `Axes` enum instead of `usize` flag for dimension.

### Features

- Assignment operators (`+=`, `-=`, etc.) now implemented for `Vector`.

### Minor Changes

- Some optimizations to `variance` computation for `Matrix`.
- Some code cleanup - thanks to [clippy](https://github.com/Manishearth/rust-clippy). 

## 0.3.1

### Breaking Changes

- None

### Features

- New helper methods to access GMM distribution parameters.
- New GMM constructor to choose different prior mixture weights.

### Minor Changes

- Fixed a bug where GMM covariances were incorrectly computed when using diagonal constraint.

## 0.3.0

### New Contributors

- [rrichardson](https://github.com/rrichardson)

### Breaking Changes

- All fields on `GradDesc` and `StochasticGD` are now private.
- Matrix slices now have the same lifetime as their target data.

### Features

- Adding new slice utility methods : `from_raw_parts` for `MatrixSlice`s and `as_slice` methods for `Matrix`.
- Adding framework for regularization. Implementing regularization for nnets.
- Adding early stopping to gradient descent algorithms.
- Adding `AdaGrad` gradient descent algorithm.
- Implementing `Into` and `From` for `Matrix`, `Vector`, and `MatrixSlice`s.

### Minor Changes

- Bug fixing naive bayes : no longer attempts to update empty class.
- Removing unneeded trait bounds on `Matrix`/`Vector` implementations.

## 0.2.8

### Breaking Changes

- The `new` constructors for `Matrix` and `Vector` now take an `Into<Vec>` generic type. May break some type inference.

### Features

- Added row iterators for each matrix struct.
- Implemented OpAssign overloading for `Matrix` and `MatrixSliceMut`.

### Minor Changes

- Moved unit tests into respective modules.
- Modified slice iterators to make the `offset` usage safe(er).
- Removed some compiler warnings from the tests.

## 0.2.7

### Breaking Changes

- None

### Features

- `Matrix` and `Vector` now implement [PartialEq](https://doc.rust-lang.org/core/cmp/trait.PartialEq.html).

### Minor Changes

- Fixed a bug where eigendecomposition for 2x2 matrices was incorrect.

## 0.2.6

### Breaking Changes

- None

### Features

- None

### Minor Changes

- Fixing a bug with matrix slice multiplication.
- Removing unneeded NumCast import.

## 0.2.5

### Breaking Changes

- None

### Features

- Adding Naive Bayes classifiers.
- Adding a prelude for common imports.
- Adding MatrixSlice and MatrixSliceMut for efficient matrix views.

### Minor Changes

- Using [matrixmultiply](https://github.com/bluss/matrixmultiply) to get huge performance gains! Thanks [bluss](https://github.com/bluss/).
- Code refactor to split up the matrix module.


## 0.2.4

### New Contributors

- [vishalsodani](https://github.com/vishalsodani) (fixing some typos)
- [danlrobertson](https://github.com/danlrobertson) (added the `KMeansClassifierBuilder`)

### Breaking Changes

- None

### Features

- `KMeansClassifier` now has a builder!

### Minor Changes

- We're now using travis for CI.
- Deriving Debug, Clone, Copy for Gaussian and Exponential distributions.


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
