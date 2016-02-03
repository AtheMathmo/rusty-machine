# Change Log

This document will be used to keep track of changes made between release versions. I'll do my best to note any breaking changes!

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