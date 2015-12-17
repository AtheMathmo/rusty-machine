# Development

This document will keep track of my development goals for this project.

---

## Current Progress

I have completed a first pass at the linear algebra library. It is now functional enough that I can start adding some ML! I've implemented a basic linear regression module.

I will need to update and optimize the linear algebra library but should be able to do this in tandem.

---

## Timeline

This marks my intended release goals. I won't estimate the actual dates of release but rather the content I want to include in each version. I am actively developing and so expect to move through these at a good pace!

| Version | Features | Dependencies |
|: --- :| --- | --- |
| 0.0.8 | <ul><li>K-means Clustering</li></ul> | <ul><li>None</li></ul> |
| 0.0.9 | <ul><li>Matrix macros</li></ul> | <ul><li>None</li></ul> |
| 0.1.0 | <ul><li>Feed forward neural net</li></ul> | <ul><li>Matrix Macros</li><li>Gradient descent</li></ul> |
| 0.1.1 | <ul><li>Cholesky Decomp</li></ul> | <ul><li>None</li></ul> |
| 0.1.2 | <ul><li>Gaussian Processes</li></ul> | <ul><li>Cholesky Decomp</li></ul> |
| 0.2.0 | <ul><li>Generalized linear regression</li><li>SVM</li><li>Linalg optimization</li></ul> | <ul><li>Lots</li></ul> |

I have chosen to push out a number of different algorithms before focused optimizing. This is partly so I can have use-cases for profiling but mostly for fun!  I will also be working through optimization throughout this process.

The term "Matrix Macros" is a little ambiguous. By this I mean matlab style matrix construction, i.e. [Vector; Matrix] would create a new matrix with the leftmost column being the vector. This may require a compiler plugin...

### Unplanned:

- Multi-threaded divide and conquer matrix multiplication (currently iterative).
- Tidy up indexing.
- Start work on statistics components - R.V. sampling etc.