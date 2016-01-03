# Development

This document will keep track of my development goals for this project.

---

## Current Progress

There is now a first pass at the linear algebra library and some basic machine learning algorithms in place.

### Matrices

- Generic data matrices
- Concatenation
- Data manipulation (row and column selection/repetition etc.)
- Arithmetic

### Machine Learning

- Linear Regression
- K-Means Clustering
- Neural Networks

I will need to update and optimize the linear algebra library but should be able to do this in tandem.

---

## Timeline

This marks my intended release goals. I won't estimate the actual dates of release but rather the content I want to include in each version. I am actively developing and so expect to move through these at a good pace!

<table>
    <tr>
        <th>Version</th><th>Features</th><th>Dependencies</th>
    </tr>
    <tr>
        <td>0.0.9</td><td><ul><li>Gradient descent.</li><li>Neural Nets</li></ul></td><td><ul><li>None</li></ul></td>
    </tr>
    <tr>
        <td>0.1.0</td><td><ul><li>Optimization (with basic data handling - unreleased).</li></ul></td><td></td>
    </tr>
    <tr>
        <td>0.1.1.</td><td><ul><li>Cholesky Decomp</li></ul></td><td><ul><li>None</li></ul></td>
    </tr>
    <tr>
        <td>0.1.2</td><td><ul><li>Gaussian Processes</li></ul></td><td><ul><li>Cholesky Decomp</li><li>Kernels</li></ul></td>
    </tr>
    <tr>
        <td>0.2.0</td><td><ul><li>Generalized linear regression</li><li>SVM</li><li>Linalg optimization</li></ul></td><td><ul><li>Lots</li></ul></td>
    </tr>
</table>

I have chosen to push out a number of different algorithms before focused optimizing. This is partly so I can have use-cases for profiling but mostly for fun!  I will also be working through optimization throughout this process.

### Unplanned:

- Multi-threaded divide and conquer matrix multiplication (currently iterative).
- Tidy up indexing.
- Start work on statistics components - R.V. sampling etc.
- Data Handling.
- Convolutional and Recurrent neural nets.
- Regularization in existing models.
