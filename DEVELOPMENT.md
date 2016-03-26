# Development

This document will keep track of my development goals for this project.

---

## Current Progress

The linear algebra library is now fairly filled out. But there is still lots of room for optimization (it is almost definitely better to switch to BLAS/LAPACK).

### Linear Algebra

- Generic data matrices
- Concatenation
- Data manipulation (row and column selection/repetition etc.)
- Arithmetic
- Inverses and decompositions

### Machine Learning

- Linear Regression
- K-Means Clustering
- Neural Networks
- Gaussian Processes
- Logistic Regression
- Generalized Linear Model
- Support Vector Machines
- Gaussian Mixture Models

---

## Timeline

This marks my intended release goals. I won't estimate the actual dates of release but rather the content I want to include in each version. I am actively developing and so expect to move through these at a good pace!

<table>
    <tr>
        <th>Version</th><th>Features</th><th>Dependencies</th>
    </tr>
    <tr>
        <td>0.2.4</td><td>Matrix Slices</td><td><ul><li>None</li></ul></td>
    </tr>
    <tr>
        <td>0.2.5</td><td>Parallel matrix multiplication</td><td><ul><li>None</li></ul></td>
    </tr>
    <tr>
        <td>0.2.X</td><td><ul><li>More advanced GD algorithms.</li><li>Model Regularization.</li><li>Improvements to GLM.</li></ul></td><td><ul><li>Lots</li></ul></td>
    </tr>
</table>

As part of 0.3 I plan to review the API to try and make it conform a little more to idiomatic Rust. This is mostly an issue right now within the linear algebra. We have some methods like `select` for matrices that are copying chunks of data - this should be made clearer with the name - perhaps `copy_chunk` or `copy_rows` for `select_rows`.

After this we will aim to separate the linear algebra from rusty-machine and introduce experimental BLAS bindings.

Whilst working on the above there will of course be ongoing development on the existing and new machine learning algorithms.

### Unplanned:

- Multi-threaded divide and conquer matrix multiplication (currently iterative).
- Tidy up indexing.
- Data Handling.
- Convolutional and Recurrent neural nets.
- SVM coordinate descent as per [this paper](http://www.loshchilov.com/publications/GECCO2011_AdaptiveCoordinateDescent.pdf).
