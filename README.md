# rusty-machine

[![Join the chat at https://gitter.im/AtheMathmo/rusty-machine](https://badges.gitter.im/AtheMathmo/rusty-machine.svg)](https://gitter.im/AtheMathmo/rusty-machine?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Here is the API documentation for the rust crate. Currently up to date for [version 0.2.3](https://crates.io/crates/rusty-machine/0.2.3).

- [API Documentation](https://AtheMathmo.github.io/rusty-machine/)

And here is a document detailing development efforts. Including a projected timeline for immediate features. Please feel free to give feedback and let me know if there any features you believe should take precedence.

- [Development](DEVELOPMENT.md)

---

## Summary

This library is very much in early stages of development. There is a lot of work needing to be done and optimization is needed before this can be used for any serious applications. Despite this, feel free to take a look at the source code and try out the [crate](https://crates.io/crates/rusty-machine).

This project was originally intended to be an implementation of Online Collaborative Filtering in Rust. However, after putting a lot of effort into building the linear algebra library that was needed I decided I shouldn't waste the effort and should make something more general!

So this project is now going to be a machine learning crate for Rust. I began this project as a fun way to learn Rust and so there will be lots of things that need improving (I'm still not very familiar with LLVM). I hope that this crate will provide a number of standard out-the-box machine learning algorithms.

#### Contributing

This project is currently looking for contributors of all capacities!

I have now created a dedicated page for [contributing](CONTRIBUTING.md). If you're interested please take a look.

---

## Implementation

This project is implemented using [Rust](https://www.rust-lang.org/).

## Motivation

This key motivation behind this project was for me to learn some systems programming and a new language! I wanted to try implementing a linear algebra library within a systems programming language and then extend this to explore some machine learning algorithms.

---

## Current Progress

The linear algebra library is now fairly filled out. But there is still lots of room for optimization (it is almost definitely better to switch to BLAS/LAPACK).

There is also a `stats` module behind an optional features flag.

### Matrices

- Generic data matrices
- Concatenation
- Data manipulation (row and column selection/repetition etc.)
- Arithmetic

### Machine Learning

- Linear Regression
- Logistic Regression
- Generalized Linear Models
- K-Means Clustering
- Neural Networks
- Gaussian Process Regression
- Support Vector Machines
- Gaussian Mixture Models

---

## Usage

The library usage is described well in the [API documentation](https://AtheMathmo.github.io/rusty-machine/) - including example code. I will provide a brief overview of the library in it's current state and intended usage.

### Installation

The library is most easily used with [cargo](http://doc.crates.io/guide.html). Simply include the following in your Cargo.toml file:

```
[dependencies.rusty-machine]
version="*"
```

And then import the library using:

```
extern crate rusty_machine as rm;
```

The library consists of two core components. The linear algebra module and the learning module.

#### Linalg

The linear algebra module contains the Matrix and Vector data structures and related methods - such as matrix decomposition. Usage looks like this:

```
extern crate rusty_machine as rm;

use rm::linalg::matrix::Matrix;

let a = Matrix::new(2,2, vec![1.0, 2.0, 3.0, 4.0]); // Create a 2x2 matrix [[1,2],[3,4]]

let b = Matrix::new(2,3, vec![1.0,2.0,3.0,4.0,5.0,6.0]); // Create a 2x3 matrix [[1.0,2.0,3.0],[4.0,5.0,6.0]]

let c = &a * &b; // Matrix product of a and b
```

More detailed coverage can be found in the [API documentation](https://AtheMathmo.github.io/rusty-machine/).

#### Learning

The learning module contains machine learning models. The machine learning implementations are designed with customizability in mind. This means you can control the optimization algorithms but still retain the ease of using default values. This is an area I am actively trying to improve on!

The current algorithms included are:

- Linear Regression
- Logistic Regression
- Generalized Linear Models
- K-Means Clustering
- Neural Networks
- Gaussian Process Regression
- Support Vector Machines
- Gaussian Mixture Models

---