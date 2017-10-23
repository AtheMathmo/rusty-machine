# rusty-machine

[![Join the chat at https://gitter.im/AtheMathmo/rusty-machine](https://badges.gitter.im/AtheMathmo/rusty-machine.svg)](https://gitter.im/AtheMathmo/rusty-machine?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/AtheMathmo/rusty-machine.svg?branch=master)](https://travis-ci.org/AtheMathmo/rusty-machine)

The crate is currently on version [0.5.4](https://crates.io/crates/rusty-machine/).

Read the [API Documentation](https://AtheMathmo.github.io/rusty-machine/) to learn more.

And here is a document detailing development efforts. Including a projected timeline for immediate features.
Please feel free to give feedback and let me know if there any features you believe should take precedence.

- [Development](DEVELOPMENT.md)

---

## Summary

Rusty-machine is a general purpose machine learning library implemented entirely in Rust.
It aims to combine speed and ease of use - without requiring a huge number of external dependencies.

This project began as a way for me to learn Rust and brush up on some less familiar machine learning algorithms and techniques.
Now the project aims to provide a complete, easy to use, machine learning library for Rust.

This library is still very much in early stages of development. Although there are a good number of algorithms many other
things are missing. Rusty-machine is probably not the best choice for any serious projects - but hopefully that can change in the near future!

#### Contributing

This project is currently looking for contributors of all capacities!

I have now created a dedicated page for [contributing](CONTRIBUTING.md). If you're interested please take a look.

---

## Implementation

This project is implemented using [Rust](https://www.rust-lang.org/). Currently there are no other dependencies!
Though, we are planning on introducing optional BLAS/LAPACK dependencies soon.

---

## Current Progress

Rusty-machine uses [rulinalg](https://github.com/AtheMathmo/rulinalg) for its linear algebra back end.
This is fairly complete but there is still lots of room for optimization and we should provide BLAS/LAPACK support.

### Machine Learning

- Linear Regression
- Logistic Regression
- Generalized Linear Models
- K-Means Clustering
- Neural Networks
- Gaussian Process Regression
- Support Vector Machines
- Gaussian Mixture Models
- Naive Bayes Classifiers
- DBSCAN
- k-Nearest Neighbor Classifiers

There is also a basic `stats` module behind a feature flag.

---

## Usage

The library usage is described well in the [API documentation](https://AtheMathmo.github.io/rusty-machine/) - including example code.
I will provide a brief overview of the library in it's current state and intended usage.

### Installation

The library is most easily used with [cargo](http://doc.crates.io/guide.html). Simply include the following in your Cargo.toml file:

```toml
[dependencies]
rusty-machine="0.5.4"
```

And then import the library using:

```rust
extern crate rusty_machine as rm;
```

The library consists of two core components. The linear algebra module and the learning module.

#### Linalg

The linear algebra module contains reexports from the [rulinalg](https://github.com/AtheMathmo/rulinalg) crate. This is to
provide easy access to components which are used frequently within rusty-machine.

More detailed coverage can be found in the [API documentation](https://AtheMathmo.github.io/rusty-machine/).

#### Learning

The learning module contains machine learning models. The machine learning implementations are designed with
simpicity and customization in mind. This means you can control the optimization algorithms but still retain
the ease of using default values. This is an area I am actively trying to improve on!

The models all provide `predict` and `train` methods enforced by the `SupModel` and `UnSupModel` traits.

There are some examples within this repository that can help you familiarize yourself with the library.
