//! # The rusty-machine crate.
//!
//! A crate built for machine learning that works out-of-the-box.
//!
//! ---
//!
//! ## Structure
//!
//! The crate is made up of two primary modules: learning and linalg.
//!
//! ### learning
//!
//! The learning module contains all of the machine learning modules.
//! This means the algorithms, models and related tools.
//!
//! The currently supported techniques are:
//!
//! - Linear Regression
//! - Logistic Regression
//! - Generalized Linear Models
//! - K-Means Clustering
//! - Neural Networks
//! - Gaussian Process Regression
//! - Support Vector Machines
//! - Gaussian Mixture Models
//! - Naive Bayes Classifiers
//! - DBSCAN
//! - k-Nearest Neighbor Classifiers
//! - Principal Component Analysis
//!
//! ### linalg
//!
//! The linalg module reexports some structs and traits from the
//! [rulinalg](https://crates.io/crates/rulinalg) crate. This is to provide
//! easy access to common linear algebra tools within this library.
//!
//! ---
//!
//! ## Usage
//!
//! Specific usage of modules is described within the modules themselves. This section
//! will focus on the general workflow for this library.
//!
//! The models contained within the learning module should implement either
//! `SupModel` or `UnSupModel`. These both provide a `train` and a `predict`
//! function which provide an interface to the model.
//!
//! You should instantiate the model, with your chosen options and then train using
//! the training data. Followed by predicting with your test data. *For now*
//! cross-validation, data handling, and many other things are left explicitly
//! to the user.
//!
//! Here is an example usage for Gaussian Process Regression:
//!
//! ```
//! use rusty_machine::linalg::Matrix;
//! use rusty_machine::linalg::Vector;
//! use rusty_machine::learning::gp::GaussianProcess;
//! use rusty_machine::learning::gp::ConstMean;
//! use rusty_machine::learning::toolkit::kernel;
//! use rusty_machine::learning::SupModel;
//!
//! // First we'll get some data.
//!
//! // Some example training data.
//! let inputs = Matrix::new(3,3,vec![1.,1.,1.,2.,2.,2.,3.,3.,3.]);
//! let targets = Vector::new(vec![0.,1.,0.]);
//!
//! // Some example test data.
//! let test_inputs = Matrix::new(2,3, vec![1.5,1.5,1.5,2.5,2.5,2.5]);
//!
//! // Now we'll set up our model.
//! // This is close to the most complicated a model in rusty-machine gets!
//!
//! // A squared exponential kernel with lengthscale 2, and amplitude 1.
//! let ker = kernel::SquaredExp::new(2., 1.);
//!
//! // The zero function
//! let zero_mean = ConstMean::default();
//!
//! // Construct a GP with the specified kernel, mean, and a noise of 0.5.
//! let mut gp = GaussianProcess::new(ker, zero_mean, 0.5);
//!
//!
//! // Now we can train and predict from the model.
//!
//! // Train the model!
//! gp.train(&inputs, &targets).unwrap();
//!
//! // Predict the output from test data.
//! let outputs = gp.predict(&test_inputs).unwrap();
//! ```
//!
//! This code could have been a lot simpler if we had simply adopted
//! `let mut gp = GaussianProcess::default();`. Conversely, you could also implement
//! your own kernels and mean functions by using the appropriate traits.
//!
//! Additionally you'll notice there's quite a few `use` statements at the top of this code.
//! We can remove some of these by utilizing the `prelude`:
//!
//! ```
//! use rusty_machine::prelude::*;
//!
//! let _ = Matrix::new(2,2,vec![2.0;4]);
//! ```

#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

#[macro_use]
extern crate rulinalg;
extern crate num as libnum;
extern crate rand;
extern crate rand_distr;

pub mod prelude;

/// The linear algebra module
///
/// This module contains reexports of common tools from the rulinalg crate.
pub mod linalg {
    pub use rulinalg::matrix::{Axes, Matrix, MatrixSlice, MatrixSliceMut, BaseMatrix, BaseMatrixMut};
    pub use rulinalg::vector::Vector;
    pub use rulinalg::norm;
    pub use rulinalg::matrix::decomposition::*;
}

/// Module for data handling
pub mod data {
    pub mod transforms;
}

/// Module for machine learning.
pub mod learning {
    pub mod dbscan;
    pub mod glm;
    pub mod gmm;
    pub mod lin_reg;
    pub mod logistic_reg;
    pub mod k_means;
    pub mod nnet;
    pub mod gp;
    pub mod svm;
    pub mod naive_bayes;
    pub mod knn;
    pub mod pca;

    pub mod error;

    /// A new type which provides clean access to the learning errors
    pub type LearningResult<T> = Result<T, error::Error>;

    /// Trait for supervised model.
    pub trait SupModel<T, U> {
        /// Predict output from inputs.
        fn predict(&self, inputs: &T) -> LearningResult<U>;

        /// Train the model using inputs and targets.
        fn train(&mut self, inputs: &T, targets: &U) -> LearningResult<()>;
    }

    /// Trait for unsupervised model.
    pub trait UnSupModel<T, U> {
        /// Predict output from inputs.
        fn predict(&self, inputs: &T) -> LearningResult<U>;

        /// Train the model using inputs.
        fn train(&mut self, inputs: &T) -> LearningResult<()>;
    }

    /// Module for optimization in machine learning setting.
    pub mod optim {

        /// Trait for models which can be gradient-optimized.
        pub trait Optimizable {
            /// The input data type to the model.
            type Inputs;
            /// The target data type to the model.
            type Targets;

            /// Compute the gradient for the model.
            fn compute_grad(&self,
                            params: &[f64],
                            inputs: &Self::Inputs,
                            targets: &Self::Targets)
                            -> (f64, Vec<f64>);
        }

        /// Trait for optimization algorithms.
        pub trait OptimAlgorithm<M: Optimizable> {
            /// Return the optimized parameter using gradient optimization.
            ///
            /// Takes in a set of starting parameters and related model data.
            fn optimize(&self,
                        model: &M,
                        start: &[f64],
                        inputs: &M::Inputs,
                        targets: &M::Targets)
                        -> Vec<f64>;
        }

        pub mod grad_desc;
        pub mod fmincg;
    }

    /// Module for learning tools.
    pub mod toolkit {
        pub mod activ_fn;
        pub mod cost_fn;
        pub mod kernel;
        pub mod rand_utils;
        pub mod regularization;
    }
}

#[cfg(feature = "stats")]
/// Module for computational statistics
pub mod stats {

    /// Module for statistical distributions.
    pub mod dist;
}

/// Module for evaluating models.
pub mod analysis {
    pub mod confusion_matrix;
    pub mod cross_validation;
    pub mod score;
}

#[cfg(feature = "datasets")]
/// Module for datasets.
pub mod datasets;
