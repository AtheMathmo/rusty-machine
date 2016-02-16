//! # The rusty-machine crate.
//!
//! Crate built for machine learning with some linear algebra thrown in.
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
//! - Gaussian Process Regression
//! - K-means classification
//! - Linear Regression
//! - Logistic Regression
//! - Neural Networks (simple feed forward)
//! - Support Vector Machines
//!
//! ### linalg
//!
//! The linalg module contains all of the linear algebra tools and structures.
//! This module is efficient but not state of the art. Development of this module
//! is not a key focus as I'm waiting for a clear community winner.
//!
//! ---
//!
//! # Usage
//!
//! Specific usage of modules is described within the modules themselves. This section
//! will focus on the general workflow for this library.
//!
//! The models contained within the learning module should implement either SupModel or UnSupModel.
//! These both provide a `train` and a `predict` function which provide an interface to the model.
//!
//! You should instantiate the model, with your chosen options and then train using the training data.
//! Followed by predicting with your test data. *For now* cross-validation, data handling, and many
//! other things are left explicitly to the user.
//!
//! Here is an example usage for Gaussian Process Regression:
//!
//! ```
//! use rusty_machine::linalg::matrix::Matrix;
//! use rusty_machine::linalg::vector::Vector;
//! use rusty_machine::learning::gp::GaussianProcess;
//! use rusty_machine::learning::gp::ConstMean;
//! use rusty_machine::learning::toolkit::kernel;
//! use rusty_machine::learning::SupModel;
//!
//! // Some example training data.
//! let inputs = Matrix::new(3,3,vec![1.,1.,1.,2.,2.,2.,3.,3.,3.]);
//! let targets = Vector::new(vec![0.,1.,0.]);
//!
//! // Some example test data.
//! let test_inputs = Matrix::new(2,3, vec![1.5,1.5,1.5,2.5,2.5,2.5]);
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
//! // Train the model!
//! gp.train(&inputs, &targets);
//!
//! // Predict output from test datae]
//! let outputs = gp.predict(&test_inputs);
//! ```
//!
//! Of course this code could have been a lot simpler if we had simply adopted
//! `let mut gp = GaussianProcess::default();`. Conversely, you could also implement
//! your own kernels and mean functions by using the appropriate traits.


extern crate num as libnum;
extern crate rand;

/// Module for linear algebra.
pub mod linalg {

    /// Trait for linear algebra metrics.
    ///
    /// Currently only implements basic euclidean norm.
    pub trait Metric<T> {

        /// Computes the euclidean norm.
        fn norm(&self) -> T;
    }

    pub mod matrix;
    pub mod vector;
    pub mod utils;
    pub mod macros;
}

/// Module for machine learning.
pub mod learning {
    pub mod gen_lin_mod;
    pub mod lin_reg;
    pub mod logistic_reg;
    pub mod k_means;
    pub mod nnet;
    pub mod gp;
    pub mod svm;

    /// Trait for supervised model.
    pub trait SupModel<T,U> {

        /// Predict output from inputs.
        fn predict(&self, inputs: &T) -> U;

        /// Train the model using inputs and targets.
        fn train(&mut self, inputs: &T, targets: &U);
    }

    /// Trait for unsupervised model.
    pub trait UnSupModel<T, U> {

        /// Predict output from inputs.
        fn predict(&self, inputs: &T) -> U;

        /// Train the model using inputs.
        fn train(&mut self, inputs: &T);
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
        pub trait OptimAlgorithm<M : Optimizable> {

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
        pub mod kernel;
        pub mod cost_fn;
    }
}

#[cfg(feature = "stats")]
/// Module for computational statistics
pub mod stats {

    /// Module for statistical distributions.
    pub mod dist;
}
