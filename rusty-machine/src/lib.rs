extern crate num as libnum;
extern crate rand;

/// Module for linear algebra.
pub mod linalg {

    pub trait Metric<T> {
        fn norm(&self) -> T;
	}

    pub mod matrix;
    pub mod vector;
    pub mod utils;
    pub mod macros;
}

/// Module for machine learning.
pub mod learning {
    pub mod lin_reg;
    pub mod k_means;
    pub mod nnet;

    /// Trait for supervised model.
    pub trait SupModel<T,U> {

        /// Predict output from data.
        fn predict(&self, data: &T) -> U;

        /// Train the model using data and outputs.
        fn train(&mut self, data: &T, value: &U);
	}

    /// Trait for unsupervised model.
	pub trait UnSupModel<T, U> {

        /// Predict output from data.
        fn predict(&self, data: &T) -> U;

        /// Train the model using data.
        fn train(&mut self, data: &T);
	}

    /// Module for optimization in machine learning setting.
    pub mod optim {

        /// Trait for models which can be gradient-optimized.
        pub trait Optimizable {
            type Data;
            type Target;

            /// Compute the gradient for the model.
            fn compute_grad(&self, params: &[f64], data: &Self::Data, target: &Self::Target) -> Vec<f64>;
        }

        /// Trait for optimization algorithms.
        pub trait OptimAlgorithm<M : Optimizable> {

            /// Return the optimized parameter using gradient optimization.
            ///
            /// Takes in a set of starting parameters and related model data.
            fn optimize(&self, model: &M, start: &[f64], data: &M::Data, outputs: &M::Target) -> Vec<f64>;
        }

        pub mod grad_desc;
    }

    /// Module for learning tools.
    pub mod toolkit {
        pub mod link_fn;
    }
}
