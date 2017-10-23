//! The Transforms module
//!
//! This module contains traits used to transform data using common
//! techniques. It also reexports these `Transformer`s from child modules.
//!
//! The `Transformer` trait provides a shared interface for all of the
//! data preprocessing transformations in rusty-machine. Some of these `Transformations`
//! can be inverted via the `Invertible` trait.
//!
//! Note that some `Transformer`s can not be created without first using the
//! `TransformFitter` trait.
//!
//! # Examples
//!
//! ```
//! use rusty_machine::data::transforms::{Transformer, TransformFitter, MinMaxFitter};
//! use rusty_machine::data::transforms::minmax::MinMaxScaler;
//! use rusty_machine::linalg::Matrix;
//!
//! // Some data that we want to scale between 0 and 1
//! let data = Matrix::new(3, 2, vec![-1.5, 1.0, 2.0, 3.0, -1.0, 2.5]);
//! // Create a new `MinMaxScaler` using the `MinMaxFitter`
//! let mut scaler: MinMaxScaler<f64> = MinMaxFitter::new(0.0, 1.0).fit(&data).expect("Failed to fit transformer");
//! // Transform the data using the scaler
//! let transformed = scaler.transform(data).expect("Failed to transformer data");
//! ```

pub mod lda;
pub mod minmax;
pub mod normalize;
pub mod standardize;
pub mod shuffle;

use learning::LearningResult;

pub use self::lda::LDAFitter;
pub use self::minmax::MinMaxFitter;
pub use self::normalize::Normalizer;
pub use self::shuffle::Shuffler;
pub use self::standardize::StandardizerFitter;

/// A trait used to construct Transformers which must first be fitted
pub trait TransformFitter<I, O, T: Transformer<I, O>> {
    /// Fit the inputs to create the `Transformer`
    fn fit(self, inputs: &I) -> LearningResult<T>;
}

/// Trait for data transformers
pub trait Transformer<I, O> {
    /// Transforms the inputs
    fn transform(&mut self, inputs: I) -> LearningResult<O>;
}

/// Trait for invertible data transformers
pub trait Invertible<I, O> : Transformer<I, O> {
    /// Maps the inputs using the inverse of the fitted transform.
    fn inv_transform(&self, inputs: O) -> LearningResult<I>;
}
