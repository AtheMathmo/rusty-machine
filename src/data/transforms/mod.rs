//! The Transforms module
//!
//! This module contains the `Transformer` trait and reexports
//! the transformers from child modules.
//!
//! The `Transformer` trait provides a shared interface for all of the
//! data preprocessing transformations in rusty-machine.
//!
//! The transformers provide preprocessing transformations which are 
//! commonly used in machine learning.

pub mod minmax;
pub mod standardize;

use learning::error;

pub use self::minmax::MinMaxScaler;
pub use self::standardize::Standardizer;

/// Trait for data transformers
pub trait Transformer<T> {
    /// Transforms the inputs and stores the transformation in the Transformer
    fn transform(&mut self, inputs: T) -> Result<T, error::Error>;

    /// Maps the inputs using the inverse of the fitted transform.
    fn inv_transform(&self, inputs: T) -> Result<T, error::Error>;
}