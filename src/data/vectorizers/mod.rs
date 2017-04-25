//! The Vectorizers module
//!
//! This module contains traits used to vectorize data using common
//! techniques. It also reexports these `Vectorizer`s from child modules.
//!
//! The `Vectorizer` trait provides a shared interface for all of the
//! data vectorizations in rusty-machine.

pub mod text;

use learning::LearningResult;

/// A trait used to construct Vectorizers
pub trait Vectorizer<U, V> {
    /// Fit the inputs to create the `Vectorizer`
    fn fit(&mut self, inputs: &[U]) -> LearningResult<()>;
    /// Vectorize the inputs
    fn vectorize(&mut self, inputs: &[U]) -> LearningResult<V>;
}
