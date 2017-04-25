//! The Tokenizer module
//!
//! This module contains traits used to tokenize text.
//! It also reexports these `Tokenizer`s from child modules.
//!
//! The `Tokenizer`s are intended to be used with the text
//! `Vectorizer`

mod naive;

pub use self::naive::NaiveTokenizer;

use learning::LearningResult;

/// A trait used to construct Tokenizers
pub trait Tokenizer {
    /// Tokenize the inputs
    fn tokenize(&mut self, input: &str) -> LearningResult<Vec<String>>;
}
