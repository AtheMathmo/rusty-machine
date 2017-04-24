//! Text Vectorizers
//!
//! This module contains some text vectorizers.
//!
//! The `Frequency` vectorizer is used to vectorize vectors of
//! strings using a traditional Bag of Words where each string
//! is splitted by whitespaces and each word is transformed into
//! lowercase
//!
//! # Examples
//!
//! ```
//! use rusty_machine::data::vectorizers::Vectorizer;
//! use rusty_machine::data::vectorizers::text::Frequency;
//!
//! // Constructs an empty `Frequency` vectorizer
//! let mut freq_vectorizer = Frequency::<f32>::new();
//!
//! let inputs = vec!["This is test".to_string()];
//!
//! // Transform the inputs
//! let vectorized = freq_vectorizer.vectorize(&inputs).unwrap();
//! ```

use libnum::Float;
use std::collections::HashMap;

use std::marker::PhantomData;
use linalg::Matrix;
use learning::LearningResult;
use super::Vectorizer;

/// Frequency vectorizer for text
///
/// This provides an implementation of `Vectorizer`
/// which allows us to vectorize strings by doing a non-normalized frequency count
///
/// See the module description for more information.
#[derive(Debug)]
pub struct Frequency<T: Float> {
    words: HashMap<String, usize>,
    float_type: PhantomData<T>
}

impl<T: Float> Vectorizer<Vec<String>, Matrix<T>> for Frequency<T> {
    fn vectorize(&mut self, texts: &Vec<String>) -> LearningResult<Matrix<T>> {
        let mut result = Matrix::zeros(texts.len(), self.words.len());
        for (text, row) in texts.iter().zip((0..texts.len())) {
            // this needs improvement we could offer more options
            let tokens = text.split_whitespace();
            for token in tokens {
                let token = token.to_lowercase();
                if let Some(&col) = self.words.get(&token) {
                    result[[row, col]] = result[[row, col]] + T::one();
                }
            }
        }
        Ok(result)
    }
    fn fit(&mut self, texts: Vec<String>) -> LearningResult<Matrix<T>> {
        self.words = HashMap::new();
        let mut index: usize = 0;
        for text in &texts {
            // this needs improvement we could offer more options
            let tokens = text.split_whitespace();
            for token in tokens {
                let token = token.to_lowercase();
                if !self.words.contains_key(&token) {
                    self.words.insert(token, index);
                    index += 1;
                }
            }
        }
        self.vectorize(&texts)
    }
}

impl<T: Float> Frequency<T> {
    /// Constructs an empty `Frequency` vectorizer.
    pub fn new() -> Self {
        Frequency {
            words: HashMap::new(),
            float_type: PhantomData
        }
    }
}
