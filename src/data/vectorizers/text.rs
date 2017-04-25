//! Text Vectorizers
//!
//! This module contains some text vectorizers.
//!
//! The `FreqVectorizer` vectorizer is used to vectorize vectors of
//! strings using a traditional Bag of Words where each string
//! is splitted by whitespaces and each word is transformed into
//! lowercase
//!
//! # Examples
//!
//! ```
//! use rusty_machine::data::vectorizers::Vectorizer;
//! use rusty_machine::data::vectorizers::text::FreqVectorizer;
//!
//! // Constructs an empty `FreqVectorizer` vectorizer
//! let mut freq_vectorizer = FreqVectorizer::<f32>::new();
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

/// FreqVectorizer vectorizer for text
///
/// This provides an implementation of `Vectorizer`
/// which allows us to vectorize strings by doing a non-normalized frequency count
///
/// See the module description for more information.
#[derive(Debug)]
pub struct FreqVectorizer<T: Float> {
    words: HashMap<String, usize>,
    float_type: PhantomData<T>
}

impl<'a, T: Float> Vectorizer<&'a str, Matrix<T>> for FreqVectorizer<T> {
    fn vectorize(&mut self, texts: &[&'a str]) -> LearningResult<Matrix<T>> {
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
    fn fit(&mut self, texts:  &[&str]) -> LearningResult<()> {
        self.words = HashMap::new();
        let mut index: usize = 0;
        for text in texts {
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
        Ok(())
    }
}

impl<T: Float> FreqVectorizer<T> {
    /// Constructs an empty `FreqVectorizer` vectorizer.
    pub fn new() -> Self {
        FreqVectorizer {
            words: HashMap::new(),
            float_type: PhantomData
        }
    }
}
