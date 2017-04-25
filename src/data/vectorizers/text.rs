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
//! use rusty_machine::data::tokenizers::NaiveTokenizer;
//!
//! // Constructs an empty `FreqVectorizer` vectorizer
//! let mut freq_vectorizer = FreqVectorizer::<f32, NaiveTokenizer>::new(NaiveTokenizer::new());
//!
//! let inputs = vec!["This is test"];
//!
//! // Fit the vectorizer
//! freq_vectorizer.fit(&inputs).unwrap();
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
use super::super::tokenizers::Tokenizer;

/// FreqVectorizer vectorizer for text
///
/// This provides an implementation of `Vectorizer`
/// which allows us to vectorize strings by doing a non-normalized frequency count
///
/// See the module description for more information.
#[derive(Debug)]
pub struct FreqVectorizer<U: Float, T: Tokenizer> {
    words: HashMap<String, usize>,
    tokenizer: T,
    float_type: PhantomData<U>,
}

impl<'a, U: Float, T: Tokenizer> Vectorizer<&'a str, Matrix<U>> for FreqVectorizer<U, T> {
    fn vectorize(&mut self, texts: &[&'a str]) -> LearningResult<Matrix<U>> {
        let mut result = Matrix::zeros(texts.len(), self.words.len());
        for (text, row) in texts.iter().zip((0..texts.len())) {
            let tokens = self.tokenizer.tokenize(text).unwrap();
            for token in tokens {
                if let Some(&col) = self.words.get(&token) {
                    result[[row, col]] = result[[row, col]] + U::one();
                }
            }
        }
        Ok(result)
    }
    fn fit(&mut self, texts:  &[&str]) -> LearningResult<()> {
        self.words = HashMap::new();
        let mut index: usize = 0;
        for text in texts {
            let tokens = self.tokenizer.tokenize(text).unwrap();
            for token in tokens {
                if !self.words.contains_key(&token) {
                    self.words.insert(token, index);
                    index += 1;
                }
            }
        }
        Ok(())
    }
}

impl<U: Float, T: Tokenizer> FreqVectorizer<U, T> {
    /// Constructs an empty `FreqVectorizer` vectorizer.
    pub fn new(tokenizer: T) -> Self {
        FreqVectorizer {
            words: HashMap::new(),
            tokenizer: tokenizer,
            float_type: PhantomData,
        }
    }
}
