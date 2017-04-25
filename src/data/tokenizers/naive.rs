use super::Tokenizer;
use learning::LearningResult;

/// NaiveTokenizer tokenizer
///
/// This provides an implementation of `Tokenizer`
/// which allows us to tokenize a string by splitting
/// it by whitespaces and lowering al the characters
#[derive(Debug)]
pub struct NaiveTokenizer;

impl Tokenizer for NaiveTokenizer {
    fn tokenize(&mut self, input: &str) -> LearningResult<Vec<String>> {
        let tokens = input
            .split_whitespace()
            .map(|token| token.to_lowercase())
            .collect::<Vec<String>>();
        Ok(tokens)
    }
}

impl NaiveTokenizer {
    /// Constructs a new `NaiveTokenizer` tokenizer.
    pub fn new() -> Self {
        NaiveTokenizer
    }
}
