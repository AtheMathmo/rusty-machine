use rm::data::vectorizers::text::FreqVectorizer;
use rm::data::tokenizers::{NaiveTokenizer};
use rm::data::vectorizers::Vectorizer;
use rm::prelude::Matrix;

#[test]
fn test_frequency_vectorizer() {
    let mut freq_vectorizer = FreqVectorizer::<f32, NaiveTokenizer>::new(NaiveTokenizer::new());
    let fit_inputs = vec!["This is fit"];
    freq_vectorizer.fit(&fit_inputs).unwrap();

    let inputs = vec!["this is vectorize",
                      "this is not fit"];
    let vectorized = freq_vectorizer.vectorize(&inputs).unwrap();
    let expected = Matrix::new(2, 3, vec![1., 1., 0., 1., 1., 1.]);
    assert_eq!(vectorized, expected);
}
