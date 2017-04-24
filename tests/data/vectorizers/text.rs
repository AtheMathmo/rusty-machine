use rm::data::vectorizers::text::Frequency;
use rm::data::vectorizers::Vectorizer;
use rm::prelude::Matrix;

#[test]
fn test_frequency_vectorizer() {
    let mut freq_vectorizer = Frequency::<f32>::new();
    let fit_inputs = vec!["This is fit".to_string()];
    freq_vectorizer.fit(fit_inputs).unwrap();

    let inputs = vec!["this is vectorize".to_string(),
                      "this is not fit".to_string()];
    let vectorized = freq_vectorizer.vectorize(&inputs).unwrap();
    let expected = Matrix::new(2, 3, vec![1., 1., 0., 1., 1., 1.]);
    assert_eq!(vectorized, expected);
}
