//! Latent Dirichlet Allocation Module
//!
//! Contains an implementation of Latent Dirichlet Allocation (LDA) using
//! [Gibbs Sampling](http://psiexp.ss.uci.edu/research/papers/sciencetopics.pdf)
//!
//! LDAFitter is typically used for textual analysis.  It assumes that a topic can be modeled
//! as a distribution of words, and that a document can be modeled as a distribution of
//! topics.  Thus, the output is these distributions on the input documents.
//!
//! Gibbs sampling is a Morkov Chain Monto Carlo algorithm that iteratively approximates
//! the above distributions.
//!
//! This module is able to estimate a distribution of categories over documents in an
//! unsupervised manner, and use that distrbution to estimate the categories over other
//! models.
//!
//! # Examples
//!
//! ```
//! #[allow(unused_variables)]
//! use rusty_machine::linalg::Matrix;
//! use rusty_machine::data::transforms::{LDAFitter, Transformer, TransformFitter};
//!
//! // Create a basic input array with 4 documents and 3 words
//! let input = Matrix::ones(4, 3);
//!
//! // Create a model that will find 5 topics, with parameters alpha=0.1, beta=0.1
//! let lda = LDAFitter::new(5, 0.1, 0.1, 10);
//!
//! // Fit the model to the input
//! let mut result = lda.fit(&input).unwrap();
//!
//! // Find the estimated distrbution of words over categories
//! let dist = result.word_distribution();
//!
//! // Use the model to estimate some topics for new documents
//! let esimtated = result.transform(Matrix::ones(6, 3)).unwrap();
//! ```


use linalg::{Matrix, Vector, BaseMatrix};
use super::{Transformer, TransformFitter};
use rulinalg::matrix::BaseMatrixMut;
use rand::{Rng, thread_rng};
use learning::LearningResult;

/// Latent Dirichlet Allocation
#[derive(Debug)]
pub struct LDAFitter {
    iterations: usize,
    topic_count: usize,
    alpha: f64,
    beta: f64,
}

/// An object which holds the results of Gibbs Sampling.
///
/// This object can then be used to get the distrbutions of
/// topics over documents and words over topics
#[derive(Debug)]
pub struct LDAModel {
    document_topic_count: Matrix<f64>,
    topic_word_count: Matrix<f64>,
    // The two vectors are simply used to reduce the amount of calculation that must be
    // performed during the conditional probability step
    topic_total_by_document: Vector<f64>,
    word_total_by_topic: Vector<f64>,
    alpha: f64,
    beta: f64,
}

/// Creates a default for LDAFitter with alpha = beta = 0.1, topic_coutn = 10 and iterations = 30
impl Default for LDAFitter {
    fn default() -> LDAFitter {
        LDAFitter {
            iterations: 30,
            topic_count: 10,
            alpha: 0.1,
            beta: 0.1
        }
    }
}

impl LDAFitter {
    /// Creates a new object for finding LDA.
    /// `alpha` and `beta` are the symmetric priors for the algorithm.
    /// `iterations` is the number of times the sampling algorithm will run.
    ///
    /// If you don't know what to use, try alpha = 50/topic_count and beta = 0.01
    pub fn new(topic_count: usize, alpha: f64, beta: f64, iterations: usize) -> LDAFitter {
        LDAFitter {
            topic_count: topic_count,
            alpha: alpha,
            beta: beta,
            iterations: iterations
        }
    }
}

impl LDAModel {
    fn new(input: &Matrix<usize>, topic_count: usize, alpha: f64, beta: f64) -> (LDAModel, Vec<Vec<usize> >) {
        let document_count = input.rows();
        let vocab_count = input.cols();
        let mut topics = Vec::with_capacity(document_count);
        let mut result = LDAModel {
            document_topic_count: Matrix::new(document_count, topic_count, vec![alpha; document_count * topic_count]),
            topic_word_count: Matrix::new(vocab_count, topic_count, vec![beta; topic_count * vocab_count]),
            topic_total_by_document: Vector::new(vec![alpha * topic_count as f64; document_count]),
            word_total_by_topic: Vector::new(vec![beta * vocab_count as f64; topic_count]),
            alpha: alpha,
            beta: beta
        };

        // For each word in each document, randomly assign it to a topic and update the counts
        let mut rng = thread_rng();
        for (document, row) in input.row_iter().enumerate() {
            let mut document_topics = Vec::with_capacity(row.sum());
            for (word, word_count) in row.iter().enumerate() {
                for _ in 0..*word_count{
                    let topic = rng.gen_range(0, topic_count);
                    result.document_topic_count[[document, topic]] += 1.0;
                    result.topic_total_by_document[document] += 1.0;
                    result.topic_word_count[[word, topic]] += 1.0;
                    result.word_total_by_topic[topic] += 1.0;
                    document_topics.push(topic);
                }
            }
            topics.push(document_topics);
        }
        (result, topics)
    }

    /// Find the distribution of words over topics.  This gives a matrix where the rows are
    /// topics and the columns are words.  Each entry `(topic, word)` gives the probability of
    /// `word` given `topic`.
    pub fn word_distribution(&self) -> Matrix<f64> {
        let mut distribution = self.topic_word_count.transpose();
        let row_sum = distribution.sum_rows();
        // XXX To my knowledge, there is not presently a way in rulinalg to divide a matrix
        // by a vector, so I do it with a loop here.
        for (mut row, sum) in distribution.row_iter_mut().zip(row_sum.iter()) {
            *row /= sum;
        }
        distribution
    }

    /// Finds the distribution of cateogries across the documents originally used to
    /// fit the model.  This gives a matrix where the rows are documents and the columns are
    /// topics.  Each entry `(document, topic)` gives the probability of `topic` given `word`
    pub fn category_distribution(&self) -> Matrix<f64> {
        let mut distribution = self.document_topic_count.clone();
        for (document, mut row) in distribution.row_iter_mut().enumerate() {
            for c in row.iter_mut() {
                *c /= self.topic_total_by_document[document];
            }
        }
        distribution
    }

}

impl LDAFitter {
    fn conditional_distribution(&self, result: &LDAModel, document: usize, word: usize) -> Vector<f64> {

        // Convert the column of the word count by topic into a vector
        let word_topic_count:Vector<f64> = unsafe {result.topic_word_count.row_unchecked(word)}.into();

        // Calculate the proportion of this word's contribution to each topic
        let left:Vector<f64> = (word_topic_count).elediv(&result.word_total_by_topic);

        // Convert the row of the topic count by document into a vector
        let topic_document_count:Vector<f64> = unsafe{result.document_topic_count.row_unchecked(document)}.into();

        // Calculate the proportion of each topic's contribution to this document
        let right:Vector<f64> =  topic_document_count / result.topic_total_by_document[document];

        // Multiply the proportions together to this word's contribution to the document by topic
        let mut probability:Vector<f64> = left.elemul(&right);

        // Normalize it so that it's a probability
        probability /= probability.sum();
        return probability;
    }
}

impl TransformFitter<Matrix<usize>, Matrix<f64>, LDAModel> for LDAFitter {
    /// Predict categories from the input matrix.
        fn fit(self, inputs: &Matrix<usize>) -> LearningResult<LDAModel> {
            let (mut result, mut topics) = LDAModel::new(inputs, self.topic_count, self.alpha, self.beta);
            let mut word_index:usize;
            for _ in 0..self.iterations {
                for (document, row) in inputs.row_iter().enumerate() {
                    word_index = 0;
                    let mut document_topics = unsafe{topics.get_unchecked_mut(document)};
                    for (word, word_count) in row.iter().enumerate() {
                        for _ in 0..*word_count {
                            // Remove the current word from the counts
                            let mut topic = *unsafe{document_topics.get_unchecked(word_index)};
                            result.document_topic_count[[document, topic]] -= 1.0;
                            result.topic_total_by_document[document] -= 1.0;
                            result.topic_word_count[[word, topic]] -= 1.0;
                            result.word_total_by_topic[topic] -= 1.0;

                            // Find the probability of that word being a part of each topic
                            // based on the other words in the present document
                            let probability = self.conditional_distribution(&result, document, word);

                            // Aandomly assign a new topic based on the probabilities from above
                            topic = choose_from(probability);

                            // Update the counts with the new topic
                            result.document_topic_count[[document, topic]] += 1.0;
                            result.topic_total_by_document[document] += 1.0;
                            result.topic_word_count[[word, topic]] += 1.0;
                            result.word_total_by_topic[topic] += 1.0;
                            document_topics[word_index] = topic;
                            word_index += 1;
                        }
                    }
                }
            }
            Ok(result)
        }
}

impl Transformer<Matrix<usize>, Matrix<f64>> for LDAModel {
    fn transform(&mut self, input: Matrix<usize>) -> LearningResult<Matrix<f64>> {
        assert!(input.cols() == self.topic_word_count.rows(), "The input matrix must have the same size vocabulary as the fitting model");
        let input = Matrix::from_fn(input.rows(), input.cols(), |col, row| {
            input[[row, col]] as f64
        });
        let mut distribution = input * &self.topic_word_count;

        let row_sum = distribution.sum_rows();
        for (mut row, sum) in distribution.row_iter_mut().zip(row_sum.iter()) {
            *row /= sum;
        }

        Ok(distribution)
    }
}

/// This function models sampling from the categorical distribution.
/// That is, given a series of discrete categories, each with different probability of occurring,
/// this function will choose a category according to their probabilities.
/// The sum of the probabilities must be 1, but since this is only used internally,
/// there is no need to verify that this is true.
fn choose_from(probability: Vector<f64>) -> usize {
    let mut rng = thread_rng();
    let selection:f64 = rng.gen_range(0.0, 1.0);
    let mut total:f64 = 0.0;
    for (index, p) in probability.iter().enumerate() {
        total += *p;
        if total >= selection {
            return index;
        }
    }
    return probability.size() - 1;
}

#[cfg(test)]
mod test {
    use super::{LDAModel, LDAFitter};
    use linalg::{Matrix, Vector};
    #[test]
    fn test_conditional_distribution() {
        let result = LDAModel {
            topic_word_count: Matrix::new(4, 3, vec![3.1, 0.1, 0.1,
                                                     3.1, 1.1, 0.1,
                                                     5.1, 1.1, 5.1,
                                                     0.1, 0.1, 2.1]),
            word_total_by_topic: Vector::new(vec![11.4, 2.4, 7.4]),
            document_topic_count: Matrix::new(2, 3, vec![1.1, 2.1, 3.1,
                                                         5.1, 5.1, 4.1]),
            topic_total_by_document: Vector::new(vec![6.3, 14.3]),
            alpha: 0.1,
            beta: 0.1,
        };
        let lda = LDAFitter::new(3, 0.1, 0.1, 1);
        let probability = lda.conditional_distribution(&result, 0, 2);

        // Calculated by hand and verified using https://gist.github.com/mblondel/542786
        assert_eq!(probability,
            Vector::new(vec![0.13703500146066358, 0.2680243410921803, 0.5949406574471561]));
    }
}
