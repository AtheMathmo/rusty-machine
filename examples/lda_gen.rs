extern crate rusty_machine;
extern crate rand;
extern crate rulinalg;

use rusty_machine::linalg::{Matrix, BaseMatrix, Vector};
use rusty_machine::learning::UnSupModel;
use rusty_machine::learning::lda::LDA;

use rand::{thread_rng, Rng};
use rand::distributions::{gamma, IndependentSample};

/// Given `topic_count` topics, this function will create a distrbution of words for each
/// topic.  For simplicity, this function assumes that the total number of words in the corpus
/// will be `(topic_count / 2)^2`.
fn generate_word_distribution(topic_count: usize) -> Matrix<f64> {
    let width = topic_count / 2;
    let vocab_size = width * width;
    let initial_value = 1.0 / width as f64;
    Matrix::from_fn(topic_count, vocab_size, |col, row| {
        if row < width {
            // Horizontal topics
            if col / width == row {
                initial_value
            } else {
                0.0
            }
        } else {
            //Vertical topics
            if col % width == (row - width) {
                initial_value
            } else {
                0.0
            }
        }
    })
}

/// Samples `count` times from a dirichlet distribution with alpha as given and
/// beta 1.0.
fn get_dirichlet(count: usize, alpha: f64) -> Vector<f64> {
    let mut rng = thread_rng();
    let g_dist = gamma::Gamma::new(alpha, 1.0);
    let result = Vector::from_fn(count, |_| {
        g_dist.ind_sample(&mut rng)
    });
    let sum = result.sum();
    result / sum
}

/// Generates a document based on a word distributiion as given.  The topics are randomly sampled
/// from a dirichlet distribution and then the word sampled from the selected topic.
fn generate_document(word_distribution: &Matrix<f64>, topic_count:usize, vocab_size: usize, document_length: usize, alpha: f64) -> Vec<usize> {
    let mut document = vec![0; vocab_size];
    let topic_distribution = get_dirichlet(topic_count, alpha);
    for _ in 0..document_length {
        let topic = choose_from(&topic_distribution);
        let word = choose_from(&word_distribution.row(topic).into());
        document[word] += 1;
    }
    document
}

/// Generate a collection of documents based on the word distribution
fn generate_documents(word_distribution: &Matrix<f64>, topic_count: usize, vocab_size: usize, document_count: usize, document_length: usize, alpha: f64) -> Matrix<usize> {
    let mut documents = Vec::with_capacity(vocab_size * document_count);
    for _ in 0..document_count {
        documents.append(&mut generate_document(word_distribution, topic_count, vocab_size, document_length, alpha));
    }
    Matrix::new(document_count, vocab_size, documents)
}

/// Displays the distrbution of words to a topic as a square graph
fn display_topic(topic: &Vector<f64>, width: usize) {
    let max = topic.iter().fold(0.0, |a, b|{
         if a > *b {
            a
        } else {
            *b
        }
    });
    for (index, element) in topic.iter().enumerate() {
        let col = index % width;
        let out = element / max * 9.0;
        print!("{}", out as usize);
        if col == width - 1 {
            print!("\n");
        }
    }
}

/// Chooses from a vector of probailities.
fn choose_from(probability: &Vector<f64>) -> usize {
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

pub fn main() {
    let topic_count = 10;
    let document_length = 100;
    let document_count = 500;
    let alpha = 0.1;
    let width = topic_count / 2;
    let vocab_count = width * width;
    println!("Generating documents");
    let word_distribution = generate_word_distribution(topic_count);
    let input = generate_documents(&word_distribution, topic_count, vocab_count, document_count, document_length, alpha);
    let lda = LDA::new(topic_count, alpha, 0.1);
    println!("Predicting");
    let result =  lda.predict(&(input, 30)).unwrap();
    let dist = result.phi();
    println!("Prediction completed");
    for (topic, row) in dist.row_iter().enumerate() {
        println!("\nTopic {}", topic);
        display_topic(&row.into(), width);
    }

}
