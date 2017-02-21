/// An example of how Latent Diriclhet Allocation (LDA) can be used.  This example begins by
/// generating a distribution of words to categories.  This distribution is created so that
/// there are 10 topics.  Each of the 25 words are assigned to two topics with equal probability.
/// (The distribution of words is printed to the screen as a chart.  Each entry in the chart
/// corresponds to a word in the vocabulary, arranged into a square for easy viewing).  Documents
/// are then generated based on these distributions (each topic is assumed equally likely to be
/// assigned to a document, but each document has only one topic).
///
/// Once the documents are created, then the example uses LDA to attempt to reverse engineer the
/// distrbution of words, and prints the results to the screen for comparison.

extern crate rusty_machine;
extern crate rand;
extern crate rulinalg;

use rusty_machine::linalg::{Matrix, BaseMatrix, Vector};
use rusty_machine::data::transforms::{TransformFitter, LDAFitter};

use rand::{thread_rng, Rng};
use rand::distributions::{gamma, IndependentSample};

use std::cmp::max;

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

/// Chooses from a vector of probailities.
fn choose_from(probability: &Vector<f64>) -> usize {
    let mut rng = thread_rng();
    let selection:f64 = rng.next_f64();
    let mut total:f64 = 0.0;
    for (index, p) in probability.iter().enumerate() {
        total += *p;
        if total >= selection {
            return index;
        }
    }
    return probability.size() - 1;
}

/// Displays the distrbution of words to a topic as a square graph
fn topic_to_string(topic: &Vector<f64>, width: usize, topic_index: usize) -> String {
    let max = topic.iter().fold(0.0, |a, b|{
        if a > *b {
            a
        } else {
            *b
        }
    });
    let mut result = String::with_capacity(topic.size() * (topic.size()/width) + 18);
    result.push_str(&format!("Topic {}\n", topic_index));
    result.push_str("-------\n");
    for (index, element) in topic.iter().enumerate() {
        let col = index % width;
        let out = element / max * 9.0;
        if out >= 1.0 {
            result.push_str(&(out as u32).to_string());
        } else {
            result.push('.');
        }
        if col == width - 1 {
            result.push('\n');
        }
    }
    result
}


/// Prints a collection of multiline strings in columns
fn print_multi_line(o: &Vec<String>, column_width: usize) {
    let o_split:Vec<_> = o.iter().map(|col| {col.split('\n').collect::<Vec<_>>()}).collect();
    let mut still_printing = true;
    let mut line_index = 0;
    while still_printing {
        let mut gap = 0;
        still_printing = false;
        for col in o_split.iter() {
            if col.len() > line_index {
                if gap > 0 {
                    print!("{:width$}", "", width=column_width * gap);
                    gap = 0;
                }
                let line = col[line_index];
                print!("{:width$}", line, width=column_width);
                still_printing = true;
            } else {
                gap += 1;
            }
        }
        print!("\n");
        line_index += 1

    }
}


/// Prints the word distribution within topics
fn print_topic_distribution(dist: &Matrix<f64>, topic_count: usize, width: usize) {
    let top_strings = &dist.row_iter().take(topic_count/2).enumerate().map(|(topic_index, topic)|topic_to_string(&topic.into(), width, topic_index + 1)).collect();
    let bottom_strings = &dist.row_iter().skip(topic_count/2).enumerate().map(|(topic_index, topic)|topic_to_string(&topic.into(), width, topic_index + 1 + topic_count / 2)).collect();

    print_multi_line(top_strings, max(12, width + 1));
    print_multi_line(bottom_strings, max(12, width + 1));
}

pub fn main() {
    // Set initial constants
    // These can be changed as you wish
    let topic_count = 10;
    let document_length = 100;
    let document_count = 500;
    let alpha = 0.1;

    // Don't change these though; they are calculated based on the above
    let width = topic_count / 2;
    let vocab_count = width * width;
    println!("Creating word distribution");
    let word_distribution = generate_word_distribution(topic_count);
    println!("Distrbution generated:");
    print_topic_distribution(&word_distribution, topic_count, width);
    println!("Generating documents");
    let input = generate_documents(&word_distribution, topic_count, vocab_count, document_count, document_length, alpha);
    let lda = LDAFitter::new(topic_count, alpha, 0.1, 30);
    println!("Predicting word distrbution from generated documents");
    let result =  lda.fit(&input).unwrap();
    let dist = result.word_distribution();
    println!("Prediction completed.  Predicted word distribution:");
    println!("(Should be similar generated distribution above)", );

    print_topic_distribution(&dist, topic_count, width);


}
