//! Decision Tree Module
//!
//! Contains implementation of decision tree.
//!
//! The Decisin Tree models currently only support binary tree.
//! The model inputs should be a matrix and the training targets are
//! in the form of a vector of usize target labels, like 0, 1, 2...
//!
//! # Examples
//!
//! ```
//! use rusty_machine::learning::tree::DecisionTreeClassifier;
//! use rusty_machine::learning::SupModel;
//!
//! use rusty_machine::linalg::{Matrix, Vector};
//!
//! let inputs = Matrix::new(3, 2, vec![1., 1., 1., 2., 2., 2.]);
//! let targets = Vector::new(vec![0, 1, 1]);
//! let mut tree = DecisionTreeClassifier::default();
//!
//! // Train the model
//! tree.train(&inputs, &targets).unwrap();
//!
//! // Now we'll predict a new point
//! let new_data = Matrix::new(1, 2, vec![1., 0.5]);
//! let output = tree.predict(&new_data).unwrap();
//!
//! // Hopefully we classified our new point correctly!
//! println!("{}", output[0]);
//! assert!(output[0] == 0, "Our classifier isn't very good!");
//! ```
use linalg::{Matrix, BaseMatrix};
use linalg::Vector;

use learning::{LearningResult, SupModel};
use learning::error::{Error, ErrorKind};

mod criterion;

pub use self::criterion::Metrics;
use self::criterion::{label_counts, Splitter};

/// Tree node
#[derive(Debug)]
struct Node {
    feature_index: usize,
    threshold: f64,
    left: Link,
    right: Link
}

/// Tree link (leaf or branch)
///
/// Leaf contains a label to predict
#[derive(Debug)]
enum Link {
    Leaf(usize),
    Branch(Box<Node>),
}

/// Decision Tree
#[derive(Debug)]
pub struct DecisionTreeClassifier {

    criterion: Metrics,
    max_depth: Option<usize>,
    min_samples_split: Option<usize>,

    // params set after train
    n_classes: usize,
    n_features: usize,
    root: Option<Link>
}

/// The default Decision Tree.
///
/// The defaults are:
///
/// - `max_depth` = `None`
/// - `min_samples_split` = `None`
impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        DecisionTreeClassifier{ criterion: Metrics::Gini,
                                max_depth: None,
                                min_samples_split: None,
                                n_classes: 0,
                                n_features: 0,
                                root: None }
    }
}

impl DecisionTreeClassifier {

    /// Constructs an untrained Decision Tree with specified
    ///
    /// - `criterion` - Decision tree criteria
    /// - `max_depth` - Maximum depth of the tree
    /// - `min_samples_split` - Minimum samples to split a branch.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::learning::tree::{DecisionTreeClassifier, Metrics};
    ///
    /// let _ = DecisionTreeClassifier::new(Metrics::Gini, 3, 30);
    /// ```
    pub fn new(criterion: Metrics, max_depth: usize, min_samples_split: usize) -> Self {
        DecisionTreeClassifier{ criterion: criterion,
                                max_depth: Some(max_depth),
                                min_samples_split: Some(min_samples_split),
                                n_classes: 0,
                                n_features: 0,
                                root: None }
    }
}

impl DecisionTreeClassifier {

    /// Calculate metrics
    fn metrics_weighted(&self, target: &Vector<usize>) -> f64 {
        self.criterion.from_labels(target, self.n_classes) * (target.size() as f64)
    }

    /// Check termination criteria
    fn can_split(&self, current_target: &Vector<usize>, depth: usize) -> bool {

        // Avoid to match every time
        match self.max_depth {
            None => {},
            Some(max_depth) => {
                if depth >= max_depth {
                    return false
                }
            }
        }
        match self.min_samples_split {
            None => {},
            Some(min_samples_split) => {
                if current_target.size() <= min_samples_split {
                    return false
                }
            }
        }
        true
    }

    /// Determine whether to split a node
    ///
    /// - `inputs` - Reference to the original data.
    /// - `target` - Reference to the original target.
    /// - `remains` - Index of rows to be considered.
    /// - `depth` - Depth of the node.
    /// - `criteria` - Parent node's criteria value
    fn split(&self, inputs: &Matrix<f64>, target: &Vector<usize>,
             remains: &[usize], depth: usize, prev_criteria: f64) -> Link {

        let current_target: Vector<usize> = target.select(remains);

        // ToDo: skip label_counts to simply check self.can_split
        let counts: Vector<f64> = label_counts(&current_target, self.n_classes);
        let (idx, max) = counts.argmax();
        // stop splitting
        if (max == current_target.size() as f64) | !self.can_split(&current_target, depth) {
            return Link::Leaf(idx)
        }

        let mut split_col: usize = 0;
        let mut split_val: f64 = 0.;

        let mut criteria = prev_criteria;
        let mut criteria_left: f64 = 0.;
        let mut criteria_right: f64 = 0.;

        for i in 0..inputs.cols() {
            // target feature
            let current_feature: Vec<f64> = inputs.select(remains, &[i])
                                                  .into_vec();

            let s = Splitter::new(&current_feature,
                                  &current_target, &counts.data());

            for (v, cr) in s.get_max_splits(&self.criterion).into_iter() {
                if cr < criteria {
                    split_col = i;
                    split_val = v;
                    criteria = cr
                }
            }

        }
        // ToDo: possible to optimize to remember split location
        let mut li: Vec<usize> = Vec::with_capacity(remains.len());
        let mut ri: Vec<usize> = Vec::with_capacity(remains.len());
        for (&v, &r) in inputs.select(remains, &[split_col]).iter().zip(remains.iter()) {
            if v < split_val {
                li.push(r);
            } else {
                ri.push(r);
            }
        }

        let ln = self.split(inputs, target, &li, depth + 1, criteria); //criteria_left);
        let rn = self.split(inputs, target, &ri, depth + 1, criteria); //criteria_right);
        Link::Branch(Box::new(Node{ feature_index: split_col,
                                    threshold: split_val,
                                    left: ln,
                                    right: rn }))
    }

    /// Predict a single row
    ///
    /// - `current` - Reference to the root link.
    /// - `row` - Reference to the single row (row slice of the input Matrix).
    fn predict_row(&self, mut current: &Link, row: &[f64]) -> usize {
        loop {
            match current {
                &Link::Leaf(label) => return label,
                &Link::Branch(ref n) => unsafe {
                    if *row.get_unchecked(n.feature_index) < n.threshold {
                        current = &n.left
                    } else {
                        current = &n.right
                    }
                }
            };
        }
    }

    /// Desciribe tree structure
    pub fn to_graphviz(&self) -> Result<(), Error> {
        match self.root {
            None => Err(Error::new_untrained()),
            Some(ref root) => {
                self.to_graphviz_node(root, 0, 0);
                Ok(())
            }
        }
    }

    /// Desciribe tree node
    ///
    /// - `current` - Reference to the root link.
    /// - `row` - Reference to the single row (row slice of the input Matrix).
    fn to_graphviz_node(&self, current: &Link, previd: usize, mut curid: usize) -> usize {
        match current {
            &Link::Leaf(label) => {
                println!("node{}[label=\"label={}\"];", curid, label);
                println!("node{} -> node{};", previd, curid + 1);
                return curid + 1;
            },
            &Link::Branch(ref n) => {
                println!("node{}[label=\"col {} < {}\"];", curid, n.feature_index, n.threshold);
                println!("node{} -> node{};", previd, curid + 1);
                let nid = self.to_graphviz_node(&n.left, curid + 1, curid + 1);
                self.to_graphviz_node(&n.right, curid, nid)
            }
        }
    }
}

/// Train the model and predict the model output from new data.
impl SupModel<Matrix<f64>, Vector<usize>> for DecisionTreeClassifier {

    fn predict(&self, inputs: &Matrix<f64>) -> LearningResult<Vector<usize>> {
        match self.root {
            None => Err(Error::new_untrained()),
            Some(ref root) => {
                if self.n_features != inputs.cols() {
                    Err(Error::new(ErrorKind::InvalidData,
                                   "Input data do not have the same dimensions as training data"))
                } else {

                    let results: Vec<usize> = inputs.iter_rows()
                                                    .map(|x| self.predict_row(root, x))
                                                    .collect();
                    Ok(Vector::new(results))
                }
            }
        }
    }

    fn train(&mut self, data: &Matrix<f64>, target: &Vector<usize>) -> LearningResult<()> {
        // set feature and target params

        if data.rows() != target.size() {
            panic!("error");
        }
        if target.size() == 0 {
            panic!("error");
        }

        self.n_classes = *target.iter().max().unwrap() + 1;
        self.n_features = data.cols();

        let all: Vec<usize> = (0..target.size()).collect();
        let c = self.metrics_weighted(&target);
        let root = self.split(data, target, &all, 0, c);
        self.root = Some(root);
        Ok(())
    }
}


#[cfg(test)]
mod tests {

    use linalg::Vector;

    #[test]
    fn test_xxx() {

    }
}
