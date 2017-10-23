use std::collections::BTreeMap;

use linalg::Vector;

fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0. {
        0.
    } else {
        x * y.ln()
    }
}

/// Count target label frequencies
fn freq(labels: &Vector<usize>) -> (Vector<usize>, Vector<usize>) {
    let mut map: BTreeMap<usize, usize> = BTreeMap::new();
    for l in labels {
        let e = map.entry(*l).or_insert(0);
        *e += 1;
    }

    let mut uniques: Vec<usize> = Vec::with_capacity(map.len());
    let mut counts: Vec<usize> = Vec::with_capacity(map.len());
    for (&k, &v) in map.iter() {
        uniques.push(k);
        counts.push(v);
    }
    (Vector::new(uniques), Vector::new(counts))
}

pub fn label_counts(labels: &Vector<usize>, n_classes: usize) -> Vector<f64> {
    // ToDo: make this private
    debug_assert!(n_classes >= 1);
    debug_assert!(*labels.iter().max().unwrap() <= n_classes - 1);

    let mut counts: Vec<f64> = vec![0.0f64; n_classes];

    unsafe {
        for &label in labels.iter() {
            *counts.get_unchecked_mut(label) += 1.;
        }
    }
    Vector::new(counts)
}

/// Split criterias
#[derive(Debug, Clone)]
pub enum Metrics {
    // ToDo: remove clone

    /// Gini impurity
    Gini,
    /// Information gain
    Entropy
}

impl Metrics {

    /// calculate metrics from target labels
    pub fn from_labels(&self, labels: &Vector<usize>, n_classes: usize) -> f64 {
        let counts = label_counts(labels, n_classes);
        let sum: f64 = labels.size() as f64;
        let probas: Vector<f64> = counts / sum;
        self.from_probas(&probas.data())
    }

    /// calculate metrics from label probabilities
    pub fn from_probas(&self, probas: &[f64]) -> f64 {
        match self {
            &Metrics::Entropy => {
            let res: f64 = probas.iter().map(|&x| xlogy(x, x)).sum();
            - res
        },
        &Metrics::Gini => {
            let res: f64 =  probas.iter().map(|&x| x * x).sum();
            1.0 - res
        }
      }
    }
}

pub struct Splitter {
    total_counts: Vec<f64>,
    sorter: Vec<(f64, usize)>
}

impl Splitter {
    pub fn new(features: &Vec<f64>, target: &Vector<usize>,
               total_counts: &Vec<f64>) -> Self {

        debug_assert!(features.len() == target.size());
        debug_assert!(features.len() > 0);

        let mut sorter: Vec<(f64, usize)> = Vec::with_capacity(features.len());
        for (&f, &t) in features.iter().zip(target.iter()) {
            sorter.push((f, t));
        }
        sorter.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

        Splitter {
            total_counts: total_counts.clone(),
            sorter: sorter
        }
    }

    pub fn get_max_splits(&self, metric: &Metrics) -> Vec<(f64, f64)> {
        let (mut prev_val, prev_label) = unsafe { *self.sorter.get_unchecked(0) };
        let mut left_counts = vec![0.0f64; self.total_counts.len()];
        unsafe {
            *left_counts.get_unchecked_mut(prev_label) += 1.;
        }

        // ToDo: compare perf whether to store total as f64
        let mut left_total: f64 = 1.0f64;
        let mut right_counts: Vec<f64> = self.total_counts.iter()
                                                          .zip(left_counts.iter())
                                                          .map(|(&t, &c)| t - c)
                                                          .collect();
        let mut right_total: f64 = (self.sorter.len() - 1) as f64;

        // stores tuple of split value and criterion
        let mut res: Vec<(f64, f64)> = Vec::with_capacity(self.sorter.len());

        for &(current_val, current_label) in self.sorter.iter().skip(1) {
            if prev_val != current_val {
                let split = (prev_val + current_val) / 2.0f64;
                let lp: Vec<f64> = left_counts.iter().map(|&x| x / left_total).collect();
                let rp: Vec<f64> = right_counts.iter().map(|&x| x / right_total).collect();
                let lc = metric.from_probas(&lp) * left_total;
                let rc = metric.from_probas(&rp) * right_total;
                res.push((split, lc + rc));
            }

            unsafe {
                *left_counts.get_unchecked_mut(current_label) += 1.0f64;
                *right_counts.get_unchecked_mut(current_label) -= 1.0f64;
            }
            left_total += 1.0f64;
            right_total -= 1.0f64;

            prev_val = current_val;
        }
        res
    }
}

#[cfg(test)]
mod tests {

    use linalg::Vector;

    use super::{xlogy, freq, Metrics, Splitter};

    #[test]
    fn test_xlogy() {
        assert_eq!(xlogy(3., 8.), 6.2383246250395068);
        assert_eq!(xlogy(0., 100.), 0.);
    }

    #[test]
    fn test_freq() {
        let (uniques, counts) = freq(&Vector::new(vec![1, 2, 3, 1, 2, 4]));
        assert_eq!(uniques, Vector::new(vec![1, 2, 3, 4]));
        assert_eq!(counts, Vector::new(vec![2, 2, 1, 1]));

        let (uniques, counts) = freq(&Vector::new(vec![1, 2, 2, 2, 2]));
        assert_eq!(uniques, Vector::new(vec![1, 2]));
        assert_eq!(counts, Vector::new(vec![1, 4]));
    }

    #[test]
    fn test_entropy() {
        assert_eq!(Metrics::Entropy.from_probas(&vec![1.]), 0.);
        assert_eq!(Metrics::Entropy.from_probas(&vec![1., 0., 0.]), 0.);
        assert_eq!(Metrics::Entropy.from_probas(&vec![0.5, 0.5]), 0.69314718055994529);
        assert_eq!(Metrics::Entropy.from_probas(&vec![1. / 3., 1. / 3., 1. / 3.]), 1.0986122886681096);
        assert_eq!(Metrics::Entropy.from_probas(&vec![0.4, 0.3, 0.3]), 1.0888999753452238);
    }

    #[test]
    fn test_gini_from_probas() {
        assert_eq!(Metrics::Gini.from_probas(&vec![1., 0., 0.]), 0.);
        assert_eq!(Metrics::Gini.from_probas(&vec![1. / 3., 1. / 3., 1. / 3.]), 0.6666666666666667);
        assert_eq!(Metrics::Gini.from_probas(&vec![0., 1. / 46., 45. / 46.]), 0.04253308128544431);
        assert_eq!(Metrics::Gini.from_probas(&vec![0., 49. / 54., 5. / 54.]), 0.16803840877914955);
    }

    #[test]
    fn test_entropy_from_labels() {
        assert_eq!(Metrics::Entropy.from_labels(&Vector::new(vec![0, 1, 2]), 3), 1.0986122886681096);
        assert_eq!(Metrics::Entropy.from_labels(&Vector::new(vec![0, 0, 1, 1]), 2), 0.69314718055994529);
    }

    #[test]
    fn test_gini_from_labels() {
        assert_eq!(Metrics::Gini.from_labels(&Vector::new(vec![1, 1, 1]), 2), 0.);
        assert_eq!(Metrics::Gini.from_labels(&Vector::new(vec![0, 0, 0]), 2), 0.);
        assert_eq!(Metrics::Gini.from_labels(&Vector::new(vec![0, 0, 1, 1, 2, 2]), 3), 0.6666666666666667);
    }

    #[test]
    fn test_splitter() {
        let features: Vec<f64> = vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0];
        let labels: Vector<usize> = Vector::new(vec![0, 1, 1, 1, 0, 0]);

        let s = Splitter::new(&features, &labels, &vec![3., 3.]);
        let res = s.get_max_splits(&Metrics::Gini);
        assert_eq!(res.len(), 3);

        let exp = Metrics::Gini.from_labels(&Vector::new(vec![0, 1]), 2) * 2. +
                  Metrics::Gini.from_labels(&Vector::new(vec![0, 0, 1, 1]), 2) * 4.;
        assert_eq!(res[0], (1.5, exp));

        let exp = Metrics::Gini.from_labels(&Vector::new(vec![0, 1, 1, 1]), 2) * 4. +
                  Metrics::Gini.from_labels(&Vector::new(vec![0, 0]), 2) * 2.;
        assert_eq!(res[1], (2.5, exp));

        let exp = Metrics::Gini.from_labels(&Vector::new(vec![0, 0, 1, 1, 1]), 2) * 5. +
                  Metrics::Gini.from_labels(&Vector::new(vec![0]), 2) * 1.;
        assert_eq!(res[2], (3.5, exp));
    }
}
