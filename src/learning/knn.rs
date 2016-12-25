//! - k-Nearest Nerighbors
extern crate rulinalg;

use std::f64;
use std::collections::VecDeque;

use rulinalg::matrix::{Matrix, BaseMatrix, Axes};
use rulinalg::vector::{Vector};

struct KDTree {
    data: Matrix<f64>,
    leafsize: usize,

    root: Option<Node>
}

enum Node {
    Branch(Branch),
    Leaf(Leaf)
}

struct Branch {
    /// dimension to split
    dim: usize,
    /// split value
    split: f64,

    /// min and max of bounding box
    /// i.e. hyper-rectangle contained in the branch
    min: Vec<f64>,
    max: Vec<f64>,

    /// left node
    left: Box<Node>,
    /// right node
    right: Box<Node>,
}

impl Branch {
    fn new(dim: usize, split: f64,
           min: Vec<f64>, max: Vec<f64>,
           left: Node, right: Node) -> Self {

        Branch {
            dim: dim,
            split: split,

            min: min,
            max: max,

            left: Box::new(left),
            right: Box::new(right)
        }
    }

    fn dist(&self, point: &Vec<f64>) -> f64 {
        let mut d = 0.;
        println!("self.min {:?}", &self.min);
        println!("self.max {:?}", &self.max);
        println!("point    {:?}", &point);
        for ((&p, &mi), &ma) in point.iter()
                                     .zip(self.min.iter())
                                     .zip(self.max.iter()) {
            if p < mi {
                d += (mi - p) * (mi - p);
            } else if ma < p {
                d += (ma - p) * (ma - p);
            }
            // else included in hyper-rectangle
        }
        d.sqrt()
    }
}

struct Leaf {
    children: Vec<usize>
}

impl Leaf {
    fn new(children: Vec<usize>) -> Self {
        Leaf {
            children: children
        }
    }
}

impl KDTree {

    fn new(data: Matrix<f64>, leafsize: usize) -> Self {

        KDTree {
            data: data,
            leafsize: leafsize,

            root: None
        }
    }

    fn build(&mut self) {
        let remains: Vec<usize> = (0..self.data.rows()).collect();
        let dim: usize = 0;

        // sort each dimensions here
        let indexer: Vec<Vec<usize>> = Vec::with_capacity(self.data.cols());

        let dmin = min(&self.data).data().clone();
        let dmax = max(&self.data).data().clone();

        self.root = Some(self.split_one(remains, &dmin, &dmax, dim));
    }

    fn split_one(&self, mut remains: Vec<usize>, dmin: &Vec<f64>,
                 dmax: &Vec<f64>, dim: usize) -> Node {

        if remains.len() < self.leafsize {
            println!("create leaf! {:?}", &remains);
            Node::Leaf(Leaf::new(remains))
        } else {

            // ToDo:
            remains.sort_by(|&a, &b| self.data[[a, dim]].partial_cmp(&self.data[[b, dim]]).unwrap());
            let idx = remains.len() / 2;

            let split = self.data[[idx, dim]];

            // let l_idx: Vec<usize> = (0..idx).collect();
            let mut l_max = dmax.clone();
            l_max[dim] = split;

            //let g_idx: Vec<usize> = (idx..remains.len()).collect();
            let r_remains = remains.split_off(idx);
            let mut g_min = dmin.clone();
            g_min[dim] = split;
            println!("{:?}, {:?}", &remains, &r_remains);

            let l_node = self.split_one(remains, dmin, &l_max, (dim + 1) % self.data.cols());
            let g_node = self.split_one(r_remains, &g_min, dmax, (dim + 1) % self.data.cols());

            println!("create branch! {:?} {:?}", &dim, &split);
            Node::Branch(Branch::new(dim, split, dmin.clone(), dmax.clone(),
                                     l_node, g_node))
        }
    }

    /// return minimum distance between point and data specified with row ids
    fn min_dist(&self, point: &Vec<f64>, ids: &Vec<usize>) -> (usize, f64) {
        assert!(ids.len() > 0, "");

        let mut mind: f64 = f64::MAX;
        let mut minidx: usize = 0;
        for id in ids.iter() {
            let row: Vec<f64> = self.data.select_rows(&[*id]).into_vec();

            let d = dist(point, &row);
            if d < mind {
                mind = d;
                minidx = *id;
            }
        }
        (minidx, mind)
    }

    fn search_leaf<'s, 'a>(&'s self, point: &'a Vec<f64>)
        -> (usize, f64, VecDeque<&'s Node>) {

        match self.root {
            None => panic!(""),
            Some(ref root) => {

                let mut queue: VecDeque<&Node> = VecDeque::new();
                queue.push_back(root);
                loop {
                    let current: &Node = queue.front().unwrap();
                    match current {
                        &Node::Leaf(ref l) => {
                            let (idx, d) = self.min_dist(point, &l.children);
                            return (idx, d, queue)
                        },
                        &Node::Branch(ref b) => {
                            if point[b.dim] < b.split {
                                queue.push_front(&b.left);
                            } else {
                                queue.push_front(&b.right);
                            }
                        }
                    }
                }
            }
        }
    }

    fn search(&self, point: &Vec<f64>) -> usize {

        let (mut minidx, mut mind, mut queue) = self.search_leaf(point);
        while queue.len() > 0 {
            let current = queue.pop_front().unwrap();

            match current {
                &Node::Leaf(ref l) => {
                    println!("search leaf! {:?}", &l.children);
                    let (idx, d) = self.min_dist(&point, &l.children);
                    if d < mind {
                        mind = d;
                        minidx = idx;
                    }
                },
                &Node::Branch(ref b) => {
                    let d = b.dist(&point);
                    println!("search branch! {:?} {:?}", &b.dim, &b.split);
                    if d < mind {
                        queue.push_back(&b.left);
                        queue.push_back(&b.right)
                    }
                }
            }
        }
        minidx
    }

}

fn min(data: &Matrix<f64>) -> Vector<f64> {
    // ToDo: use rulinalg .min
    // https://github.com/AtheMathmo/rulinalg/pull/115
    let mut results = Vec::with_capacity(data.cols());
    for i in 0..data.cols() {
        results.push(data[[0, i]]);
    }
    for row in data.iter_rows() {
        for (i, v) in row.iter().enumerate() {
            let current = results[i];
            if current > *v {
                results[i] = *v;
            }
        }
    }
    Vector::new(results)
}

fn max(data: &Matrix<f64>) -> Vector<f64> {
    // ToDo: use rulinalg .max
    // https://github.com/AtheMathmo/rulinalg/pull/115
    let mut results = Vec::with_capacity(data.cols());
    for i in 0..data.cols() {
        results.push(data[[0, i]]);
    }
    for row in data.iter_rows() {
        for (i, v) in row.iter().enumerate() {
            let current = results[i];
            if current < *v {
                results[i] = *v;
            }
        }
    }
    Vector::new(results)
}

fn dist(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
    let d: f64 = v1.iter()
                   .zip(v2.iter())
                   .map(|(&x, &y)| (x - y) * (x - y))
                   .fold(0., |s, v| s + v);
    d.sqrt()
}

#[cfg(test)]
mod tests {

    use linalg::{Vector, Matrix};
    use super::{KDTree, min, max, dist};

    #[test]
    fn test_kdtree() {
        let m = Matrix::new(5, 2, vec![1., 2.,
                                       3., 2.5,
                                       2., 10.,
                                       3., 6.,
                                       1., 3.]);
        let mut tree = KDTree::new(m, 3);
        tree.build();

        let (idx, d, queue) = tree.search_leaf(&vec![2., 3.]);
        assert_eq!(idx, 3);

        let idx = tree.search(&vec![1.5, 3.]);
        assert_eq!(idx, 4);
    }

    #[test]
    fn test_min_max() {
        let data = Matrix::new(3, 2, vec![1., 2.,
                                          2., 4.,
                                          3., 1.]);
        assert_eq!(min(&data), Vector::new(vec![1., 1.]));
        assert_eq!(max(&data), Vector::new(vec![3., 4.]));
    }

    #[test]
    fn test_dist() {
        let v1 = vec![1., 2., 3.];
        let v2 = vec![1., 1., 1.];
        assert_eq!(dist(&v1, &v2), 5.0f64.sqrt());

        let v1 = vec![1., 2., 4.];
        let v2 = vec![2., 1., 1.];
        assert_eq!(dist(&v1, &v2), 11.0f64.sqrt());

        let v1 = vec![1., 3., 5.];
        let v2 = vec![2., 1., 1.];
        assert_eq!(dist(&v1, &v2), 21.0f64.sqrt());
    }
}
