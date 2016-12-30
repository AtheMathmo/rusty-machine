//! KDTree implementations
use std::collections::VecDeque;
use linalg::{Matrix, BaseMatrix, Vector};

use super::{KNearest, KNearestSearch};

/// KDTree Node (either branch or leaf)
#[derive(Debug)]
enum Node {
    Branch(Branch),
    Leaf(Leaf)
}

impl Node {
    // return my leaf reference, for testing purpose
    #[allow(dead_code)]
    fn as_leaf(&self) -> &Leaf {
        match self {
            &Node::Leaf(ref leaf) => leaf,
            _ => panic!("Node is not leaf")
        }
    }

    // return my branch reference, for testing purpose
    #[allow(dead_code)]
    fn as_branch(&self) -> &Branch {
        match self {
            &Node::Branch(ref branch) => branch,
            _ => panic!("Node is not branch")
        }
    }
}

/// KDTree Branch
#[derive(Debug)]
struct Branch {
    /// dimension (column) to split
    dim: usize,
    /// split value
    split: f64,

    /// min and max of bounding box
    /// i.e. hyper-rectangle contained in the branch
    min: Vector<f64>,
    max: Vector<f64>,

    /// left node
    left: Box<Node>,
    /// right node
    right: Box<Node>,
}

impl Branch {
    fn new(dim: usize, split: f64,
           min: Vector<f64>, max: Vector<f64>,
           left: Node, right: Node) -> Self {

        Branch {
            dim: dim,
            split: split,

            min: min,
            max: max,

            // link to left / right node
            // - left node contains rows which the column specified with
            //   ``dim`` is less than ``split`` value.
            // - right node contains greater than or equal to ``split`` value
            left: Box::new(left),
            right: Box::new(right)
        }
    }

    fn dist(&self, point: &[f64]) -> f64 {
        let mut d = 0.;
        for ((&p, &mi), &ma) in point.iter()
                                     .zip(self.min.iter())
                                     .zip(self.max.iter()) {
            if p < mi {
                d += (mi - p) * (mi - p);
            } else if ma < p {
                d += (ma - p) * (ma - p);
            }
            // otherwise included in the hyper-rectangle
        }
        d.sqrt()
    }
}

/// KDTree Leaf
#[derive(Debug)]
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

/// KDTree
#[derive(Debug)]
pub struct KDTree {
    data: Matrix<f64>,

    // KDTree leaf size
    leafsize: usize,
    // KDTree
    root: Option<Node>
}

impl KDTree {

    /// initialize KDTree, must call .build to actually built tree
    pub fn new(data: Matrix<f64>, leafsize: usize) -> Self {
        KDTree {
            data: data,
            leafsize: leafsize,

            root: None
        }
    }

    /// Select next split dimension and value. Returns tuple with 6 elements
    /// - split dim
    /// - split value
    /// - remains for left node
    /// - remains for right node
    /// - updated max for left node
    /// - updated min for right node
    fn select_split(&self, mut remains: Vec<usize>, mut dmin: Vector<f64>, mut dmax: Vector<f64>)
        -> (usize, f64, Vec<usize>, Vec<usize>, Vector<f64>, Vector<f64>){

        // avoid recursive call
        loop {
            // split columns which has the widest range
            let (dim, d) = (&dmax - &dmin).argmax();
            // ToDo: use unsafe get  (v0.4.0?)
            // https://github.com/AtheMathmo/rulinalg/pull/104
            let split = dmin[dim] + d / 2.0;

            // split remains
            let mut l_remains: Vec<usize> = Vec::with_capacity(remains.len());
            let mut r_remains: Vec<usize> = Vec::with_capacity(remains.len());
            unsafe {
                for r in remains {
                    if *self.data.get_unchecked([r, dim]) < split {
                        l_remains.push(r);
                    } else {
                        r_remains.push(r);
                    }
                }
            }
            r_remains.shrink_to_fit();
            l_remains.shrink_to_fit();

            if l_remains.len() == 0 {
                // all rows are in r_remains. re-split r_remains
                remains = r_remains;
                dmin[dim] = split;
            } else if r_remains.len() == 0 {
                // all rows are in l_remains. re-split l_remains
                remains = l_remains;
                dmax[dim] = split;
            } else {
                // new hyper-rectangle's min / max
                let mut l_max = dmax.clone();
                // ToDo: use unsafe mut (v0.4.0?)
                // https://github.com/AtheMathmo/rulinalg/pull/104
                l_max[dim] = split;

                let mut r_min = dmin.clone();
                r_min[dim] = split;

                return (dim, split, l_remains, r_remains, l_max, r_min);
            }
        };
    }

    fn split_one(&self, remains: Vec<usize>, dmin: Vector<f64>, dmax: Vector<f64>) -> Node {
        if remains.len() < self.leafsize {
            Node::Leaf(Leaf::new(remains))
        } else {
            let (dim, split, l_remains, r_remains, l_max, r_min) = self.select_split(remains, dmin.clone(), dmax.clone());
            let l_node = self.split_one(l_remains, dmin.clone(), l_max);
            let g_node = self.split_one(r_remains, r_min, dmax.clone());
            Node::Branch(Branch::new(dim, split, dmin, dmax,
                                     l_node, g_node))
        }
    }

    /// return distances between given point and data specified with row ids
    fn get_distances(&self, point: &[f64], ids: &[usize]) -> Vec<f64> {
        assert!(ids.len() > 0, "target ids is empty");

        let mut distances: Vec<f64> = Vec::with_capacity(ids.len());
        for id in ids.iter() {
            // ToDo: use .row(*id)
            let row: Vec<f64> = self.data.select_rows(&[*id]).into_vec();
            // let row: Vec<f64> = self.data.row(*id).into_vec();
            let d = dist(point, &row);
            distances.push(d);
        }
        distances
    }

    fn search_leaf<'s, 'p>(&'s self, point: &'p [f64], k: usize)
        -> (KNearest, VecDeque<&'s Node>) {

        match self.root {
            None => panic!("tree is not built"),
            Some(ref root) => {

                let mut queue: VecDeque<&Node> = VecDeque::new();
                queue.push_front(root);

                loop {
                    // pop first element
                    let current: &Node = queue.pop_front().unwrap();
                    match current {
                        &Node::Leaf(ref l) => {
                            let distances = self.get_distances(point, &l.children);
                            let kn = KNearest::new(k, l.children.clone(), distances);
                            return (kn, queue);
                        },
                        &Node::Branch(ref b) => {
                            // the current branch must contains target point.
                            // store the child branch which contains target point to
                            // the front, put other side on the back.
                            if point[b.dim] < b.split {
                                queue.push_front(&b.left);
                                queue.push_back(&b.right)
                            } else {
                                queue.push_back(&b.left);
                                queue.push_front(&b.right);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Can search K-nearest items
impl KNearestSearch for KDTree {

    /// build data structure for search optimization, used in KDTree
    /// build KDTree using its data
    fn build(&mut self) {
        let remains: Vec<usize> = (0..self.data.rows()).collect();
        let dmin = min(&self.data);
        let dmax = max(&self.data);
        self.root = Some(self.split_one(remains, dmin, dmax));
    }

    /// Serch k-nearest items close to the point
    fn search(&self, point: &[f64], k: usize) -> (Vec<usize>, Vec<f64>) {
        let (mut query, mut queue) = self.search_leaf(point, k);
        while queue.len() > 0 {
            let current = queue.pop_front().unwrap();

            match current {
                &Node::Leaf(ref l) => {
                    let distances = self.get_distances(point, &l.children);
                    let mut current_dist = query.dist();

                    for (&i, d) in l.children.iter().zip(distances.into_iter()) {
                        if d < current_dist {
                            current_dist = query.add(i, d);
                        }
                    }
                },
                &Node::Branch(ref b) => {
                    let d = b.dist(&point);
                    if d < query.dist() {
                        queue.push_back(&b.left);
                        queue.push_back(&b.right)
                    }
                }
            }
        }
        query.get_results()
    }
}

/// min
fn min(data: &Matrix<f64>) -> Vector<f64> {
    // ToDo: use rulinalg .min (v0.4.1?)
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

/// max
fn max(data: &Matrix<f64>) -> Vector<f64> {
    // ToDo: use rulinalg .max (v0.4.1?)
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

fn dist(v1: &[f64], v2: &[f64]) -> f64 {
    // ToDo: use metrics
    let d: f64 = v1.iter()
                   .zip(v2.iter())
                   .map(|(&x, &y)| (x - y) * (x - y))
                   .fold(0., |s, v| s + v);
    d.sqrt()
}

#[cfg(test)]
mod tests {

    use linalg::{Vector, Matrix, BaseMatrix};
    use super::super::KNearestSearch;
    use super::{KDTree, min, max};

    #[test]
    fn test_kdtree_construct() {
        let m = Matrix::new(5, 2, vec![1., 2.,
                                       8., 0.,
                                       6., 10.,
                                       3., 6.,
                                       0., 3.]);
        let mut tree = KDTree::new(m, 3);
        tree.build();

        // split to [0, 1, 4] and [2, 3] with columns #1
        let root = tree.root.unwrap();
        let b = root.as_branch();
        assert_eq!(b.dim, 1);
        assert_eq!(b.split, 5.);
        assert_eq!(b.min, Vector::new(vec![0., 0.]));
        assert_eq!(b.max, Vector::new(vec![8., 10.]));

        // split to [0, 4] and [1] with columns #0
        let bl = b.left.as_branch();
        let br = b.right.as_leaf();
        assert_eq!(bl.dim, 0);
        assert_eq!(bl.split, 4.);
        assert_eq!(bl.min, Vector::new(vec![0., 0.]));
        assert_eq!(bl.max, Vector::new(vec![8., 5.]));
        assert_eq!(br.children, vec![2, 3]);

        let bll = bl.left.as_leaf();
        let blr = bl.right.as_leaf();
        assert_eq!(bll.children, vec![0, 4]);
        assert_eq!(blr.children, vec![1]);
    }

    #[test]
    fn test_kdtree_search() {
        let m = Matrix::new(5, 2, vec![1., 2.,
                                       8., 0.,
                                       6., 10.,
                                       3., 6.,
                                       0., 3.]);
        let mut tree = KDTree::new(m, 3);
        tree.build();

        // search first leaf
        let (kn, _) = tree.search_leaf(&vec![3., 4.9], 1);
        assert_eq!(kn.pairs, vec![(0, (2.0f64 * 2.0f64 + 2.9f64 * 2.9f64).sqrt())]);

        // search tree
        let (ind, dist) = tree.search(&vec![3., 4.9], 1);
        assert_eq!(ind, vec![3]);
        assert_eq!(dist, vec![1.0999999999999996]);

        let (ind, dist) = tree.search(&vec![3., 4.9], 3);
        assert_eq!(ind, vec![3, 0, 4]);
        assert_eq!(dist, vec![1.0999999999999996, 3.5227829907617076, 3.551056180912941]);

        // search first leaf
        let (kn, _) = tree.search_leaf(&vec![3., 4.9], 2);
        assert_eq!(kn.pairs, vec![(0, (2.0f64 * 2.0f64 + 2.9f64 * 2.9f64).sqrt()),
                                  (4, (3.0f64 * 3.0f64 + (4.9f64 - 3.0f64) * (4.9f64 - 3.0f64)).sqrt())]);
        // search tree
        let (ind, dist) = tree.search(&vec![3., 4.9], 2);
        assert_eq!(ind, vec![3, 0]);
        assert_eq!(dist, vec![1.0999999999999996, 3.5227829907617076]);
    }

    #[cfg(feature = "datasets")]
    #[test]
    fn test_kdtree_search_iris_2cols() {
        use super::super::super::super::datasets::iris;

        let dataset = iris::load();
        let data = dataset.data().select_cols(&[0, 1]);

        let mut tree = KDTree::new(data, 10);
        tree.build();

        // search tree
        let (ind, dist) = tree.search(&vec![5.8, 3.6], 4);
        assert_eq!(ind, vec![18, 85, 36, 14]);
        assert_eq!(dist, vec![0.22360679774997858, 0.2828427124746193, 0.31622776601683783, 0.3999999999999999]);

        let (ind, dist) = tree.search(&vec![7.0, 2.6], 4);
        assert_eq!(ind, vec![76, 108, 102, 107]);
        assert_eq!(dist, vec![0.28284271247461895, 0.31622776601683783, 0.41231056256176585, 0.4242640687119283]);
    }

    #[cfg(feature = "datasets")]
    #[test]
    fn test_kdtree_search_iris() {
        use super::super::super::super::datasets::iris;

        let dataset = iris::load();
        let data = dataset.data();

        let mut tree = KDTree::new(data.clone(), 10);
        tree.build();

        // search tree
        let (ind, dist) = tree.search(&vec![5.8, 3.1, 3.8, 1.2], 8);
        assert_eq!(ind, vec![64, 88, 82, 95, 99, 96, 71, 61]);
        assert_eq!(dist, vec![0.360555127546399, 0.3872983346207417, 0.41231056256176596,
                              0.4242640687119288, 0.4472135954999579, 0.4690415759823433,
                              0.4795831523312721, 0.5196152422706636]);

        let (ind, dist) = tree.search(&vec![6.5, 3.5, 3.2, 1.3], 10);
        assert_eq!(ind, vec![71, 64, 74, 82, 79, 61, 65, 97, 75, 51]);
        assert_eq!(dist, vec![1.1357816691600549, 1.1532562594670799, 1.2569805089976533,
                              1.2767145334803702, 1.2767145334803702, 1.284523257866513,
                              1.2845232578665131, 1.2884098726725122, 1.3076696830622023,
                              1.352774925846868]);
    }

    #[test]
    fn test_kdtree_dim_selection() {
        let m = Matrix::new(5, 2, vec![1., 2.,
                                       3., 0.,
                                       2., 10.,
                                       3., 6.,
                                       1., 3.]);
        let mut tree = KDTree::new(m, 3);
        tree.build();

        // split to [0, 1, 4] and [2, 3] with columns #1
        let root = tree.root.unwrap();
        let b = root.as_branch();
        assert_eq!(b.dim, 1);
        assert_eq!(b.split, 5.);
        assert_eq!(b.min, Vector::new(vec![1., 0.]));
        assert_eq!(b.max, Vector::new(vec![3., 10.]));

        // split to [0, 1] and [4] with columns #1
        let bl = b.left.as_branch();
        assert_eq!(bl.dim, 1);
        assert_eq!(bl.split, 2.5);
        assert_eq!(bl.min, Vector::new(vec![1., 0.]));
        assert_eq!(bl.max, Vector::new(vec![3., 5.]));

        let br = b.right.as_leaf();
        assert_eq!(br.children, vec![2, 3]);

        let bll = bl.left.as_leaf();
        let blr = bl.right.as_leaf();
        assert_eq!(bll.children, vec![0, 1]);
        assert_eq!(blr.children, vec![4]);
    }

    #[test]
    fn test_kdtree_dim_selection_biased() {
        let m = Matrix::new(5, 2, vec![1., 0.,
                                       3., 0.,
                                       2., 20.,
                                       3., 0.,
                                       1., 0.]);
        let mut tree = KDTree::new(m, 3);
        tree.build();

        // split to [0, 1, 3, 4] and [2] with columns #1
        let root = tree.root.unwrap();
        let b = root.as_branch();
        assert_eq!(b.dim, 1);
        assert_eq!(b.split, 10.);
        assert_eq!(b.min, Vector::new(vec![1., 0.]));
        assert_eq!(b.max, Vector::new(vec![3., 20.]));

        // split to [0, 4] and [1, 3] with columns #0
        let bl = b.left.as_branch();
        assert_eq!(bl.dim, 0);
        assert_eq!(bl.split, 2.);
        assert_eq!(bl.min, Vector::new(vec![1., 0.]));
        assert_eq!(bl.max, Vector::new(vec![3., 10.]));

        let br = b.right.as_leaf();
        assert_eq!(br.children, vec![2]);

        let bll = bl.left.as_leaf();
        let blr = bl.right.as_leaf();
        assert_eq!(bll.children, vec![0, 4]);
        assert_eq!(blr.children, vec![1, 3]);
    }

    #[test]
    fn test_min_max() {
        let data = Matrix::new(3, 2, vec![1., 2.,
                                          2., 4.,
                                          3., 1.]);
        assert_eq!(min(&data), Vector::new(vec![1., 1.]));
        assert_eq!(max(&data), Vector::new(vec![3., 4.]));
    }
}
