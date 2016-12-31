//! Binary Tree implementations
use std::borrow::Borrow;
use std::collections::VecDeque;
use linalg::{Matrix, BaseMatrix, Vector};

use super::{KNearest, KNearestSearch, get_distances, dist};

/// Binary Tree
#[derive(Debug)]
pub struct BinaryTree<B: BinarySplit> {
    // Binary Tree leaf size
    leafsize: usize,
    // Search Data
    data: Option<Matrix<f64>>,
    // Binary Tree
    root: Option<Node<B>>
}

impl<B: BinarySplit> Default for BinaryTree<B> {
    fn default() -> Self {
        BinaryTree {
            leafsize: 30,
            data: None,
            root: None
        }
    }
}

/// Binary splittable
pub trait BinarySplit: Sized {

    /// Build branch from passed args
    fn build(data: &Matrix<f64>, remains: Vec<usize>,
             dim: usize, split: f64, min: Vector<f64>, max: Vector<f64>,
             left: Node<Self>, right: Node<Self>)
        -> Node<Self>;

    /// Return a tuple of left and right node. First node is likely to be
    /// closer to the point
    unsafe fn maybe_close<'s, 'p>(&'s self, point: &'p [f64])
        -> (&'s Node<Self>, &'s Node<Self>);

    /// Return distance between the point and myself
    fn dist(&self, point: &[f64]) -> f64;

    /// Return left node
    fn left(&self) -> &Node<Self>;
    /// Return right node
    fn right(&self) -> &Node<Self>;
}

/// KDTree Branch
#[derive(Debug)]
pub struct KDTreeBranch {
    /// dimension (column) to split
    dim: usize,
    /// split value
    split: f64,

    /// min and max of bounding box
    /// i.e. hyper-rectangle contained in the branch
    min: Vector<f64>,
    max: Vector<f64>,

    // link to left / right node
    // - left node contains rows which the column specified with
    //   ``dim`` is less than ``split`` value.
    // - right node contains greater than or equal to ``split`` value
    left: Box<Node<KDTreeBranch>>,
    right: Box<Node<KDTreeBranch>>,
}

/// BallTree Branch
#[derive(Debug)]
pub struct BallTreeBranch {
    /// dimension (column) to split
    dim: usize,
    /// split value
    split: f64,

    /// ball centroid and its radius
    center: Vector<f64>,
    radius: f64,

    // link to left / right node, see KDTreeBranch comment
    left: Box<Node<BallTreeBranch>>,
    right: Box<Node<BallTreeBranch>>,
}

/// KDTree implementation
pub type KDTree = BinaryTree<KDTreeBranch>;

/// BallTree implementation
pub type BallTree = BinaryTree<BallTreeBranch>;

impl BinarySplit for KDTreeBranch {

    fn build(_: &Matrix<f64>, _: Vec<usize>,
             dim: usize, split: f64, min: Vector<f64>, max: Vector<f64>,
             left: Node<Self>, right: Node<Self>) -> Node<Self> {

        let b = KDTreeBranch {
            dim: dim,
            split: split,
            min: min,
            max: max,
            left: Box::new(left),
            right: Box::new(right)
        };
        Node::Branch(b)
    }

    unsafe fn maybe_close<'s, 'p>(&'s self, point: &'p [f64])
        -> (&'s Node<Self>, &'s Node<Self>) {

        if *point.get_unchecked(self.dim) < self.split {
            (&self.left, &self.right)
        } else {
            (&self.right, &self.left)
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

    fn left(&self) -> &Node<Self> {
        self.left.borrow()
    }

    fn right(&self) -> &Node<Self> {
        self.right.borrow()
    }
}

impl BinarySplit for BallTreeBranch {

    fn build(data: &Matrix<f64>, remains: Vec<usize>,
             dim: usize, split: f64, _: Vector<f64>, _: Vector<f64>,
             left: Node<Self>, right: Node<Self>) -> Node<Self> {

        // temp, calculate centroid (mean)
        let mut center: Vec<f64> = vec![0.; data.cols()];
        for &i in remains.iter() {
            let row: Vec<f64> = data.select_rows(&[i]).into_vec();
            for (c, r) in center.iter_mut().zip(row.iter()) {
                *c = *c + r;
            }
        }
        let len = remains.len() as f64;
        for c in center.iter_mut() {
            *c = *c / len;
        }
        let mut radius = 0.;
        for &i in remains.iter() {
            let row: Vec<f64> = data.select_rows(&[i]).into_vec();
            let d = dist(&center, &row);
            if d > radius {
                radius = d;
            }
        }

        let b = BallTreeBranch {
            dim: dim,
            split: split,
            center: Vector::new(center),
            radius: radius,
            left: Box::new(left),
            right: Box::new(right)
        };
        Node::Branch(b)
    }

    unsafe fn maybe_close<'s, 'p>(&'s self, point: &'p [f64]) -> (&'s Node<Self>, &'s Node<Self>) {
        if *point.get_unchecked(self.dim) < self.split {
            (&self.left, &self.right)
        } else {
            (&self.right, &self.left)
        }
    }

    fn dist(&self, point: &[f64]) -> f64 {
        let d = dist(&self.center.data(), point);
        if d < self.radius {
            0.
        } else {
            d - self.radius
        }
    }

    fn left(&self) -> &Node<Self> {
        self.left.borrow()
    }

    fn right(&self) -> &Node<Self> {
        self.right.borrow()
    }
}

/// Binary Tree Node (either branch or leaf)
#[derive(Debug)]
pub enum Node<B: BinarySplit> {
    /// Binary Tree branch
    Branch(B),
    /// Binary Tree leaf
    Leaf(Leaf)
}

impl<B: BinarySplit> Node<B> {
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
    fn as_branch(&self) -> &B {
        match self {
            &Node::Branch(ref branch) => branch,
            _ => panic!("Node is not branch")
        }
    }
}

/// Binary Tree Leaf
#[derive(Debug)]
pub struct Leaf {
    children: Vec<usize>
}

impl Leaf {
    fn new(children: Vec<usize>) -> Self {
        Leaf {
            children: children
        }
    }
}

impl<B: BinarySplit> BinaryTree<B> {

    fn new(leafsize: usize) -> Self {
        BinaryTree {
            leafsize: leafsize,
            data: None,
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
    fn select_split(&self, data: &Matrix<f64>, mut remains: Vec<usize>,
                    mut dmin: Vector<f64>, mut dmax: Vector<f64>)
        -> (usize, f64, Vec<usize>, Vec<usize>, Vector<f64>, Vector<f64>){

        // avoid recursive call
        loop {
            // split columns which has the widest range
            let (dim, d) = (&dmax - &dmin).argmax();
            // ToDo: use unsafe get  (v0.4.0?)
            // https://github.com/AtheMathmo/rulinalg/pull/104
            let split = unsafe {
                dmin.data().get_unchecked(dim) + d / 2.0
            };

            // split remains
            let mut l_remains: Vec<usize> = Vec::with_capacity(remains.len());
            let mut r_remains: Vec<usize> = Vec::with_capacity(remains.len());
            unsafe {
                for r in remains {
                    if *data.get_unchecked([r, dim]) < split {
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

    /// find next binary split
    fn split(&self, data: &Matrix<f64>, remains: Vec<usize>,
             dmin: Vector<f64>, dmax: Vector<f64>) -> Node<B> {

        if remains.len() < self.leafsize {
            Node::Leaf(Leaf::new(remains))
        } else {

            // ToDo: avoid this clone
            let (dim, split, l_remains, r_remains, l_max, r_min) =
                self.select_split(data, remains.clone(), dmin.clone(), dmax.clone());

            let l_node = self.split(data, l_remains, dmin.clone(), l_max);
            let g_node = self.split(data, r_remains, r_min, dmax.clone());
            B::build(data, remains, dim, split, dmin, dmax, l_node, g_node)
        }
    }

    /// find leaf contains search point
    fn search_leaf<'s, 'p>(&'s self, point: &'p [f64], k: usize)
        -> (KNearest, VecDeque<&'s Node<B>>) {

        if let (&Some(ref root), &Some(ref data)) = (&self.root, &self.data) {

            let mut queue: VecDeque<&Node<B>> = VecDeque::new();
            queue.push_front(root);

            loop {
                // pop first element
                let current: &Node<B> = queue.pop_front().unwrap();
                match current {
                    &Node::Leaf(ref l) => {
                        let distances = get_distances(data, point, &l.children);
                        let kn = KNearest::new(k, l.children.clone(), distances);
                        return (kn, queue);
                    },
                    &Node::Branch(ref b) => {
                        // the current branch must contains target point.
                        // store the child branch which contains target point to
                        // the front, put other side on the back.
                        let (close, far) = unsafe {
                            b.maybe_close(point)
                        };
                        queue.push_front(close);
                        queue.push_back(far);
                    }
                }
            }
        } else {
            panic!("tree is not built")
        }
    }
}

/// Can search K-nearest items
impl<B: BinarySplit> KNearestSearch for BinaryTree<B> {

    /// build data structure for search optimization
    fn build(&mut self, data: Matrix<f64>) {
        let remains: Vec<usize> = (0..data.rows()).collect();
        let dmin = min(&data);
        let dmax = max(&data);
        self.root = Some(self.split(&data, remains, dmin, dmax));
        self.data = Some(data);
    }

    /// Serch k-nearest items close to the point
    fn search(&self, point: &[f64], k: usize) -> (Vec<usize>, Vec<f64>) {
        // ToDo: should return Error
        if let &Some(ref data) = &self.data {
            let (mut query, mut queue) = self.search_leaf(point, k);
            while queue.len() > 0 {
                let current = queue.pop_front().unwrap();

                match current {
                    &Node::Leaf(ref l) => {
                        let distances = get_distances(data, point, &l.children);
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
                            queue.push_back(b.left());
                            queue.push_back(b.right());
                        }
                    }
                }
            }
            query.get_results()
        } else {
            panic!("error")
        }
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

#[cfg(test)]
mod tests {

    use linalg::{Vector, Matrix, BaseMatrix};
    use super::super::KNearestSearch;
    use super::{KDTree, BallTree, min, max};

    #[test]
    fn test_kdtree_construct() {
        let m = Matrix::new(5, 2, vec![1., 2.,
                                       8., 0.,
                                       6., 10.,
                                       3., 6.,
                                       0., 3.]);
        let mut tree = KDTree::new(3);
        tree.build(m);

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
        let mut tree = KDTree::new(3);
        tree.build(m);

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

        let mut tree = KDTree::new(10);
        tree.build(data);

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

        let mut tree = KDTree::new(10);
        tree.build(data.clone());

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
        let mut tree = KDTree::new(3);
        tree.build(m);

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
        let mut tree = KDTree::new(3);
        tree.build(m);

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
    fn test_balltree_construct() {
        let m = Matrix::new(5, 2, vec![1., 2.,
                                       8., 0.,
                                       6., 10.,
                                       3., 6.,
                                       0., 3.]);
        let mut tree = BallTree::new(3);
        tree.build(m);

        // split to [0, 1, 4] and [2, 3] with columns #1
        let root = tree.root.unwrap();
        let b = root.as_branch();
        assert_eq!(b.dim, 1);
        assert_eq!(b.split, 5.);
        assert_eq!(b.center, Vector::new(vec![18. / 5., 21. / 5.]));
        // distance between the center and [2]
        let exp_d: f64 = (6. - 3.6) * (6. - 3.6) + (10. - 4.2) * (10. - 4.2);
        assert_eq!(b.radius, exp_d.sqrt());

        // split to [0, 4] and [1] with columns #0
        let bl = b.left.as_branch();
        let br = b.right.as_leaf();
        assert_eq!(bl.dim, 0);
        assert_eq!(bl.split, 4.);
        assert_eq!(bl.center, Vector::new(vec![3., 5. / 3.]));
        // distance between the center and [1]
        let exp_d: f64 = (3. - 8.) * (3. - 8.) + 5. / 3. * 5. / 3.;
        assert_eq!(bl.radius, exp_d.sqrt());
        assert_eq!(br.children, vec![2, 3]);

        let bll = bl.left.as_leaf();
        let blr = bl.right.as_leaf();
        assert_eq!(bll.children, vec![0, 4]);
        assert_eq!(blr.children, vec![1]);
    }

    #[test]
    fn test_balltree_search() {
        let m = Matrix::new(5, 2, vec![1., 2.,
                                       8., 0.,
                                       6., 10.,
                                       3., 6.,
                                       0., 3.]);
        let mut tree = BallTree::new(3);
        tree.build(m);

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
    fn test_balltree_search_iris() {
        use super::super::super::super::datasets::iris;

        let dataset = iris::load();
        let data = dataset.data();

        let mut tree = BallTree::new(10);
        tree.build(data.clone());

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
    fn test_balltree_dim_selection_biased() {
        let m = Matrix::new(5, 2, vec![1., 0.,
                                       3., 0.,
                                       2., 20.,
                                       3., 0.,
                                       1., 0.]);
        let mut tree = BallTree::new(3);
        tree.build(m);

        // split to [0, 1, 3, 4] and [2] with columns #1
        let root = tree.root.unwrap();
        let b = root.as_branch();
        assert_eq!(b.dim, 1);
        assert_eq!(b.split, 10.);
        assert_eq!(b.center, Vector::new(vec![10. / 5., 20. / 5.]));
        // distance between the center and [2]
        let exp_d: f64 = (2. - 2.) * (2. - 2.) + (4. - 20.) * (4. - 20.);
        assert_eq!(b.radius, exp_d.sqrt());

        // split to [0, 4] and [1, 3] with columns #0
        let bl = b.left.as_branch();
        assert_eq!(bl.dim, 0);
        assert_eq!(bl.split, 2.);
        assert_eq!(bl.center, Vector::new(vec![8. / 4., 0.]));
        // distance between the center and [0]
        let exp_d: f64 = (2. - 1.) * (2. - 1.);
        assert_eq!(bl.radius, exp_d.sqrt());

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
