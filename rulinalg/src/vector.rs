//! The vector module.
//!
//! Currently contains all code
//! relating to the vector linear algebra struct.

use std::ops::{Mul, Add, Div, Sub, Index, Neg, MulAssign, DivAssign, SubAssign, AddAssign};
use libnum::{One, Zero, Float, FromPrimitive};
use std::cmp::PartialEq;
use Metric;
use utils;

/// The Vector struct.
///
/// Can be instantiated with any type.
#[derive(Debug, PartialEq)]
pub struct Vector<T> {
    size: usize,
    data: Vec<T>,
}

impl<T> Vector<T> {
    /// Constructor for Vector struct.
    ///
    /// Requires the vector data.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let vec = Vector::new(vec![1.0,2.0,3.0,4.0]);
    /// ```
    pub fn new<U: Into<Vec<T>>>(data: U) -> Vector<T> {
        let our_data = data.into();
        let size = our_data.len();

        Vector {
            size: size,
            data: our_data,
        }
    }

    /// Returns the size of the Vector.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns a non-mutable reference to the underlying data.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Returns a mutable slice of the underlying data.
    pub fn mut_data(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consumes the Vector and returns the Vec of data.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T: Clone> Clone for Vector<T> {
    /// Clones the Vector.
    fn clone(&self) -> Vector<T> {
        Vector {
            size: self.size,
            data: self.data.clone(),
        }
    }
}

impl<T: Copy> Vector<T> {
    /// Applies a function to each element in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    /// fn add_two(a: f64) -> f64 {
    ///     a + 2f64
    /// }
    ///
    /// let a = Vector::new(vec![0.;4]);
    ///
    /// let b = a.apply(&add_two);
    ///
    /// assert_eq!(b.into_vec(), vec![2.0; 4]);
    /// ```
    pub fn apply(mut self, f: &Fn(T) -> T) -> Vector<T> {
        for val in &mut self.data {
            *val = f(*val);
        }
        self
    }
}

impl<T: Copy + PartialOrd> Vector<T> {
    /// Find the argmax of the Vector.
    ///
    /// Returns the index of the largest value in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let a = Vector::new(vec![1.0,2.0,0.0,5.0]);
    /// let b = a.argmax();
    /// assert_eq!(b.0, 3);
    /// assert_eq!(b.1, 5.0);
    /// ```
    pub fn argmax(&self) -> (usize, T) {
        utils::argmax(&self.data)
    }

    /// Find the argmin of the Vector.
    ///
    /// Returns the index of the smallest value in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let a = Vector::new(vec![1.0,2.0,0.0,5.0]);
    /// let b = a.argmin();
    /// assert_eq!(b.0, 2);
    /// assert_eq!(b.1, 0.0);
    /// ```
    pub fn argmin(&self) -> (usize, T) {
        utils::argmin(&self.data)
    }

    /// Select elements from the Vector and form a new Vector from them.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let a = Vector::new(vec![1.0,2.0,3.0,4.0,5.0]);
    ///
    /// let a_lower = a.select(&[2,3,4]);
    ///
    /// // Prints [3,4,5]
    /// println!("{:?}", a_lower.data());
    /// ```
    pub fn select(&self, idxs: &[usize]) -> Vector<T> {
        let mut new_data = Vec::with_capacity(idxs.len());

        for idx in idxs.into_iter() {
            new_data.push(self[*idx]);
        }

        Vector::new(new_data)
    }
}

impl<T: Clone + Zero> Vector<T> {
    /// Constructs Vector of all zeros.
    ///
    /// Requires the size of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let vec = Vector::<f64>::zeros(10);
    /// ```
    pub fn zeros(size: usize) -> Vector<T> {
        Vector {
            size: size,
            data: vec![T::zero(); size],
        }
    }
}

impl<T: Clone + One> Vector<T> {
    /// Constructs Vector of all ones.
    ///
    /// Requires the size of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let vec = Vector::<f64>::ones(10);
    /// ```
    pub fn ones(size: usize) -> Vector<T> {
        Vector {
            size: size,
            data: vec![T::one(); size],
        }
    }
}

impl<T: Copy + Zero + Mul<T, Output = T> + Add<T, Output = T>> Vector<T> {
    /// Compute dot product with specified Vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let a = Vector::new(vec![1.0,2.0,3.0,4.0]);
    /// let b = Vector::new(vec![2.0; 4]);
    ///
    /// let c = a.dot(&b);
    /// assert_eq!(c, 20.0);
    /// ```
    pub fn dot(&self, v: &Vector<T>) -> T {
        utils::dot(&self.data, &v.data)
    }
}

impl<T: Copy + Zero + Add<T, Output = T>> Vector<T> {
    /// The sum of the vector.
    ///
    /// Returns the sum of all elements in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let a = Vector::new(vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = a.sum();
    /// assert_eq!(c, 10.0);
    /// ```
    pub fn sum(&self) -> T {
        utils::unrolled_sum(&self.data[..])
    }
}

impl<T: Copy + Mul<T, Output = T>> Vector<T> {
    /// The elementwise product of two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let a = Vector::new(vec![1.0,2.0,3.0,4.0]);
    /// let b = Vector::new(vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = &a.elemul(&b);
    /// assert_eq!(*c.data(), vec![1.0, 4.0, 9.0, 16.0]);
    /// ```
    pub fn elemul(&self, v: &Vector<T>) -> Vector<T> {
        assert_eq!(self.size, v.size);
        Vector::new(utils::ele_mul(&self.data, &v.data))
    }
}

impl<T: Copy + Div<T, Output = T>> Vector<T> {
    /// The elementwise division of two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let a = Vector::new(vec![1.0,2.0,3.0,4.0]);
    /// let b = Vector::new(vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = &a.elediv(&b);
    /// assert_eq!(*c.data(), vec![1.0; 4]);
    /// ```
    pub fn elediv(&self, v: &Vector<T>) -> Vector<T> {
        assert_eq!(self.size, v.size);
        Vector::new(utils::ele_div(&self.data, &v.data))
    }
}

impl<T: Float + FromPrimitive> Vector<T> {
    /// The mean of the vector.
    ///
    /// Returns the arithmetic mean of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let a = Vector::<f32>::new(vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = a.mean();
    /// assert_eq!(c, 2.5);
    /// ```
    pub fn mean(&self) -> T {
        let sum = self.sum();
        sum / FromPrimitive::from_usize(self.size()).unwrap()
    }

    /// The variance of the vector.
    ///
    /// Returns the unbiased sample variance of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    ///
    /// let a = Vector::<f32>::new(vec![1.0,2.0,3.0,4.0]);
    ///
    /// let c = a.variance();
    /// assert_eq!(c, 5.0/3.0);
    /// ```
    pub fn variance(&self) -> T {
        let m = self.mean();
        let mut var = T::zero();

        for u in &self.data {
            var = var + (*u - m) * (*u - m);
        }

        var / FromPrimitive::from_usize(self.size() - 1).unwrap()
    }
}

/// Multiplies vector by scalar.
impl<T: Copy + Mul<T, Output = T>> Mul<T> for Vector<T> {
    type Output = Vector<T>;

    fn mul(self, f: T) -> Vector<T> {
        self * &f
    }
}

/// Multiplies vector by scalar.
impl<'a, T: Copy + Mul<T, Output = T>> Mul<T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn mul(self, f: T) -> Vector<T> {
        self * (&f)
    }
}

/// Multiplies vector by scalar.
impl<'a, T: Copy + Mul<T, Output = T>> Mul<&'a T> for Vector<T> {
    type Output = Vector<T>;

    fn mul(mut self, f: &T) -> Vector<T> {
        for val in &mut self.data {
            *val = *val * *f;
        }

        self
    }
}

/// Multiplies vector by scalar.
impl<'a, 'b, T: Copy + Mul<T, Output = T>> Mul<&'b T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn mul(self, f: &T) -> Vector<T> {
        let new_data = self.data.iter().map(|v| (*v) * (*f)).collect();

        Vector {
            size: self.size,
            data: new_data,
        }
    }
}

/// Divides vector by scalar.
impl<T: Copy + Zero + PartialEq + Div<T, Output = T>> Div<T> for Vector<T> {
    type Output = Vector<T>;

    fn div(self, f: T) -> Vector<T> {
        self / &f
    }
}

/// Divides vector by scalar.
impl<'a, T: Copy + Zero + PartialEq + Div<T, Output = T>> Div<T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn div(self, f: T) -> Vector<T> {
        self / &f
    }
}

/// Divides vector by scalar.
impl<'a, T: Copy + Zero + PartialEq + Div<T, Output = T>> Div<&'a T> for Vector<T> {
    type Output = Vector<T>;

    fn div(mut self, f: &T) -> Vector<T> {
        assert!(*f != T::zero());

        for val in &mut self.data {
            *val = *val / *f;
        }

        self
    }
}

/// Divides vector by scalar.
impl<'a, 'b, T: Copy + Zero + PartialEq + Div<T, Output = T>> Div<&'b T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn div(self, f: &T) -> Vector<T> {
        assert!(*f != T::zero());
        let new_data = self.data.iter().map(|v| *v / *f).collect();

        Vector {
            size: self.size,
            data: new_data,
        }
    }
}

/// Adds scalar to vector.
impl<T: Copy + Add<T, Output = T>> Add<T> for Vector<T> {
    type Output = Vector<T>;

    fn add(self, f: T) -> Vector<T> {
        self + &f
    }
}

/// Adds scalar to vector.
impl<'a, T: Copy + Add<T, Output = T>> Add<T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn add(self, f: T) -> Vector<T> {
        self + &f
    }
}

/// Adds scalar to vector.
impl<'a, T: Copy + Add<T, Output = T>> Add<&'a T> for Vector<T> {
    type Output = Vector<T>;

    fn add(mut self, f: &T) -> Vector<T> {
        for val in &mut self.data {
            *val = *val + *f;
        }

        self
    }
}

/// Adds scalar to vector.
impl<'a, 'b, T: Copy + Add<T, Output = T>> Add<&'b T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn add(self, f: &T) -> Vector<T> {
        let new_data = self.data.iter().map(|v| *v + *f).collect();

        Vector {
            size: self.size,
            data: new_data,
        }
    }
}

/// Adds vector to vector.
impl<T: Copy + Add<T, Output = T>> Add<Vector<T>> for Vector<T> {
    type Output = Vector<T>;

    fn add(self, v: Vector<T>) -> Vector<T> {
        self + &v
    }
}

/// Adds vector to vector.
impl<'a, T: Copy + Add<T, Output = T>> Add<Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;

    fn add(self, v: Vector<T>) -> Vector<T> {
        v + self
    }
}

/// Adds vector to vector.
impl<'a, T: Copy + Add<T, Output = T>> Add<&'a Vector<T>> for Vector<T> {
    type Output = Vector<T>;

    fn add(mut self, v: &Vector<T>) -> Vector<T> {
        utils::in_place_vec_bin_op(&mut self.data, &v.data, |x, &y| *x = *x + y);

        self
    }
}

/// Adds vector to vector.
impl<'a, 'b, T: Copy + Add<T, Output = T>> Add<&'b Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;

    fn add(self, v: &Vector<T>) -> Vector<T> {
        assert!(self.size == v.size);

        let new_data = utils::vec_sum(&self.data, &v.data);

        Vector {
            size: self.size,
            data: new_data,
        }
    }
}

/// Subtracts scalar from vector.
impl<T: Copy + Sub<T, Output = T>> Sub<T> for Vector<T> {
    type Output = Vector<T>;

    fn sub(self, f: T) -> Vector<T> {
        self - &f
    }
}

/// Subtracts scalar from vector.
impl<'a, T: Copy + Sub<T, Output = T>> Sub<T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn sub(self, f: T) -> Vector<T> {
        self - &f
    }
}

/// Subtracts scalar from vector.
impl<'a, T: Copy + Sub<T, Output = T>> Sub<&'a T> for Vector<T> {
    type Output = Vector<T>;

    fn sub(mut self, f: &T) -> Vector<T> {
        for val in &mut self.data {
            *val = *val - *f;
        }

        self
    }
}

/// Subtracts scalar from vector.
impl<'a, 'b, T: Copy + Sub<T, Output = T>> Sub<&'b T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn sub(self, f: &T) -> Vector<T> {
        let new_data = self.data.iter().map(|v| *v - *f).collect();

        Vector {
            size: self.size,
            data: new_data,
        }
    }
}

/// Subtracts vector from vector.
impl<T: Copy + Sub<T, Output = T>> Sub<Vector<T>> for Vector<T> {
    type Output = Vector<T>;

    fn sub(self, v: Vector<T>) -> Vector<T> {
        self - &v
    }
}

/// Subtracts vector from vector.
impl<'a, T: Copy + Sub<T, Output = T>> Sub<Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;

    fn sub(self, mut v: Vector<T>) -> Vector<T> {
        utils::in_place_vec_bin_op(&mut v.data, &self.data, |x, &y| *x = y - *x);

        v
    }
}

/// Subtracts vector from vector.
impl<'a, T: Copy + Sub<T, Output = T>> Sub<&'a Vector<T>> for Vector<T> {
    type Output = Vector<T>;

    fn sub(mut self, v: &Vector<T>) -> Vector<T> {
        utils::in_place_vec_bin_op(&mut self.data, &v.data, |x, &y| *x = *x - y);

        self
    }
}

/// Subtracts vector from vector.
impl<'a, 'b, T: Copy + Sub<T, Output = T>> Sub<&'b Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;

    fn sub(self, v: &Vector<T>) -> Vector<T> {
        assert!(self.size == v.size);

        let new_data = utils::vec_sub(&self.data, &v.data);

        Vector {
            size: self.size,
            data: new_data,
        }
    }
}

/// Gets negative of vector.
impl<T: Neg<Output = T> + Copy> Neg for Vector<T> {
    type Output = Vector<T>;

    fn neg(mut self) -> Vector<T> {
        for val in &mut self.data {
            *val = -*val;
        }

        self
    }
}

/// Gets negative of vector.
impl<'a, T: Neg<Output = T> + Copy> Neg for &'a Vector<T> {
    type Output = Vector<T>;

    fn neg(self) -> Vector<T> {
        let new_data = self.data.iter().map(|v| -*v).collect::<Vec<_>>();

        Vector::new(new_data)
    }
}

/// Indexes vector.
impl<T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, idx: usize) -> &T {
        assert!(idx < self.size);
        unsafe { self.data.get_unchecked(idx) }
    }
}

impl<T: Float> Metric<T> for Vector<T> {
    /// Compute euclidean norm for vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use rulinalg::vector::Vector;
    /// use rulinalg::Metric;
    ///
    /// let a = Vector::new(vec![3.0,4.0]);
    /// let c = a.norm();
    ///
    /// assert_eq!(c, 5.0);
    /// ```
    fn norm(&self) -> T {
        let mut s = T::zero();

        for u in &self.data {
            s = s + (*u) * (*u);
        }

        s.sqrt()
    }
}

macro_rules! impl_op_assign_vec_scalar (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (

/// Performs
#[doc=$doc]
/// assignment between a vector and a scalar.
impl<T : Copy + $trt<T, Output=T>> $assign_trt<T> for Vector<T> {
    fn $op_assign(&mut self, _rhs: T) {
        for x in &mut self.data {
            *x = (*x).$op(_rhs)
        }
    }
}

/// Performs
#[doc=$doc]
/// assignment between a vector and a scalar.
impl<'a, T : Copy + $trt<T, Output=T>> $assign_trt<&'a T> for Vector<T> {
    fn $op_assign(&mut self, _rhs: &T) {
        for x in &mut self.data {
            *x = (*x).$op(*_rhs)
        }
    }
}
    );
);

impl_op_assign_vec_scalar!(AddAssign, Add, add, add_assign, "addition");
impl_op_assign_vec_scalar!(SubAssign, Sub, sub, sub_assign, "subtraction");
impl_op_assign_vec_scalar!(DivAssign, Div, div, div_assign, "division");
impl_op_assign_vec_scalar!(MulAssign, Mul, mul, mul_assign, "multiplication");

macro_rules! impl_op_assign_vec (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// assignment between two vectors.
impl<T : Copy + $trt<T, Output=T>> $assign_trt<Vector<T>> for Vector<T> {
    fn $op_assign(&mut self, _rhs: Vector<T>) {
        utils::in_place_vec_bin_op(&mut self.data, &_rhs.data, |x, &y| {*x = (*x).$op(y) });
    }
}

/// Performs elementwise
#[doc=$doc]
/// assignment between two vectors.
impl<'a, T : Copy + $trt<T, Output=T>> $assign_trt<&'a Vector<T>> for Vector<T> {
    fn $op_assign(&mut self, _rhs: &Vector<T>) {
        utils::in_place_vec_bin_op(&mut self.data, &_rhs.data, |x, &y| {*x = (*x).$op(y) });
    }
}
    );
);

impl_op_assign_vec!(AddAssign, Add, add, add_assign, "addition");
impl_op_assign_vec!(SubAssign, Sub, sub, sub_assign, "subtraction");

#[cfg(test)]
mod tests {
    use super::Vector;
    use super::super::Metric;

    #[test]
    fn test_equality() {
        let v = Vector::new(vec![1, 2, 3, 4]);
        let v_redux = v.clone();
        assert_eq!(v, v_redux);
    }

    #[test]
    fn create_vector_new() {
        let a = Vector::new(vec![1.0; 12]);

        assert_eq!(a.size(), 12);

        for i in 0..12 {
            assert_eq!(a[i], 1.0);
        }
    }

    #[test]
    fn create_vector_new_from_slice() {
        let data_vec: Vec<u32> = vec![1, 2, 3];
        let data_slice: &[u32] = &data_vec[..];
        let from_vec = Vector::new(data_vec.clone());
        let from_slice = Vector::new(data_slice);
        assert_eq!(from_vec, from_slice);
    }

    #[test]
    fn create_vector_zeros() {
        let a = Vector::<f32>::zeros(7);

        assert_eq!(a.size(), 7);

        for i in 0..7 {
            assert_eq!(a[i], 0.0);
        }
    }

    #[test]
    fn vector_dot_product() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Vector::new(vec![3.0; 6]);

        let c = a.dot(&b);

        assert_eq!(c, 63.0);
    }

    #[test]
    fn vector_f32_mul() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = 3.0;

        // Allocating new memory
        let c = &a * &b;

        for i in 0..6 {
            assert_eq!(c[i], 3.0 * ((i + 1) as f32));
        }

        // Allocating new memory
        let c = &a * b;

        for i in 0..6 {
            assert_eq!(c[i], 3.0 * ((i + 1) as f32));
        }

        // Reusing memory
        let c = a.clone() * &b;

        for i in 0..6 {
            assert_eq!(c[i], 3.0 * ((i + 1) as f32));
        }

        // Reusing memory
        let c = a * b;

        for i in 0..6 {
            assert_eq!(c[i], 3.0 * ((i + 1) as f32));
        }
    }

    #[test]
    fn vector_f32_div() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = 3.0;

        // Allocating new memory
        let c = &a / &b;

        for i in 0..6 {
            assert_eq!(c[i], ((i + 1) as f32) / 3.0);
        }

        // Allocating new memory
        let c = &a / b;

        for i in 0..6 {
            assert_eq!(c[i], ((i + 1) as f32) / 3.0);
        }

        // Reusing memory
        let c = a.clone() / &b;

        for i in 0..6 {
            assert_eq!(c[i], ((i + 1) as f32) / 3.0);
        }

        // Reusing memory
        let c = a / b;

        for i in 0..6 {
            assert_eq!(c[i], ((i + 1) as f32) / 3.0);
        }
    }

    #[test]
    fn vector_add() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Vector::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        // Allocating new memory
        let c = &a + &b;

        for i in 0..6 {
            assert_eq!(c[i], ((2 * i + 3) as f32));
        }

        // Reusing memory
        let c = &a + b.clone();

        for i in 0..6 {
            assert_eq!(c[i], ((2 * i + 3) as f32));
        }

        // Reusing memory
        let c = a.clone() + &b;

        for i in 0..6 {
            assert_eq!(c[i], ((2 * i + 3) as f32));
        }

        // Reusing memory
        let c = a + b;

        for i in 0..6 {
            assert_eq!(c[i], ((2 * i + 3) as f32));
        }
    }

    #[test]
    fn vector_f32_add() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = 2.0;

        // Allocating new memory
        let c = &a + &b;

        for i in 0..6 {
            assert_eq!(c[i], ((i + 1) as f32) + 2.0);
        }

        // Allocating new memory
        let c = &a + b;

        for i in 0..6 {
            assert_eq!(c[i], ((i + 1) as f32) + 2.0);
        }

        // Reusing memory
        let c = a.clone() + &b;

        for i in 0..6 {
            assert_eq!(c[i], ((i + 1) as f32) + 2.0);
        }

        // Reusing memory
        let c = a + b;

        for i in 0..6 {
            assert_eq!(c[i], ((i + 1) as f32) + 2.0);
        }
    }

    #[test]
    fn vector_sub() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Vector::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

        // Allocating new memory
        let c = &a - &b;

        for i in 0..6 {
            assert_eq!(c[i], -1.0);
        }

        // Reusing memory
        let c = &a - b.clone();

        for i in 0..6 {
            assert_eq!(c[i], -1.0);
        }

        // Reusing memory
        let c = a.clone() - &b;

        for i in 0..6 {
            assert_eq!(c[i], -1.0);
        }

        // Reusing memory
        let c = a - b;

        for i in 0..6 {
            assert_eq!(c[i], -1.0);
        }
    }

    #[test]
    fn vector_f32_sub() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = 2.0;

        // Allocating new memory
        let c = &a - &b;

        for i in 0..6 {
            assert_eq!(c[i], ((i + 1) as f32) - 2.0);
        }

        // Allocating new memory
        let c = &a - b;

        for i in 0..6 {
            assert_eq!(c[i], ((i + 1) as f32) - 2.0);
        }

        // Reusing memory
        let c = a.clone() - &b;

        for i in 0..6 {
            assert_eq!(c[i], ((i + 1) as f32) - 2.0);
        }

        // Reusing memory
        let c = a - b;

        for i in 0..6 {
            assert_eq!(c[i], ((i + 1) as f32) - 2.0);
        }
    }

    #[test]
    fn vector_norm() {
        let a = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let b = a.norm();

        assert_eq!(b, (1. + 4. + 9. + 16. + 25. + 36. as f32).sqrt());
    }

    #[test]
    fn vector_add_assign() {
        let mut a = Vector::new((0..9).collect::<Vec<_>>());

        a += &2;
        assert_eq!(a.into_vec(), (2..11).collect::<Vec<_>>());

        let mut a = Vector::new((0..9).collect::<Vec<_>>());

        a += 2;
        assert_eq!(a.into_vec(), (2..11).collect::<Vec<_>>());

        let mut a = Vector::new((0..9).collect::<Vec<_>>());
        let b = Vector::new((0..9).collect::<Vec<_>>());

        a += &b;
        assert_eq!(a.into_vec(), (0..9).map(|x| 2 * x).collect::<Vec<_>>());

        let mut a = Vector::new((0..9).collect::<Vec<_>>());

        a += b;
        assert_eq!(a.into_vec(), (0..9).map(|x| 2 * x).collect::<Vec<_>>());
    }

    #[test]
    fn vector_sub_assign() {
        let mut a = Vector::new((0..9).collect::<Vec<_>>());

        a -= &2;
        assert_eq!(a.into_vec(), (-2..7).collect::<Vec<_>>());

        let mut a = Vector::new((0..9).collect::<Vec<i32>>());
        a -= 2;
        assert_eq!(a.into_vec(), (-2..7).collect::<Vec<_>>());

        let mut a = Vector::new((0..9).collect::<Vec<_>>());
        let b = Vector::new((0..9).collect::<Vec<_>>());

        a -= &b;
        assert_eq!(a.into_vec(), vec![0; 9]);

        let mut a = Vector::new((0..9).collect::<Vec<_>>());

        a -= b;
        assert_eq!(a.into_vec(), vec![0; 9]);
    }

    #[test]
    fn vector_div_assign() {
        let a_data = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let res_data = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
        let mut a = Vector::new(a_data.clone());

        a /= &2f32;
        assert_eq!(a.into_vec(), res_data.clone());

        let mut a = Vector::new(a_data.clone());
        a /= 2f32;
        assert_eq!(a.into_vec(), res_data.clone());
    }

    #[test]
    fn vector_mul_assign() {
        let a_data = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let res_data = vec![2f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let mut a = Vector::new(a_data.clone());

        a *= &2f32;
        assert_eq!(a.into_vec(), res_data.clone());

        let mut a = Vector::new(a_data.clone());
        a *= 2f32;
        assert_eq!(a.into_vec(), res_data.clone());
    }

}
