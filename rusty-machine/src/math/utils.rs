//! Linear algebra utils module.
//! 
//! Contains support methods for linear algebra structs.

use std::cmp;
use libnum::{Zero};
use std::ops::{Add, Mul};

/// Compute dot product of two slices.
///
/// # Examples
///
/// ```
/// use rusty_machine::math::utils;
/// let a = vec![1.0,2.0,3.0,4.0];
/// let b = vec![1.0,2.0,3.0,4.0];
///
/// let c = utils::dot(&a,&b);
/// ```
pub fn dot<T: Copy + Zero + Add<T, Output=T> + Mul<T, Output=T>>(u: &[T], v: &[T]) -> T {
    let len = cmp::min(u.len(), v.len());
    let mut xs = &u[..len];
    let mut ys = &v[..len];

    let mut s = T::zero();
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (T::zero(), T::zero(), T::zero(), T::zero(),
            T::zero(), T::zero(), T::zero(), T::zero());

    while xs.len() >= 8 {
        p0 = p0 + xs[0] * ys[0];
        p1 = p1 + xs[1] * ys[1];
        p2 = p2 + xs[2] * ys[2];
        p3 = p3 + xs[3] * ys[3];
        p4 = p4 + xs[4] * ys[4];
        p5 = p5 + xs[5] * ys[5];
        p6 = p6 + xs[6] * ys[6];
        p7 = p7 + xs[7] * ys[7];

        xs = &xs[8..];
        ys = &ys[8..];
    }
    s = s + p0 + p4;
    s = s + p1 + p5;
    s = s + p2 + p6;
    s = s + p3 + p7;

    for i in 0..xs.len() {
        s = s + xs[i] * ys[i];
    }
    s
 }

/// Compute sum of two slices.
///
/// # Examples
///
/// ```
/// use rusty_machine::math::utils;
/// let a = vec![1.0,2.0,3.0,4.0];
/// let b = vec![1.0,2.0,3.0,4.0];
///
/// let c = utils::unrolled_sum(&a,&b);
/// ```
pub fn unrolled_sum<T: Copy + Zero + Add<T, Output=T>> (u: &[T], v: &[T]) -> Vec<T> {
    let len = cmp::min(u.len(), v.len());
    let mut xs = &u[..len];
    let mut ys = &v[..len];
    
    let mut sum_data = vec![T::zero(); len];
    let mut holder = 0;

    while xs.len() >= 8 {
        sum_data[0+holder] = xs[0] + ys[0];
        sum_data[1+holder] = xs[1] + ys[1];
        sum_data[2+holder] = xs[2] + ys[2];
        sum_data[3+holder] = xs[3] + ys[3];
        sum_data[4+holder] = xs[4] + ys[4];
        sum_data[5+holder] = xs[5] + ys[5];
        sum_data[6+holder] = xs[6] + ys[6];
        sum_data[7+holder] = xs[7] + ys[7];

        xs = &xs[8..];
        ys = &ys[8..];
        
        holder += 8;
    }

    for i in 0..xs.len() {
        sum_data[i+holder] = xs[i] + ys[i];
    }
    sum_data
}

/// Find argmax of slice.
///
/// Returns index of first occuring maximum.
///
/// # Examples
///
/// ```
/// use rusty_machine::math::utils;
/// let a = vec![1.0,2.0,3.0,4.0];
///
/// let c = utils::argmax(&a);
/// assert_eq!(c, 3);
/// ```
pub fn argmax<T: Copy + PartialOrd>(u: &[T]) -> usize {
    assert!(u.len() != 0);

    let mut max_index = 0;
    let mut max = u[max_index];

    for (i, v) in (u.iter()).enumerate() {
        if max < *v {
            max_index = i;
            max = *v;
        }
    }

    max_index
}

/// Find index of value in slice.
///
/// Returns index of first occuring value.
///
/// # Examples
///
/// ```
/// use rusty_machine::math::utils;
/// let a = vec![1.0,2.0,3.0,4.0];
///
/// let c = utils::find(&a, 3.0);
/// assert_eq!(c, 2);
/// ```
pub fn find<T: PartialEq>(p: &[T], u: T) -> usize {
        for (i, v) in p.iter().enumerate() {
            if *v == u {
                return i;
            }
        }

        panic!("Value not found.")
    }