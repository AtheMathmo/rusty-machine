use std::cmp;
use libnum::{Zero};
use std::ops::{Add, Mul};

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