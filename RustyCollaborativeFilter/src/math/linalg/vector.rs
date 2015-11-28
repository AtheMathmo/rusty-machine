use std::ops::{Mul, Add, Div, Sub, Index};
use std::cmp;

pub struct Vector {
	pub size: usize,
	pub data: Vec<f32>
}

impl Vector {
    fn new(size: usize, data: Vec<f32>) -> Vector {
        Vector {
            size: size,
            data: data
        }
    }

    fn zeros(size: usize) -> Vector {
    	Vector {
            size: size,
            data: vec![0.0; size]
        }
    }

    fn dot(self, v: Vector) -> f32 {
    	let len = cmp::min(self.size, v.size);
        let mut xs = &self.data[..len];
        let mut ys = &v.data[..len];

        let mut s = 0.;
        let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
            (0., 0., 0., 0., 0., 0., 0., 0.);

        while xs.len() >= 8 {
            p0 += xs[0] * ys[0];
            p1 += xs[1] * ys[1];
            p2 += xs[2] * ys[2];
            p3 += xs[3] * ys[3];
            p4 += xs[4] * ys[4];
            p5 += xs[5] * ys[5];
            p6 += xs[6] * ys[6];
            p7 += xs[7] * ys[7];

            xs = &xs[8..];
            ys = &ys[8..];
        }
        s += p0 + p4;
        s += p1 + p5;
        s += p2 + p6;
        s += p3 + p7;

        for i in 0..xs.len() {
            s += xs[i] * ys[i];
        }
        s
    }
}

impl Mul<f32> for Vector {
    type Output = Vector;

    fn mul(self, f: f32) -> Vector {
        let new_data = self.data.into_iter().map(|v| v * f).collect();

        Vector {
            size: self.size,
            data: new_data
        }
    }
}

impl Div<f32> for Vector {
    type Output = Vector;

    fn div(self, f: f32) -> Vector {
        let new_data = self.data.into_iter().map(|v| v / f).collect();

        Vector {
            size: self.size,
            data: new_data
        }
    }
}

impl Add<f32> for Vector {
	type Output = Vector;

	fn add(self, f: f32) -> Vector {
		let new_data = self.data.into_iter().map(|v| v + f).collect();

        Vector {
            size: self.size,
            data: new_data
        }
    }
}

impl Sub<f32> for Vector {
	type Output = Vector;

	fn sub(self, f: f32) -> Vector {
		let new_data = self.data.into_iter().map(|v| v - f).collect();

        Vector {
            size: self.size,
            data: new_data
        }
    }
}

impl Add<Vector> for Vector {
	type Output = Vector;

	fn add(self, v: Vector) -> Vector {
		assert!(self.size == v.size);

		let new_data = self.data.into_iter().enumerate().map(|(i,s)| s + v.data[i]).collect();

        Vector {
            size: self.size,
            data: new_data
        }
	}
}

impl Sub<Vector> for Vector {
	type Output = Vector;

	fn sub(self, v: Vector) -> Vector {
		assert!(self.size == v.size);

		let new_data = self.data.into_iter().enumerate().map(|(i,s)| s - v.data[i]).collect();

        Vector {
            size: self.size,
            data: new_data
        }
	}
}

impl Index<usize> for Vector {
	type Output = f32;

	fn index(&self, idx : usize) -> &f32 {
		assert!(idx < self.size);

		&self.data[idx]
	}
}