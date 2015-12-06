use std::ops::{Mul, Add, Div, Sub, Index};
use math::utils;

pub struct Vector {
	pub size: usize,
	pub data: Vec<f32>
}

impl Vector {
    pub fn new(size: usize, data: Vec<f32>) -> Vector {
        Vector {
            size: size,
            data: data
        }
    }

    pub fn zeros(size: usize) -> Vector {
    	Vector {
            size: size,
            data: vec![0.0; size]
        }
    }

    fn dot(&self, v: &Vector) -> f32 {
    	utils::dot(&self.data, &v.data)
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