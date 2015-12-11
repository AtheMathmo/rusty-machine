use std::ops::{Mul, Add, Div, Sub, Index};
use libnum::{One, Zero, Float};
use std::cmp::PartialEq;
use math::linalg::Metric;
use math::utils::dot;

pub struct Vector<T> {
	pub size: usize,
	pub data: Vec<T>
}

impl<T: Zero + One + Copy> Vector<T> {
    pub fn new(data: Vec<T>) -> Vector<T> {

    	let size = data.len();

        Vector {
            size: size,
            data: data
        }
    }

    pub fn zeros(size: usize) -> Vector<T> {
    	Vector {
            size: size,
            data: vec![T::zero(); size]
        }
    }

    pub fn ones(size: usize) -> Vector<T> {
    	Vector {
            size: size,
            data: vec![T::one(); size]
        }
    }
}

impl<T: Copy + One + Zero + Mul<T, Output=T> + Add<T, Output=T>> Vector<T> {
	pub fn dot(&self, v: &Vector<T>) -> T {
    	dot(&self.data, &v.data)
    }
}

impl<T: Copy + One + Zero + Mul<T, Output=T>> Mul<T> for Vector<T> {
    type Output = Vector<T>;

    fn mul(self, f: T) -> Vector<T> {
        (&self) * (&f)
    }
}

impl<'a, T: Copy + One + Zero + Mul<T, Output=T>> Mul<T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn mul(self, f: T) -> Vector<T> {
        self * (&f)
    }
}

impl<'a, T: Copy + One + Zero + Mul<T, Output=T>> Mul<&'a T> for Vector<T> {
    type Output = Vector<T>;

    fn mul(self, f: &T) -> Vector<T> {
        (&self) * f
    }
}

impl<'a, 'b, T: Copy + One + Zero + Mul<T, Output=T>> Mul<&'b T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn mul(self, f: &T) -> Vector<T> {
        let new_data = self.data.iter().map(|v| (*v) * (*f)).collect();

        Vector {
            size: self.size,
            data: new_data
        }
    }
}

impl<T: Copy + One + Zero + PartialEq + Div<T, Output=T>> Div<T> for Vector<T> {
    type Output = Vector<T>;

    fn div(self, f: T) -> Vector<T> {
        (&self) * (&f)
    }
}

impl<'a, T: Copy + One + Zero + PartialEq + Div<T, Output=T>> Div<T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn div(self, f: T) -> Vector<T> {
        self * (&f)
    }
}

impl<'a, T: Copy + One + Zero + PartialEq + Div<T, Output=T>> Div<&'a T> for Vector<T> {
    type Output = Vector<T>;

    fn div(self, f: &T) -> Vector<T> {
        (&self) * f
    }
}

impl<'a, 'b, T: Copy + One + Zero + PartialEq + Div<T, Output=T>> Div<&'b T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn div(self, f: &T) -> Vector<T> {
    	assert!(*f != T::zero());

        let new_data = self.data.iter().map(|v| *v / *f).collect();

        Vector {
            size: self.size,
            data: new_data
        }
    }
}

impl<T: Copy + One + Zero + Add<T, Output=T>> Add<T> for Vector<T> {
    type Output = Vector<T>;

    fn add(self, f: T) -> Vector<T> {
        (&self) * (&f)
    }
}

impl<'a, T: Copy + One + Zero + Add<T, Output=T>> Add<T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn add(self, f: T) -> Vector<T> {
        self * (&f)
    }
}

impl<'a, T: Copy + One + Zero + Add<T, Output=T>> Add<&'a T> for Vector<T> {
    type Output = Vector<T>;

    fn add(self, f: &T) -> Vector<T> {
        (&self) * f
    }
}

impl<'a, 'b, T: Copy + One + Zero + Add<T, Output=T>> Add<&'b T> for &'a Vector<T> {
	type Output = Vector<T>;

	fn add(self, f: &T) -> Vector<T> {
		let new_data = self.data.iter().map(|v| *v + *f).collect();

        Vector {
            size: self.size,
            data: new_data
        }
    }
}

impl<T: Copy + One + Zero + Add<T, Output=T>> Add<Vector<T>> for Vector<T> {
    type Output = Vector<T>;

    fn add(self, v: Vector<T>) -> Vector<T> {
        (&self) + (&v)
    }
}

impl<'a, T: Copy + One + Zero + Add<T, Output=T>> Add<Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;

    fn add(self, v: Vector<T>) -> Vector<T> {
        self + (&v)
    }
}

impl<'a, T: Copy + One + Zero + Add<T, Output=T>> Add<&'a Vector<T>> for Vector<T> {
    type Output = Vector<T>;

    fn add(self, v: &Vector<T>) -> Vector<T> {
        (&self) + v
    }
}

impl<'a, 'b, T: Copy + One + Zero + Add<T, Output=T>> Add<&'b Vector<T>> for &'a Vector<T> {
	type Output = Vector<T>;

	fn add(self, v: &Vector<T>) -> Vector<T> {
		assert!(self.size == v.size);

		let new_data = self.data.iter().enumerate().map(|(i,s)| *s + v.data[i]).collect();

        Vector {
            size: self.size,
            data: new_data
        }
	}
}

impl<T: Copy + One + Zero + Sub<T, Output=T>> Sub<T> for Vector<T> {
    type Output = Vector<T>;

    fn sub(self, f: T) -> Vector<T> {
        (&self) - (&f)
    }
}

impl<'a, T: Copy + One + Zero + Sub<T, Output=T>> Sub<T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn sub(self, f: T) -> Vector<T> {
        self - (&f)
    }
}

impl<'a, T: Copy + One + Zero + Sub<T, Output=T>> Sub<&'a T> for Vector<T> {
    type Output = Vector<T>;

    fn sub(self, f: &T) -> Vector<T> {
            (&self) - f
        }
    }

impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output=T>> Sub<&'b T> for &'a Vector<T> {
	type Output = Vector<T>;

	fn sub(self, f: &T) -> Vector<T> {
		let new_data = self.data.iter().map(|v| *v - *f).collect();

        Vector {
            size: self.size,
            data: new_data
        }
    }
}

impl<T: Copy + One + Zero + Sub<T, Output=T>> Sub<Vector<T>> for Vector<T> {
    type Output = Vector<T>;

    fn sub(self, v: Vector<T>) -> Vector<T> {
        (&self) - (&v)
    }
}

impl<'a, T: Copy + One + Zero + Sub<T, Output=T>> Sub<Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;

    fn sub(self, v: Vector<T>) -> Vector<T> {
        (&self) - v
    }
}

impl<'a, T: Copy + One + Zero + Sub<T, Output=T>> Sub<&'a Vector<T>> for Vector<T> {
    type Output = Vector<T>;

    fn sub(self, v: &Vector<T>) -> Vector<T> {
        (&self) - v
    }
}

impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output=T>> Sub<&'b Vector<T>> for &'a Vector<T> {
	type Output = Vector<T>;

	fn sub(self, v: &Vector<T>) -> Vector<T> {
		assert!(self.size == v.size);

		let new_data = self.data.iter().enumerate().map(|(i,s)| *s - v.data[i]).collect();

        Vector {
            size: self.size,
            data: new_data
        }
	}
}

impl<T> Index<usize> for Vector<T> {
	type Output = T;

	fn index(&self, idx : usize) -> &T {
		assert!(idx < self.size);

		&self.data[idx]
	}
}

impl<T: Float> Metric<T> for Vector<T> {
    fn norm(&self) -> T {
        let mut s = T::zero();

        for u in &self.data {
            s = s + (*u) * (*u);
        }

        s.sqrt()
    }
}