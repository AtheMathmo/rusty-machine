use linalg::vector::Vector;

pub trait Kernel {
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64;
}

pub struct SquaredExp {
    ls: f64,
    ampl: f64,
}

impl SquaredExp {
    pub fn new(ls: f64, ampl: f64) -> SquaredExp {
        SquaredExp {
            ls: ls,
            ampl: ampl,
        }
    }
}

impl Default for SquaredExp {
    fn default() -> SquaredExp {
        SquaredExp {
            ls: 1f64,
            ampl: 1f64,
        }
    }
}

impl Kernel for SquaredExp {
    fn kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        assert_eq!(x1.len(), x2.len());

        let v1 = Vector::new(x1.to_vec());
        let v2 = Vector::new(x2.to_vec());

        let x = -(&v1 - &v2).dot(&(v1 - v2)) / (2f64 * self.ls * self.ls);
        (self.ampl * x.exp())
    }
}
