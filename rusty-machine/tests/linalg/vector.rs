use rm::linalg::vector::Vector;
use rm::linalg::Metric;

#[test]
fn create_vector_new() {
    let a = Vector::new(vec![1.0; 12]);

    assert_eq!(a.size(), 12);

    for i in 0..12 {
        assert_eq!(a[i], 1.0);
    }
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
