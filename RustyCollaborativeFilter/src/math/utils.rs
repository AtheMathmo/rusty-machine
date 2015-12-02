use std::cmp;

pub fn dot(u: Vec<f32>, v: Vec<f32>) -> f32 {
    	let len = cmp::min(u.len(), v.len());
        let mut xs = &u[..len];
        let mut ys = &v[..len];

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