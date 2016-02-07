mod ziggurat_tables;

/// Sample a random number using the Ziggurat method (specifically the
/// ZIGNOR variant from Doornik 2005). Most of the arguments are
/// directly from the paper:
///
/// * `rng`: source of randomness
/// * `symmetric`: whether this is a symmetric distribution, or one-sided with P(x < 0) = 0.
/// * `X`: the $x_i$ abscissae.
/// * `F`: precomputed values of the PDF at the $x_i$, (i.e. $f(x_i)$)
/// * `F_DIFF`: precomputed values of $f(x_i) - f(x_{i+1})$
/// * `pdf`: the probability density function
/// * `zero_case`: manual sampling from the tail when we chose the
///    bottom box (i.e. i == 0)

// the perf improvement (25-50%) is definitely worth the extra code
// size from force-inlining.
#[inline(always)]
fn ziggurat<R: Rng, P, Z>(
            rng: &mut R,
            symmetric: bool,
            x_tab: ziggurat_tables::ZigTable,
            f_tab: ziggurat_tables::ZigTable,
            mut pdf: P,
            mut zero_case: Z)
            -> f64 where P: FnMut(f64) -> f64, Z: FnMut(&mut R, f64) -> f64 {
    const SCALE: f64 = (1u64 << 53) as f64;
    loop {
        // reimplement the f64 generation as an optimisation suggested
        // by the Doornik paper: we have a lot of precision-space
        // (i.e. there are 11 bits of the 64 of a u64 to use after
        // creating a f64), so we might as well reuse some to save
        // generating a whole extra random number. (Seems to be 15%
        // faster.)
        //
        // This unfortunately misses out on the benefits of direct
        // floating point generation if an RNG like dSMFT is
        // used. (That is, such RNGs create floats directly, highly
        // efficiently and overload next_f32/f64, so by not calling it
        // this may be slower than it would be otherwise.)
        // FIXME: investigate/optimise for the above.
        let bits: u64 = rng.gen();
        let i = (bits & 0xff) as usize;
        let f = (bits >> 11) as f64 / SCALE;

        // u is either U(-1, 1) or U(0, 1) depending on if this is a
        // symmetric distribution or not.
        let u = if symmetric {2.0 * f - 1.0} else {f};
        let x = u * x_tab[i];

        let test_x = if symmetric { x.abs() } else {x};

        // algebraically equivalent to |u| < x_tab[i+1]/x_tab[i] (or u < x_tab[i+1]/x_tab[i])
        if test_x < x_tab[i + 1] {
            return x;
        }
        if i == 0 {
            return zero_case(rng, u);
        }
        // algebraically equivalent to f1 + DRanU()*(f0 - f1) < 1
        if f_tab[i + 1] + (f_tab[i] - f_tab[i + 1]) * rng.gen::<f64>() < pdf(x) {
            return x;
        }
    }
}

pub mod consts {
    pub const SQRT_2_PI = 2.50662827463100050241576528481104525_f64;
    pub const LN_2_PI = 1.83787706640934548356065947281123527_f64;
}