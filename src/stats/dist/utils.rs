//! Utilities for  distribution calculations
use std::f64;

/// Some of the logic around bounds checking is based on code found written by W. Fullerton,
/// Found at http://www.netlib.org/slatec/fnlib/


// Pre-calculate to save computation later
const LN_SQRT_2_PI:f64 = 0.918938533204672741780329736406;


// These constants are taken from Paul Godfrey's analysis of the Lanczos coefficients
// See http://www.numericana.com/answer/info/godfrey.htm
const LANCZOS_COEFFICIENTS:[f64;11] = [
        1.000000000000000174663,
     5716.400188274341379136,
   -14815.30426768413909044,
    14291.49277657478554025,
    -6348.160217641458813289,
     1301.608286058321874105,
     -108.1767053514369634679,
        2.605696505611755827729,
       -0.7423452510201416151527e-2,
        0.5384136432509564062961e-7,
       -0.4023533141268236372067e-8
];


/// Computes the log of the absolute value of the gamma function.
/// That is, it finds ln|gamma(x)|.  This function is better than
/// simply calling gamma(x).abs().ln() because it is defined on a much
/// larger domain than gamma.
///
/// This function is defined on [-2.5327372760800758e+305, 2.5327372760800758e+305], excluding
/// integers less than 1.  Results outside this domain is too large to be held in an f64.
///
/// This function uses the
/// [Lanczos approximation](https://en.wikipedia.org/wiki/Lanczos_approximation) to estimate
/// gamma.  This estimation has a relative error of less than 10^-13
/// (see [Godfrey](http://www.numericana.com/answer/info/godfrey.htm) for details)
///
pub fn lgamma(x: f64) -> f64 {

    // Gamma is not defined on negative integers
    if x < 0.0 && x.trunc() == x {
        return f64::INFINITY;
    }
    let y = x.abs();

    // For very small values of x, we can approximate ln(gamma(x)) as ln(x)
    if y < 1e-306 {
        return -y.ln();
    }


    // For very large x, the log of the gamma function isn't computable
    if y > 2.5327372760800758e+305 {
        return f64::INFINITY;
    }

    // We use the reflection formula to handle the negative case
    if x < 0.5 {
        // XXX It may be possible to speed this up a bit by
        // inlining this, instead of recursing, since we don't need all
        // the above checks in that case
        return (f64::consts::PI / sinpi(x).abs()).ln() -  lgamma(1.0 - x);
    }

    let mut divisor = x;
    let mut sum = LANCZOS_COEFFICIENTS[0];

    for k in 1..11 {
        sum += LANCZOS_COEFFICIENTS[k]/ divisor;
        divisor += 1.0;
    }
    let term = x + 8.5;
    return  sum.ln() + LN_SQRT_2_PI + (x - 0.5) * term.ln() - term;
}

/// Calculates sin(pi * x).  For some known values of x, is able to compute very fast and accurate
fn sinpi(x: f64) -> f64 {
    let mut y = x % 2.0;
    if y <= -1.0 {
        y += 2.0
    } else if y > 1.0 {
        y -= 2.0;
    }
    if y == 0.0 || y == 1.0 {
        return 0.0;
    }
    if y == 0.5 {
        return 1.0;
    }
    if y == -0.5 {
        return -1.0;
    }
    return (y * f64::consts::PI).sin()
}




/// Computes an approximation of the gamma function.
///
/// This function is only defined on [-170.5674972726612,  171.61447887182298], exceluding
/// integers less than 1.  Results outside this domain is too large to be held in an f64.
///
/// This function uses the
/// [Lanczos approximation](https://en.wikipedia.org/wiki/Lanczos_approximation) to estimate
/// gamma.  This estimation has a relative error of less than 10^-13
// (see [Godfrey](http://www.numericana.com/answer/info/godfrey.htm) for details)
#[allow(dead_code)]
pub fn gamma(z: f64) -> f64 {
    if z.is_nan() {
        return z;
    }

    if z.trunc() == z {
        // Gamma is not defined on 0 or negative integers
        if z <= 0.0 {
			println!("Truncated");
            return f64::NAN;
        } else if z <= 50.0 {
            // For small enough integers, just calculate the factorial
            return (2..z as u64).product::<u64>() as f64;
        }
    }

    // For sufficiently large z, we cannot calculate the gamma function,
    // so we return infinity
    if z > 171.61447887182298 {
        return f64::INFINITY;
    }

    // For suffiently small z, we cannoy calculate the gamma function, so we return 0
    if z < -170.5674972726612 {
        return 0.0
    }

    // When z is close to 0, we cannot guarantee an accurate result, so we go to infinity.
    if z.abs() < 2.2474362225598545e-308 {
        if z > 0.0 {
             return f64::INFINITY;
         } else {
             return f64::NEG_INFINITY;
         }
    }

    // We use the reflection formula to handle the negative case
    if z < 0.5 {
        return f64::consts::PI / (sinpi(z) * gamma(1.0 - z));
    }

    let mut divisor = z;
    let mut sum = LANCZOS_COEFFICIENTS[0];

    for k in 1..11 {
        sum += LANCZOS_COEFFICIENTS[k]/ divisor;
        divisor += 1.0;
    }
    let term = z + 8.5;
	let result = sum.ln() + LN_SQRT_2_PI + (z - 0.5) * term.ln() - term;
    return result.exp();
}

#[cfg(test)]
mod test {
    use super::{gamma, lgamma};
    use std::f64;
    use rand::{thread_rng, Rng};

    macro_rules! assert_close {
        ($left: expr, $right: expr) => {
            assert!((($left - $right)/$left).abs() < 0.000000000001, "{:?} != {:?}", $left, $right);
        };
    }
    #[test]
    fn test_gamma() {
        assert_eq!(gamma(1.0), 1.0);
        assert_eq!(gamma(2.0), 1.0);
        assert_eq!(gamma(3.0), 2.0);
        assert_eq!(gamma(4.0), 6.0);
        assert_eq!(gamma(5.0), 24.0);
        assert_eq!(gamma(6.0), 120.0);
        assert!(gamma(f64::NAN).is_nan());
        assert!(gamma(0.0).is_nan());
        assert!(gamma(-1.0).is_nan());
        assert!(gamma(-22.0).is_nan());
        assert_close!(gamma(-0.5), -3.54490770181103205);
        assert_close!(gamma(-1.3333333333333), 3.046765363709648);
        assert_close!(gamma(-2.25), -1.74281486572825265);
        assert_close!(gamma(5.6), 61.5539150062892670);
        assert_close!(gamma(6.23567), 180.2847985322989466);
        println!("gamma(100): {:?}", gamma(100.0));
        assert_close!(gamma(100.0), 9.33262154439441526e155);
        println!("gamma(70): {:?}", gamma(70.0));
        assert_close!(gamma(70.0), 1.7112245242814131e98);
        //assert_close!(gamma(-50.234), -6.14821595274063619286e-65);

    }

    #[test]
    fn test_lgamma() {
        for i in 0..100 {
            let test = thread_rng().next_f64() * 340.0 - 170.0;
            println!("Test {}: {}", i, test);
            assert_close!(lgamma(test), gamma(test).abs().ln());
        }
    }


}
