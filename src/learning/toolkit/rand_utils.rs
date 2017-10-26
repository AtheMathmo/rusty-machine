//! Utility functions for random functionality.
//!
//! This module provides sampling and shuffling which are used
//! within the learning modules.

use rand::{Rng, thread_rng};

/// ```
/// use rusty_machine::learning::toolkit::rand_utils;
///
/// let mut pool = &mut [1,2,3,4];
/// let sample = rand_utils::reservoir_sample(pool, 3);
///
/// println!("{:?}", sample);
/// ```
pub fn reservoir_sample<T: Copy>(pool: &[T], reservoir_size: usize) -> Vec<T> {
    assert!(pool.len() >= reservoir_size,
            "Sample size is greater than total.");

    let mut pool_mut = &pool[..];

    let mut res = pool_mut[..reservoir_size].to_vec();
    pool_mut = &pool_mut[reservoir_size..];

    let mut ele_seen = reservoir_size;
    let mut rng = thread_rng();

    while !pool_mut.is_empty() {
        ele_seen += 1;
        let r = rng.gen_range(0, ele_seen);

        let p_0 = pool_mut[0];
        pool_mut = &pool_mut[1..];

        if r < reservoir_size {
            res[r] = p_0;
        }
    }

    res
}

/// The inside out Fisher-Yates algorithm.
///
/// # Examples
///
/// ```
/// use rusty_machine::learning::toolkit::rand_utils;
///
/// // Collect the numbers 0..5
/// let a = (0..5).collect::<Vec<_>>();
///
/// // Perform a Fisher-Yates shuffle to get a random permutation
/// let permutation = rand_utils::fisher_yates(&a);
/// ```
pub fn fisher_yates<T: Copy>(arr: &[T]) -> Vec<T> {
    let n = arr.len();
    let mut rng = thread_rng();

    let mut shuffled_arr = Vec::with_capacity(n);

    unsafe {
        // We set the length here
        // We only access data which has been initialized in the algorithm
        shuffled_arr.set_len(n);
    }

    for i in 0..n {
        let j = rng.gen_range(0, i + 1);

        // If j isn't the last point in the active shuffled array
        if j != i {
            // Copy value at position j to the end of the shuffled array
            // This is safe as we only read initialized data (j < i)
            let x = shuffled_arr[j];
            shuffled_arr[i] = x;
        }

        // Place value at end of active array into shuffled array
        shuffled_arr[j] = arr[i];
    }

    shuffled_arr
}

/// The in place Fisher-Yates shuffle.
///
/// # Examples
///
/// ```
/// use rusty_machine::learning::toolkit::rand_utils;
///
/// // Collect the numbers 0..5
/// let mut a = (0..5).collect::<Vec<_>>();
///
/// // Permute the values in place with Fisher-Yates
/// rand_utils::in_place_fisher_yates(&mut a);
/// ```
pub fn in_place_fisher_yates<T>(arr: &mut [T]) {
    let n = arr.len();
    let mut rng = thread_rng();

    for i in 0..n {
        // Swap i with a random point after it
        let j = rng.gen_range(0, n - i);
        arr.swap(i, i + j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservoir_sample() {
        let a = vec![1, 2, 3, 4, 5, 6, 7];

        let b = reservoir_sample(&a, 3);

        assert_eq!(b.len(), 3);
    }

    #[test]
    fn test_fisher_yates() {
        let a = (0..10).collect::<Vec<_>>();

        let b = fisher_yates(&a);

        for val in a.iter() {
            assert!(b.contains(val));
        }
    }

    #[test]
    fn test_in_place_fisher_yates() {
        let mut a = (0..10).collect::<Vec<_>>();

        in_place_fisher_yates(&mut a);

        for val in 0..10 {
            assert!(a.contains(&val));
        }
    }
}
