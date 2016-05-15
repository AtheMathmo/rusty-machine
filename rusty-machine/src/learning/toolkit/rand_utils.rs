//! Utility functions for random sampling
//!
//! Currently this module only includes reservoir sampling.

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
	assert!(pool.len() >= reservoir_size, "Sample size is greater than total.");
    
	let mut pool_mut = &pool[..];

	let mut res = pool_mut[..reservoir_size].to_vec();
	pool_mut = &pool_mut[reservoir_size..];
	
	let mut ele_seen = reservoir_size;
	let mut rng = thread_rng();

	while pool_mut.len() > 0 {
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
pub fn fisher_yates<T: Copy>(arr: &[T]) -> Vec<T> {
	let n = arr.len();
	let mut rng = thread_rng();

	let mut shuffled_arr = Vec::with_capacity(n);

	for i in 0..n {
		let j = rng.gen_range(0, i + 1);

		if j != i {
			let x = shuffled_arr[j];
			shuffled_arr.push(x);
		}

		shuffled_arr[j] = arr[i];
	}

	shuffled_arr
}

/// The in place Fisher-Yates shuffle.
pub fn in_place_fisher_yates<T>(arr: &mut [T]) {
	let n = arr.len();
	let mut rng = thread_rng();

	for i in 0..n {
		let j = rng.gen_range(0, n - i);
		arr.swap(i, i + j);
	}
}