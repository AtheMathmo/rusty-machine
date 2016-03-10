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