extern crate rusty_collaborative_filter as rcf;

use rcf::math::utils;

#[test]
fn argmax() {
	let a = vec![1.0, 2.0, 3.0, 4.0];
	assert_eq!(utils::argmax(&a), 3);

	let b = vec![0., 0., 0.];
	assert_eq!(utils::argmax(&b), 0);

	let c = vec![0., 1., 0.];
	assert_eq!(utils::argmax(&c), 1);
}