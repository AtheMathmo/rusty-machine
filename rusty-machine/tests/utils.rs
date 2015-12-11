extern crate rusty_machine as rm;

use rm::math::utils;

#[test]
fn argmax() {
	let a = vec![1.0, 2.0, 3.0, 4.0];
	assert_eq!(utils::argmax(&a), 3);

	let b = vec![0., 0., 0.];
	assert_eq!(utils::argmax(&b), 0);

	let c = vec![0., 1., 0.];
	assert_eq!(utils::argmax(&c), 1);
}

#[test]
fn find() {
	let a = vec![1.0, 2.0, 4.0, 5.0];

	let b = utils::find(&a, 1.0);
	let c = utils::find(&a, 4.0);

	assert_eq!(b, 0);
	assert_eq!(c, 2);
}