extern crate rusty_machine as rm;

use rm::datasets;
use rm::linalg::BaseMatrix;

#[test]
fn test_iris() {
    let dt = datasets::load_iris();
    assert_eq!(dt.data().rows(), 150);
    assert_eq!(dt.data().cols(), 4);

    assert_eq!(dt.target().size(), 150);
}