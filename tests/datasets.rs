extern crate rusty_machine as rm;


#[cfg(datasets)]
mod test {

    use rm::datasets::iris;
    use rm::linalg::BaseMatrix;

    #[test]
    fn test_iris() {
        let dt = iris::load_();
        assert_eq!(dt.data().rows(), 150);
        assert_eq!(dt.data().cols(), 4);

        assert_eq!(dt.target().size(), 150);
    }
}