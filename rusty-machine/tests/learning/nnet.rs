use rm::linalg::matrix::Matrix;
use rm::learning::SupModel;
use rm::learning::nnet::NeuralNet;

#[test]
fn test_model() {

	let data = Matrix::new(5,3, vec![1.,1.,1.,2.,2.,2.,3.,3.,3.,
									4.,4.,4.,5.,5.,5.,]);
	let outputs = Matrix::new(5,3, vec![1.,0.,0.,0.,1.,0.,0.,0.,1.,
										0.,0.,1.,0.,0.,1.]);

	let layers = &[3,5,11,7,3];
	let mut model = NeuralNet::new(layers);

	model.train(data, outputs);

	let test_data = Matrix::new(5,3, vec![1.,1.,1.,2.,2.,2.,3.,3.,3.,
									4.,4.,4.,5.,5.,5.,]);

	model.predict(test_data);
}