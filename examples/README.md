Examples with rusty-machine

This directory gathers fully-fledged programs, each using a piece of
`rusty-machine`'s API.

## Overview

* [K-Means](#k-means)
* [SVM](#svm)
* [Neural Networks](#neural-networks)
* [Naïve Bayes](#naïve-bayes)
* [LDA](#lda)

## The Examples

### K Means

#### Generating Clusters

[Generating Clusters](k-means_generating_clusters.rs) randomly generates data around a pair of clusters.
It then trains a K-Means model to learn new centroids from this sample.

The example shows a basic usage of the K-Means API - an Unsupervised model. We also show some basic usage
of [rulinalg](https://github.com/AtheMathmo/rulinalg) to generate the data.

Sample run:

```
cargo run --example k-means_generating_cluster
   Compiling rusty-machine v0.4.0 (file:///rusty-machine/rusty-machine)
     Running `target/debug/examples/k-means_generating_cluster`
K-Means clustering example:
Generating 2000 samples from each centroids:
⎡-0.5 -0.5⎤
⎣   0  0.5⎦
Training the model...
Model Centroids:
⎡-0.812 -0.888⎤
⎣-0.525  0.877⎦
Classifying the samples...
Samples closest to first centroid: 1878
Samples closest to second centroid: 2122
```

### SVM

#### Sign Learner

[Sign learner](svm-sign_learner.rs) constructs and evaluates a model that learns to recognize the sign of an input number.

The sample shows a basic usage of the SVM API. It also configures the SVM algorithm with a specific kernel (`HyperTan`).
Evaluations are run in a loop to log individual predictions and do some book keeping for reporting the performance at the end.
The salient part from `rusty-machine` is to use the `train` and `predict` methods of the SVM model.

The accuracy evaluation is simplistic, so the model manages 100% accuracy (which is *really* too simple an example).

Sample run:

```
cargo run --example svm-sign_learner
   Compiling rusty-machine v0.3.0 (file:///rusty-machine/rusty-machine)
     Running `target/debug/examples/svm-sign_learner`
Sign learner sample:
Training...
Evaluation...
-1000 -> -1: true
-900 -> -1: true
-800 -> -1: true
-700 -> -1: true
-600 -> -1: true
-500 -> -1: true
-400 -> -1: true
-300 -> -1: true
-200 -> -1: true
-100 -> -1: true
0 -> -1: true
100 -> 1: true
200 -> 1: true
300 -> 1: true
400 -> 1: true
500 -> 1: true
600 -> 1: true
700 -> 1: true
800 -> 1: true
900 -> 1: true
Performance report:
Hits: 20, Misses: 0
Accuracy: 100
```

### Neural Networks

#### AND Gate

[AND gate](nnet-and_gate.rs) makes an AND gate out of a perceptron.

The sample code generates random data to learn from.
The input data is like an electric signal between 0 and 1, with some jitter that makes it not quite 0 or 1.
By default, the code decides that any pair input "above"
(0.7, 0.7) is labeled as 1.0 (AND gate passing), otherwise labeled as 0.0 (AND gate blocking).
This means that the training set is biased toward learning the passing scenario: An AND gate passes
25% of the time on average, and we'd like it to learn it.

The test data uses only the 4 "perfect" inputs for a gate: (0.0, 0.0), (1.0, 0.0), etc.

The code generates 10,000 training data points by default. Please give it a try, and then change `SAMPLE`,
the number of training data points, and `THRESHOLD`, the value for "deciding" for a passing gate.

Sample run:

```
> cargo run --example nnet-and_gate
   Compiling rusty-machine v0.3.0 (file:///rusty-machine/rusty-machine)
     Running `target/debug/examples/nnet-and_gate`
AND gate learner sample:
Generating 10000 training data and labels...
Training...
Evaluation...
Got  Expected
0.00  0
0.00  0
0.96  1
0.01  0
Hits: 4, Misses: 0
Accuracy: 100%
```

### Naïve Bayes

#### Dog Classification

Suppose we have a population composed of red dogs and white dogs,
whose friendliness, furriness, and speed can be measured. In this
example we train a Naïve Bayes model to determine whether
a dog is white or red.

The group of white dogs are friendlier, furrier, and slower than
the red dogs. Given the color of a dog, friendliness, furriness,
and speed are independent of each other (a requirement of the Naïve
Bayes model).

In the example code we will generate our own data and then train
our model using it. This is a common technique used to validate
a model. We generate the data by sampling each of the dogs features
from Gaussian random variables. We will have a total of 6 Gaussian
random variables representing three features for both colors of dog.
As we are using Gaussian random variables we will use a Gaussian
Naive Bayes model. Once we have generated our data we will convert
it into `Matrix` structures and train our model.


Sample run:

```
$ cargo run --example naive_bayes_dogs
...
Predicted: Red; Actual: Red; Accurate? true
Predicted: Red; Actual: Red; Accurate? true
Predicted: White; Actual: Red; Accurate? false
Predicted: Red; Actual: White; Accurate? false
Predicted: Red; Actual: Red; Accurate? true
Predicted: White; Actual: White; Accurate? true
Predicted: White; Actual: White; Accurate? true
Predicted: White; Actual: White; Accurate? true
Predicted: White; Actual: White; Accurate? true
Predicted: Red; Actual: Red; Accurate? true
Accuracy: 822/1000 = 82.2%
```

### LDA

#### Word distribution

The [word distribution](lda_gen.rs) example starts by generating a distribution
of words over topics, then generating documents based on a distribution of
topics.  The example then tries to estimate the distribution of words based only
on the generated documents.

The generated distribution (G) of words are visualized as a grid, with each cell
in the grid corresponding to the probability of a particular word being
selected. Following this, documents (D) are generated based on a distribution
over these topics.

The distribution for each topic is shown, then Linear Dirichlet Allocation is
used to try to estimate the distribution (E) of words to topic, based solely on
generated documents (D).

The resulting word distribution(E) is printed.  The order may not be the same,
but for each estimated topic in (E), there should be a corresponding generated
distribution in (G).

Sample run:
```
$ cargo run --example lda_gen
...
Creating word distribution
Distribution generated:
Topic 1     Topic 2     Topic 3     Topic 4     Topic 5     
-------     -------     -------     -------     -------     
99999       .....       .....       .....       .....       
.....       99999       .....       .....       .....       
.....       .....       99999       .....       .....       
.....       .....       .....       99999       .....       
.....       .....       .....       .....       99999       


Topic 6     Topic 7     Topic 8     Topic 9     Topic 10    
-------     -------     -------     -------     -------     
9....       .9...       ..9..       ...9.       ....9       
9....       .9...       ..9..       ...9.       ....9       
9....       .9...       ..9..       ...9.       ....9       
9....       .9...       ..9..       ...9.       ....9       
9....       .9...       ..9..       ...9.       ....9       


Generating documents
Predicting word distribution from generated documents
Prediction completed.  Predicted word distribution:
(Should be similar to generated distribution above)
Topic 1     Topic 2     Topic 3     Topic 4     Topic 5     
-------     -------     -------     -------     -------     
..8..       .....       .....       ....8       8....       
..8..       .....       .....       ....8       8....       
..9..       98888       .....       ....9       8....       
..8..       .....       .....       ....8       8....       
..8..       .....       88988       ....8       9....       


Topic 6     Topic 7     Topic 8     Topic 9     Topic 10    
-------     -------     -------     -------     -------     
...8.       .....       .8...       .....       89888       
...8.       .....       .8...       88889       .....       
...8.       .....       .9...       .....       .....       
...8.       88889       .8...       .....       .....       
...9.       .....       .8...       .....       .....       
```
