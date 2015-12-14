# rusty-machine

[Documentation](https://AtheMathmo.github.io/rusty-machine/)

This project was originally intended to be an implementation of Online Collaborative Filtering in Rust. However, after putting a lot of effort into building the linear algebra library that was needed I decided I shouldn't waste the effort and should make something more general!

So this project is now going to be a machine learning crate for Rust. I began this project as a fun way to learn Rust and so there will be lots of things that need improving (I'm still not very familiar with LLVM). I hope that this crate will provide a number of standard out-the-box machine learning algorithms.

#### Help!
I've probably introduced some bad practices as I've been learning Rust. If you have any suggestions for structure or optimization please let me know.

## Implementation

This project is implemented using [Rust](https://www.rust-lang.org/).

## Motivation

This key motivation behind this project was for me to learn some systems programming and a new language! I wanted to try implementing a linear algebra library within a systems programming language and then extend this to explore some machine learning algorithms.

## Current Progress

I have completed a first pass at the linear algebra library. It is now functional enough that I can start adding some ML! I've implemented a basic linear regression module.

I will need to update and optimize the linear algebra library but should be able to do this in tandem.

### TODO:

- Multi-threaded divide and conquer matrix multiplication (currently iterative).
- Tidy up indexing.
- Start work on statistics components - R.V. sampling etc.

## References

[Online Collaborative Filtering](http://canini.me/research_files/OnlineCollaborativeFiltering.pdf)
