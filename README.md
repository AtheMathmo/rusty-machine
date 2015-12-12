# rusty-machine

[Documentation](https://AtheMathmo.github.io/rusty-machine/)

This project was originally intended to be an implementation of Online Collaborative Filtering in Rust. However, after putting a lot of effort into building the linear algebra library that was needed I decided I shouldn't waste the effort and should make something more general!

So this project is now going to be a machine learning crate for Rust. I began this project as a fun way to learn Rust and so there will be lots of things that need improving (I'm still not very familiar with LLVM). I hope that this crate will provide a number of standard out-the-box machine learning algorithms.

## Implementation

This project is implemented using [Rust](https://www.rust-lang.org/).

## Motivation

This key motivation behind this project was for me to learn some systems programming and a new language! I wanted to try implementing a linear algebra library within a systems programming language and then extend this to explore some machine learning algorithms.

## Current Progress

Working on the linear algebra library that will power the machine learning algorithms. Once this implemented the core algorithms should be easy to add on.

### TODO:

- Implement linear systems solving using LPU (which will give inversion).
- Multi-threaded divide and conquer matrix multiplication (currently iterative).
- Tidy up indexing.

After the above are done I will move onto the actual machine learning stuff!

## References

[Online Collaborative Filtering](http://canini.me/research_files/OnlineCollaborativeFiltering.pdf)
