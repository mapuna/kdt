kd-tree
=======

Code is self explanatory, at least the python one.

- Simple DS: Divide and conquer in high dimension. Not highly optimized and simple for understanding the concepts. Still works better than all point distance comparisons.

## TODO

- Add optimizations. Try to follow the `scipy`'s implementation.
- Add a `C/C++` implementation.
    * Still deciding if I use BLAS/LAPACK/Eigen/uBLAS etc or do a CUDA implementation, or directly use `libtorch` API.