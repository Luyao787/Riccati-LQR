# Riccati-LQR

A C++ implementation of Linear Quadratic Regulator (LQR) using the square-root Riccati recursion.

## Overview

This project implements an efficient LQR solver designed for control applications, with quadrotor systems as an example.
For more information, please refer to the [tutorial](https://luyao787.github.io/blog/2025/LQP/).

## Dependencies

- **Eigen3**: Linear algebra library
- **CMake**: Build system

## Building

1. Clone the repository:
```bash
git clone https://github.com/Luyao787/Riccati-LQR.git
cd Riccati-LQR
```

2. Build the project:
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

3. Run the example:
```bash
./example
```

## References

- G. Frison and J. B. Jørgensen, “Efficient implementation of the Riccati recursion for solving linear-quadratic control problems,” 2013 IEEE International Conference on Control Applications (CCA), Hyderabad, India, 2013, pp. 1117-1122.
