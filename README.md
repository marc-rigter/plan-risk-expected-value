# Planning for Risk-Aversion and Expected Value in MDPs

# Usage
src/main.cpp specifies the experiments to be run. The paths in this file specify where the results are saved as .csv files. Build the code by running make in the root directory. Then ./run in the root directory to start running experiments.

# Dependencies
The following dependencies are required:
- GSL
- Eigen3
- GMP
- lpSolve
- Shogun Machine Learning
- Gurobi 9.1
- Parma Polyhedron Library
- Boost
- G++

The dependencies can be installed using the following commands (tested on Ubuntu 20.04):
- sudo apt install libgmp-dev
- sudo apt install libppl-dev
- sudo apt install lp-solve
- sudo apt install libgsl-dev
- sudo apt install libboost-all-dev
- sudo apt install g++
- sudo apt install libeigen3-dev

The easiest way to install Shogun ML is with conda:
- conda install -c conda-forge shogun-cpp

However, alternative methods can be found at https://www.shogun-toolbox.org/install

Gurobi can be installed from the Gurobi website https://www.gurobi.com/downloads/

# Makefile
Edit GUROBI_PATH, SHOGUN_PATH, EIGEN_PATH, LP_SOLVE_PATH in the Makefile to point to where these dependencies are installed on your machine.


