# Accelerating Genetic Algorithms using CUDA and GPU
CUDA is a parallel computing architecture developed by NVIDIA to massively increase computation power by using CPU and GPU cores together as a heterogeneous system. 
This project aims at improving the performance of genetic algorithms by parallelizing it using CUDA and GPU.  It shows about <b> 19 times </b> decrease in the computation time as compared to its sequential implementation. 

## Optimizing Benchmarking Functions
A set of 7 benchmarking functions are optimized to find their global minima using the parallely implementing GA. It takes about 19 times less time than the sequential implementation.

## Optimizing COCOMO Model 
COCOMO model is used for effort and cost estimations of software projects. But most of the projects require more time than that calculated by the conventional COCOMO model. We tend to solve this problem by using genetic algorithm to optimize the coefficients used in COCOMO model. The parallel implementation of GA on CUDA is used to optimize the COCOMO model coefficients on NASA dataset. This approach results in efforts and development time close to that of actual effort and time. Values obtained by basic COCOMO model drifts from actual effort calculated by about 73% whereas values obtained using optimized GA based approach drifts by just 17%. Also, the time required for calculations has been reduced by 19 times as compared to its sequential implementation.

