# Accelerated Gibbs Sampling on Nvidia GPUs

This project is based on an assignment for UCLA's graduate-level course -- 
Current Topics in Computer Science: System Design/Architecture: Learning Machines.

# Building

Useful commands:

- `make` -- build project with default configurations
- `make NSAMPLES=n KCLASSES=k` -- build project and set parameters 
- `make debug` -- build project with -G flag for device (kernel) debugging
- `make profile` -- build project with --generate-line-info flag to generate line-number info for device code
- `make clean` -- runs `rm -rf` on `./obj` and `./bin`

https://docs.nvidia.com/cuda/curand/host-api-overview.html

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

https://docs.nvidia.com/cuda/curand/host-api-overview.html#generator-types

https://scikit-learn.org/stable/modules/mixture.html

https://docs.nvidia.com/cuda/curand/device-api-overview.html#distributions

https://docs.nvidia.com/cuda/curand/device-api-overview.html#poisson-api-example