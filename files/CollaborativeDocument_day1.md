![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document (day 1)

2026-03-17-ds-gpu

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------


##  🫱🏽‍🫲🏻 Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.

 If you feel that the code of conduct is breached, please talk to one of the instructors (if the complaint is for one of the participants) or send an email to training@esciencecenter.nl (if the complaint is for one of the instructors).
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🔧 Exercises

### Challenge: convolution on the GPU without CuPy

Try to convolve the NumPy array `diracs` with the NumPy array `gauss` directly on the GPU, without using CuPy arrays. If this works, it should save us the time and effort of transferring `diracs` and `gauss` to the GPU.

#### Solution

Unfortunately, it is impossible to access directly from the GPU the NumPy arrays that live in the host RAM.

`TypeError: Cannot construct a dtype from an array`


### Challenge: fairer runtime comparison CPU vs. GPU

Compute again the speedup achieved using the GPU, but try to take also into account the time spent transferring the data to the GPU and back.

Hint: to copy a CuPy array back to the host (CPU), use the `cp.asnumpy()` function.

#### Solution

Define a function that includes all transfers to and from the host, plus the computation on the device.
```
def push_compute_pull():
    diracs_gpu = cp.asarray(diracs)
    gauss_gpu = cp.asarray(gauss)
    convolved_image_gpu = convolve2d_gpu(diracs_gpu, gauss_gpu)
    convolved_image_gpu_in_host = cp.asnumpy(convolved_image_gpu)

%gpu_timeit -n 10 push_compute_pull()
```

## 🧠 Collaborative Notes

### Accelerating Numpy code with CuPy

Login to [Snellius](https://ondemand.snellius.surf.nl) to use the Jupyter Notebooks and follow the setup instructions.

We start by replacing NumPy with a library, CuPy, that accelerates NumPy primitives on the GPU.

First we import NumPy.
```python
import numpy as np
```

We then create an image and plot it.
```python
diracs = np.zeros((2048, 2048))
diracs[8::16, 8::16] = 1
```

```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(diracs[:32, :32])
plt.show()
```

We want to perform a convolution on this image. To do this we create a convolution kernel first.
```python
x, y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
dist = np.sqrt(x**2 + y**2)
gauss = np.exp(-dst**2 / 2.0)

plt.imshow(gauss)
```

And then convolve the image using SciPy.
```python
from scipy.signal import convolve2d as convolve2d_cpu

convolved_image_cpu = convolve2d_cpu(diracs, gauss)
plt.imshow(convolved_image_cpu[:32, :32])
```

The output image is padded. This is all done by the SciPy library. We can check that this is the case by printing the image size.
```python
print(convolved_image_cpu.shape)
```

Time the Python CPU code.
```python
%time convolve2d_cpu(diracs, gauss)
```

During the course it took approximately 2 seconds to run the code.

To use the code on the GPU we need to first copy the input image from the host memory, to the GPU (device) memory.
```python
import cupy as cp
```

If we forgot to install CuPy the previous line will not work, we then need to execute this command to install all packages.
```python
!pip install --user cuda-python==12.9.4 cuda-core[cu12] cupy-cuda12x numba kernel_tuner[cuda] torch astropy
```

To install CuPy for AMD GPUs you need to follow the instructions provided in the [documentation](https://docs.cupy.dev/en/stable/install.html#using-cupy-on-amd-gpu-experimental).

We have CuPy installed now.

To copy the data from the CPU to the GPU we use `asarray` from CuPy.
```python
diracs_gpu = cp.asarray(diracs)
gauss_gpu = cp.asarray(gauss)
```

We can import the convolution function from CuPy and run it on the GPU.
```python
from cupyx.scipy.signal import convolve2d as convolve2d_gpu

convolved_image_gpu = convolve2d_gpu(diracs_gpu, gauss_gpu)
```

We can now time how long this operation takes on the GPU.
```python
%time convolved_image_gpu = convolve2d_gpu(diracs_gpu, gauss_gpu)
```

The time we get is way to small (less than one millisecond!). GPUs are asynchronous with respect to the host, so timing works differently.To time the code correctly we need to use a different timing function.
```python
%load_ext cupyx.profiler

%gpu_timeit convolve2d_gpu(diracs_gpu, gauss_gpu)
```

The convolution takes only a few milliseconds on the GPU we are using on Snellius. The timing does not include transferring the data to and from the GPU.

But is the image computed on the GPU correct?
```python
np.allclose(convolved_image_cpu, convolved_image_gpu)
```

In this case it is.

We cannot pass CuPy arrays to NumPy, the following code does not work.
```python
convolve2d_cpu(diracs_gpu, gauss_gpu)
```

But there are ways to combine NumPy and CuPy.
```python
diracs_1d_cpu = diracs.ravel()
gauss_1d_cpu = gauss.diagonal()

%time np.convolve(diracs_1d_cpu, gauss_1d_cpu)

diracs_1d_gpu = cp.asarray(diracs_1d_cpu)
gauss_1d_gpu = cp.asarray(gauss_1d_cpu)

%time np.convolve(diracs_1d_gpu, gauss_1d_gpu)

results_gpu = np.convolve(diracs_1d_gpu, gauss_1d_gpu)
type(result_gpu)
```

### Using PyTorch to accelerate your code

First we import PyTorch and check if we can see the GPU.
```python
import torch

print("GPU available: ", torch.cuda.is_available())

device = torch.device("cuda:0")

name = torch.cuda.get_device_name(device)
print("GPU detected: ", name)
```

We can create tensors on the device, in our case the GPU.
```python
x = torch.tensor([1, 2, 3, 4], device=device)
print(x.numel())
print(x.dtype)
```

We can also convert NumPy arrays to Torch tensors, and viceversa.
```python
import numpy as np

# NumPy array
a = np.array([1, 2, 3])
b = torch.from_numpy(a)

# Operations on the GPU
c = b.to(device)
print(c)

# Back to the CPU and NumPy
d c.cpu()
e = d.numpy()
```

More ways to create and initialize tensors on the GPU with PyTorch.
```python
a = torch.zeros(10, device=device)
b = torch.ones(10, device=device)
c = torch.rand(10, device=device)
d = torch.linspace(0, 1, 10, device=device)
```

By default Torch allocates tensors on the CPU, this is why we pass the `device` specifier.

Tensors can be multidimensional.
```python
x = torch.rand(3, 3, device=device)
print(x)

y = torch.rand(3, 3, 3, device=device)
print(y)

print(x[0, 0])
print(x[0,:])
print(x[:,0])
```

We can also perform operations on the GPU.
```python
x = torch.rand(5, device=device)
y = torch.rand(5, device=device)

print(x + y)
print(x * y)
print(torch.sqrt(x))
```

The following code produces an error, because it is not possible to have operations mixing tensors and NumPy arrays.
```python
x = torch.rand(5, device=device)
y = np.arange(5)

print(x + y)
```

An example of gravitational forces between a star and its planets.
```python
# Initializing position and mass for the star and planets
n = 10
position = torch.rand((2, n), device=device)
mass = torch.rand(n, device=device) + 100
sun_pos = [0.5, 0.5]
G = 6.67e-11

# Plotting
%matplotlib inline
import matplotlib.pyplot as plt

plt.axis("equal")
plt.scatter(position[0,:].cpu(), position[1,:].cpu(), s=mass.cpu())
plt.scatter(sun_pos[0], sun_pos[1], s=1000)
plt.show()
```

And now we can compute the forces between the celestial bodies.
```python
def calculate_forces(pos, mass):
    dx = sun_pos[0] - pos[0,:] # Row
    dy = sun_pos[1] - pos[1,:] # Row
    diff = torch.stack([dx, dy]) # Matrix
    dist = diff.norm(dim=0) # Row
    direction = diff / dist
    magnitude = G * mass / dist**2
    forces = magnitude * direction
    return forces

forces = calculate_forces(position, mass)

plt.axis("equal")
plt.scatter(position[0,:].cpu(), position[1,:].cpu(), s=mass.cpu())
plt.scatter(sun_pos[0], sun_pos[1], s=1000)
plt.quiver(position[0,:].cpu(), position[1,:].cpu(), forces[0,:].cpu(), forces[1,:].cpu())
plt.show()
```

As usual we can finish by measure the execution time of our GPU code.

```python
%load_ext cupyx.profiler

n = 1000
pos = torch.rand(2, n, device=device)
mass = torch.rand(n, device=device)
forces = calculate_forces(pos, mass)

%gpu_timeit calculate_forces(pos, mass)
```

Torch can also compile your code, and this is generally faster, but only applies to functions that uses tensors and operations on tensors.
```python
@torch.compile
def calculate_forces_faster(pos, mass):
    return calculate_forces(pos, mass)

n = 1000
pos = torch.rand(2, n, device=device)
mass = torch.rand(n, device=device)
forces = calculate_forces_faster(pos, mass)

%gpu_timeit calculate_forces_faster(pos, mass)
```

### Accelerate your Python code with Numba

Define an element-wise vector add.
```python
def vector_add(n, A, B):
    C = []
    for i in range(n):
        result = A[i] + B[i]
        C.append(result)
    return C
```

Generate some random data and check if the defined vector add is correct.
```python
import numpy as np

n = 1_000_000
A = np.random.rand(n)
B = np.random.rand(n)

C = vector_add(n, A, B)

print("Is correct: ", np.allclose(C, A + B))
%time vector_add(n, A, B);
```

Now we try to improve the performance on the CPU using Numba just-in-time compilation.
```python
import numba

@numba.jit
def vector_add_cpu(n, A, B):
    C = []
    for i in range(n):
        result = A[i] + B[i]
        C.append(result)
    return C

C = vector_add_cpu(n, A, B)
print("Is correct: ", np.allclose(C, A + B))
%time vector_add_cpu(n, A, B);
```

We can also use Numba to accelerate our own Python code on the GPU.

```python
import numba.cuda

print("GPU available: ", numba.cuda.is_available())
print("Which GPU: ", numba.cuda.get_current_device())

@numba.cuda.jit
def vector_add_gpu(n, A, B):
    C = []
    for i in range(n):
        result = A[i] + B[i]
        C.append(result)
    return C

vector_add_gpu[1, 1](n, A, B)
```

The previous code is not correct. We cannot use `append` on the GPU. This is a way to fix it.
```python
@numba.cuda.jit
def vector_add_gpu(n, A, B, C):
    for i in range(n):
        result = A[i] + B[i]
        C[i] = result

n = 1_000_000
A = np.random.rand(n)
B = np.random.rand(n)
C = np.zeros(n)
vector_add_gpu[1, 1](n, A, B, C)

%load_ext cupyx.profiler
%gpu_timeit -n 5 vector_add_gpu[1,1](n, A, B, C)
```

To speed-up the GPU implementation we need to change the code to expose more parallel work. We are now faster than the CPU.
```python
@numba.cuda.jit
def vector_add_parallel_gpu(n, A, B, C):
    i = numba.cuda.grid(1) # Gets the id of the thread
    C[i] = A[i] + B[i]
    
threads_per_block = 1000
nunber_of_blocks = 1000
n = threads_per_block * number_of_blocks

A = np.random.rand(n)
B = np.random.rand(n)
C = np.zeros(n)

vector_add_parallel_gpu[number_of_blocks, threads_per_block](n, A, B, C)

print("Is correct: ", np.allclose(C, A + B))

%gpu_timeit -n 5 vector_add_parallel_gpu()[number_of_blocks, threads_per_block](n, A, B, C)
```

Numba manages the transfers to and from the device for you. But it can also be manually managed.

```python
@numba.cuda.jit
def vector_add_parallel_gpu(n, A, B, C):
    i = numba.cuda.grid(1) # Gets the id of the thread
    C[i] = A[i] + B[i]
    
threads_per_block = 1000
nunber_of_blocks = 1000
n = threads_per_block * number_of_blocks

A = np.random.rand(n)
B = np.random.rand(n)
C = np.zeros(n)

A_gpu = numba.cuda.to_device(A)
B_gpu = numba.cuda.to_device(B)
C_gpu = numba.cuda.to_device(C)

vector_add_parallel_gpu[number_of_blocks, threads_per_block](n, A_gpu, B_gpu, C_gpu)

C = C_gpu.copy_to_host()

print("Is correct: ", np.allclose(C, A + B))

%gpu_timeit -n 5 vector_add_parallel_gpu()[number_of_blocks, threads_per_block](n, A_gpu, B_gpu, C_gpu)
```

Another example we can do with Numba is computing prime numbers. In this case the implementation is up to you.

```python
def find_primes(numbers):
    results = []
    for number in numbers:
        result = True
        for k in range(2, number):
            if number % k == 0:
                result = False
        results.append(result)
    return results

n = [3, 5, 10, 100]

print(find_primes(n))

numbers = np.arange(10_000)
%time results = find_primes(numbers)
```

### Introduction to CUDA/HIP
We will explain NVIDIA CUDA, but everything also applies to AMD's HIP. The GPU code is the same, the host functions only have different names (e.g. `hipMemcpy` vs `cudaMemcpy`).

```python
def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]
```

The first CUDA kernel, equivalent to the above Python vector add:
```cpp
__global__ void vector_add(const float * A, const float * B, float * C, const int size) {
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
```

The code still needs to be compiled for the GPU, this will be done using cuda-python and cupy.
```python
import cupy as cp
from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

# Initialize the GPU
gpu = Device()
gpu.set_current()
stream = gpu.create_stream()

# Set compiler options: the C++ standard, and the architecture of the GPU we are using
program_options = ProgramOptions(std="c++17", arch=f"sm_{gpu.arch}")

# Create data
size = 1024

a_gpu = cp.random.rand(size, dtype=cp.float32)
b_gpu = cp.random.rand(size, dtype=cp.float32)
c_gpu = cp.zeros(size, dtype=cp.float32)

# The CUDA code is given as a string here. Usually this would be a separate file, but in the course we want
# to keep the notebooks self-contained.
vector_add_cuda_code = r"""
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size) {
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
"""

# Compile the code
prog = Program(vector_add_cuda_code, code_type="c++", options=program_options)
mod = prog.compile("cubin", name_expressions=("vector_add", ))
vector_add_gpu = mod.get_kernel("vector_add")

# Set up the thread blocks
# grid = 3D set of blocks
# block = 3D set of threads
# For now we create one block of 1024 threads
config = LaunchConfig(grid=(1, 1, 1), block=(size, 1, 1))

# Finally, launch the GPU function. For each cupy array, we pass the raw pointer to CUDA
launch(stream, config, vector_add_gpu, a_gpu.data.ptr, b_gpu.data.ptr, c_gpu.data.ptr, size)
```

```python
import numpy as np

a_cpu = cp.asnumpy(a_gpu)
b_cpu = cp.asnumpy(b_gpu)
c_cpu = np.zeros(size, dtype=np.float32)

vector_add(a_cpu, b_cpu, c_cpu, size)

np.allclose(c_cpu, c_gpu)
```

Two main differences between the CUDA code and normal C++:
1. The `__global__` keyword means that the code will run on the GPU, but the function is visible to the CPU. There is also `__host__` for CPU-only code (this is added implicitly when you don't add anything yourself) and `__device__` for code that is visible on the GPU only.
2. The `threadIdx.x`. The `.x` means the first dimension. Thread blocks and grids are 3D, `.y` and `.z` access the other dimensions. `threadIdx` contains the ID of a thread within a block. 

Let's make the code work for any input size, instead of being limited to 1024 (i.e. the max number of threads in a block). Two parts of the code need to change:
The CUDA code needs to incorporate the potential existence of multiple blocks, and avoid writing out of bounds.
The python code needs to calculate the amount of threads per block and number of blocks.
Relevant CUDA keywords:
1. `threadIdx` = thread ID inside a block
2. `blockIdx` = block ID inside a grid
3. `blockDim` = dimension of a block (in terms of threads)
4. `gridDim` = dimension of the grid (in terms of blocks)
```cpp
# old code:
int item = threadIdx.x;
C[item] = A[item] + B[item];
# new code:
int item = blockIdx.x * blockDim.x + threadIdx.x;
if (item < size) {
    C[item] = A[item] + B[item];
}
```

```python
# old code:
config = LaunchConfig(grid=(1, 1, 1), block=(size, 1, 1))
# new code:
import math
threads_per_block = 1024
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)
config = LaunchConfig(grid=grid_size, block=block_size)
```

The input size can now be anything and the code will still be correct. 
Finally, we will time the CPU and GPU codes for a size of 10 million.

```python
%timeit -n 2 vector_add(a_cpu, b_cpu, c_cpu, size)
```
Takes 3 s.

```python
%timeit -n 20 launch(stream, config, vector_add_gpu, a_gpu.data.ptr, b_gpu.data.ptr, c_gpu.data.ptr, size)
```
Takes ~4 us. This is not correct! The GPU works asynchronously, we need to explicitly wait for the GPU to finish by synchronizing the stream:
```python
%timeit -n 20 launch(stream, config, vector_add_gpu, a_gpu.data.ptr, b_gpu.data.ptr, c_gpu.data.ptr, size); stream.sync()
```
Takes ~700 us.

## ❓ Questions

- How common is it to have a GPU that will speed up the code? Will it usually work on an average univeristy laptop/desktop, only a dedicated one, or does it only make sense if you have access to a mainframe or supercomputer?
    - Answer: GPUs on laptops or workstations can already speed-up the execution time of your code, even if they are not as capable as the one that we are using today on Snellius. Obviously the larger gains you get by using supercomputers.
-can we get access to your latest notebook Allessio? The introduciontoCUDA.ipynb?
    - Answer: everything in the notebook is also in this document. 

## 📚 Resources

* [How to install CuPy for AMD](https://docs.cupy.dev/en/stable/install.html#using-cupy-on-amd-gpu-experimental)
* [CuPy API](https://docs.cupy.dev/en/stable/reference/index.html)
* [How to install PyTorch for AMD](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)
* [PyTorch API](https://docs.pytorch.org/docs/stable/pytorch-api.html)
* [How to use Numba on AMD](https://numba.pydata.org/numba-doc/0.48.0/roc/index.html)
* [Numba documentation](https://numba.readthedocs.io/en/stable/index.html)
* [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
