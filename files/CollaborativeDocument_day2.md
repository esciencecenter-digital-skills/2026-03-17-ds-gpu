![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document (day 2)

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

### Challenge: Compute prime numbers with CUDA

Given the following Python code, similar to what we have seen in the previous episode about Numba, write the missing CUDA kernel that computes all the prime numbers up to a certain upper bound.

```python
import math
import numpy as np
import cupy as cp
from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

# Initialize the GPU
gpu = Device()
gpu.set_current()
stream = gpu.create_stream()
program_options = ProgramOptions(std="c++17", arch=f"sm_{gpu.arch}")

# CPU version
def all_primes_to(upper : int, prime_list : list):
    for num in range(0, upper):
        prime = True
        for i in range(2, (num // 2) + 1):
            if (num % i) == 0:
                prime = False
                break
        if prime:
            prime_list[num] = 1

upper_bound = 100_000
all_primes_cpu = np.zeros(upper_bound, dtype=np.int32)

# GPU version
check_prime_gpu_code = r'''
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
    // Write the CUDA code
}
'''

# Allocate memory
all_primes_gpu = cp.zeros(upper_bound, dtype=cp.int32)

# Compile the CUDA code
prog = Program(check_prime_gpu_code, code_type="c++", options=program_options)
mod = prog.compile("cubin", name_expressions=("all_primes_to",))
all_primes_to_gpu = mod.get_kernel("all_primes_to")

# Setup the grid
grid_size = (int(math.ceil(upper_bound / 1024)), 1, 1)
block_size = (1024, 1, 1)
config = LaunchConfig(grid=grid_size, block=block_size)

# Benchmark and test
%timeit -n 1 -r 1 all_primes_to(upper_bound, all_primes_cpu)
%timeit launch(stream, config, all_primes_to_gpu, upper_bound, all_primes_gpu.data.ptr); stream.sync()

if np.allclose(all_primes_cpu, all_primes_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
```

There is no need to modify anything in the code, except writing the body of the CUDA `all_primes_to` inside the `check_prime_gpu_code` string, as we did in the examples so far.

Possible answer for the CUDA kernel:

```cpp
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
  int item = (blockIdx.x * blockDim.x) + threadIdx.x;
  
  all_primbe_numbers[item] = 1;
  if (item < size) {
      for (int i = 2; i < item / 2 + 1; i++ ) {
          if (item % i == 0) {
              all_prime_numbers[item] = 0;
              break;
          }
      } 
  }
}
```



## 🧠 Collaborative Notes

**Question**: Can GPUs only do 32-bit float (single precision)? Or can they also do 64-bit floating-point numbers (double precision)? 
**Answer**: Yes. GPUs can do both 32-bit and 64-bit numbers. However, 32-bit numbers are faster on GPUs. There are two kinds of GPUs: regular GPUs for consumers and expensive GPUs for servers and supercomputers. The regular GPUs are slow for double precision (typically float is 16x faster than double). The server GPUs are faster for double precision, but still a bit slower than single precision (typically float is 2x faster than double).

**Question**: Can I include header files? (Like #include <math.h>)
**Answer**: Yes! However, you do need to setup where the NVIDIA compiler must search to find the header files. Unfortunately, this is not set up by default. 

### Memories on the GPU
GPUs have multiple memories, as opposed to CPUs. For example, in our prime example, there are three memories: one two store the prime numbers (`all_prime_numbers`), one two store the variables (`int item; int i;`), and one to store the kernel arguments (`int size`).

#### Global memory
In Python, we can use `cupy` (for example) to create global GPU memory. For example, the function `cupy.zeros(n)` will put the data in the global GPU memory. 

Global memory is not "_coherent_". This means that if one thread writes to global memory and another thread tries to read the same location from global memory immediately after, it might see the old value or the new value of global memory. Only after the computation finishes (when the kernel is done), then the final values are in global memory.

#### Registers

Every variable in a kernel code is put into "_registers_". For example, things like `int item;` and `int i;`. Each thread has its own local private registers. Threads cannot see the registers of other threads, only their own registers. You do not need manually assign registers to your code, the CUDA compiler automatically decides how to put your variables into registers. 

Typically you do not need to think about registers. However, the compiler might give an error "too many registers used" if you kernel code is big and contains too many variables than registers are available.


#### Constant memory
So where are kernel arguments stored? (like `size` in `__global__ void kernel(int size)`).

There is a special memory used to store these kernel arguments. This is called "constant memory". It is used to put the parameters of your kernel call. This is done automatically by the compiler.

You can also manually create constant memory, but this feature is rarely used. Nowadays it is only used to store the kernel arguments that are sent from the CPU (which submit the GPU kernel) to GPU (which actually runs the GPU threads).



#### Shared Memory

The next section will be on exchange values between threads in a thread block using _shared memory_.


### Histogram example

First, we define our **histogram** function. This counts how often each value in `input_array` occurs in `output_array`. For example, if `output_array[5] == 10` then `5` occurs `10` times in `input_array`.

```python
def histogram(input_array, output_array):
    for item in input_array:
        output_array[item] = output_array + 1
```

We can use your function as follows:

```python
import numpy as np

size = 2048

input_array = np.random.randint(256, size=size, dtype=np.int32)
output_array = np.zeros(256, dtype=np.int32)

histogram(input_array, output_array)
```

This could be a possible corresponding CUDA kernel.

```cpp
__global__ void histogram(const int * input, int * const output, const int size) {
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (item < size) {
        int value = input[item];
        output[value] = output[value] + 1;  // Warning: is this line ok?
    }
}
```

But the above code is incorrect! Two threads might read `output[value]` at exactly the same time, read the same value (for example, they both read a value of `5`), both increment this by one (`5` -> `6`), and they both write the same result (they both write `6`). In this case, the results will be wrong as you will read `6` but the value should be `7`!

We can solve this using `atomicAdd`, which performs this operations in one instruction (read the old value, increment the value, write the new value). If two threads call `atomicAdd` on the same variable at the same time, then the GPU will make sure that one thread performs its update before the other thread.


```cpp
__global__ void histogram(const int * input, int * const output, const int size) {
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (item < size) {
        int value = input[item];
        //output[value] = output[value] + 1; // <--- This is incorrect :-(!
        atomicAdd(&output[value], 1);   // <--- This is correct :-)!
    }
}
```

This code is correct, but it might be slow. This is because `atomicAdd` is slower than just a regular global memory read/write. This is especially a problem if all threads call `atomicAdd` exactly at the same time, since the GPU has to make sure that all threads can perform their update one-by-one, which is slow. 

Instead, what we can do is used something call _shared memory_. All threads in the same block can see the content of the shared memory. We can do the `atomicAdd` on shared memory first and then on global memory (two steps), instead of directly on global memory. 


```cpp
__global__ void histogram(const int * input, int * const output, const int size) {
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];
    
    if (item < size) {
        int value = input[item];
        
        atomicAdd(&(temp_histogram[value]), 1);
        __syncthreads();
        atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);   
    }
}
```

This code is still wrong! The `temp_histogram` is not set to zero when the thread block starts. We need to to set `temp_histogram` to zero before we do the `atomicAdd` on `temp_histogram`.



```cpp
__global__ void histogram(const int * input, int * const output, const int size) {
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];
    
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();  // Wait until all threads have zeroed `temp_histogram`
    
    if (item < size) {
        int value = input[item];
        atomicAdd(&(temp_histogram[value]), 1);
    }
    
    __syncthreads(); // Wait until everybody has done `atomicAdd` on `temp_histogram`
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);   
}
```

Note that we cannot put a `__syncthreads` inside the `if` statement. If we would do this, the threads that do take the `if` statement will wait for the other to join, but the other never join. This leads to problems on the GPU. 

### Streams

Copy this code for the next part.

```python
import numpy as np
import cupy as cp
import math
from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

upper_bound = 10_000_000
histogram_size = 2**27

# Initialize the GPU
gpu = Device()
gpu.set_current()
stream = gpu.create_stream()
program_options = ProgramOptions(std="c++17", arch=f"sm_{gpu.arch}")

# GPU code
check_prime_gpu_code = r'''
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
    int number = (blockIdx.x * blockDim.x) + threadIdx.x;
    int result = 1;

    if ( number < size )
    {
        for ( int factor = 2; factor <= number / 2; factor++ )
        {
            if ( number % factor == 0 )
            {
                result = 0;
                break;
            }
        }

        all_prime_numbers[number] = result;
    }
}
'''
histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];
 
    // Initialize shared memory and synchronize
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();

    // Compute shared memory histogram and synchronize
    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();

    // Update global histogram
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
'''

# Allocate memory
all_primes_gpu = cp.zeros(upper_bound, dtype=cp.int32)
input_gpu = cp.random.randint(256, size=histogram_size, dtype=cp.int32)
output_gpu = cp.zeros(256, dtype=cp.int32)

# Compile and setup the grid for all_primes_to
prog = Program(check_prime_gpu_code, code_type="c++", options=program_options)
mod = prog.compile("cubin", name_expressions=("all_primes_to",))
all_primes_to_gpu = mod.get_kernel("all_primes_to")
grid_size = (int(math.ceil(upper_bound / 1024)), 1, 1)
block_size = (1024, 1, 1)
config = LaunchConfig(grid=grid_size, block=block_size)

# Compile and setup the grid for histogram
prog = Program(histogram_cuda_code, code_type="c++", options=program_options)
mod = prog.compile("cubin", name_expressions=("histogram",))
histogram_gpu = mod.get_kernel("histogram")
grid_size = (int(math.ceil(histogram_size / 256)), 1, 1)
block_size = (256, 1, 1)
config = LaunchConfig(grid=grid_size, block=block_size)

# Execute the kernels
launch(stream, config, all_primes_to_gpu, upper_bound, all_primes_gpu.data.ptr)
launch(stream, config, histogram_gpu, input_gpu.data.ptr, output_gpu.data.ptr)
stream.sync()

# Save results to do something else later
output_one = all_primes_gpu
output_two = output_gpu
```


We will make some changes to this code to make in parallel:

```python
import numpy as np
import cupy as cp
import math
from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

upper_bound = 10_000_000
histogram_size = 2**27

# Initialize the GPU
gpu = Device()
gpu.set_current()
stream_one = gpu.create_stream()  // <--- modified 
stream_two = gpu.create_stream()  // <--- modified
program_options = ProgramOptions(std="c++17", arch=f"sm_{gpu.arch}")

# GPU code
check_prime_gpu_code = r'''
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
    int number = (blockIdx.x * blockDim.x) + threadIdx.x;
    int result = 1;

    if ( number < size )
    {
        for ( int factor = 2; factor <= number / 2; factor++ )
        {
            if ( number % factor == 0 )
            {
                result = 0;
                break;
            }
        }

        all_prime_numbers[number] = result;
    }
}
'''
histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];
 
    // Initialize shared memory and synchronize
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();

    // Compute shared memory histogram and synchronize
    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();

    // Update global histogram
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
'''

# Allocate memory
all_primes_gpu = cp.zeros(upper_bound, dtype=cp.int32)
input_gpu = cp.random.randint(256, size=histogram_size, dtype=cp.int32)
output_gpu = cp.zeros(256, dtype=cp.int32)

# Compile and setup the grid for all_primes_to
prog = Program(check_prime_gpu_code, code_type="c++", options=program_options)
mod = prog.compile("cubin", name_expressions=("all_primes_to",))
all_primes_to_gpu = mod.get_kernel("all_primes_to")
grid_size = (int(math.ceil(upper_bound / 1024)), 1, 1)
block_size = (1024, 1, 1)
config = LaunchConfig(grid=grid_size, block=block_size)

# Compile and setup the grid for histogram
prog = Program(histogram_cuda_code, code_type="c++", options=program_options)
mod = prog.compile("cubin", name_expressions=("histogram",))
histogram_gpu = mod.get_kernel("histogram")
grid_size = (int(math.ceil(histogram_size / 256)), 1, 1)
block_size = (256, 1, 1)
config = LaunchConfig(grid=grid_size, block=block_size)

interesting_point = gpu.create_event()  // <-- modified

# Execute the kernels
launch(stream_one, config, all_primes_to_gpu, upper_bound, all_primes_gpu.data.ptr) // <--- modified
stream_one.record(interesting_point) // <-- modified
launch(stream_one, config, all_primes_to_gpu, upper_bound, all_primes_gpu.data.ptr) // <--- modified

stream_two.wait(interesting_point)
launch(stream_two, config, histogram_gpu, input_gpu.data.ptr, output_gpu.data.ptr) // <--- modified
stream.sync()

# Save results to do something else later
stream_one.sync() // <--- modified
output_one = all_primes_gpu
stream_two() // <--- modified
output_two = output_gpu
```

### Auto-tuning
How should one pick parameters such as the number of threads per block?
This is very application-specific, as well as gpu-specific. There are tools that can help you.
We will look at one such tool: Kernel Tuner.

As an application, we will look at matrix-matrix multiplication.

```python
import numpy as np

n = 1024
A = np.random.rand(n, n).astype("float32")
B = np.random.rand(n, n).astype("float32")
C = np.matmul(A, B)

%time np.matmul(A, B);
```

Now we will do this in cupy
```python
%load_ext cupyx.profiler

import cupy

A_gpu = cupy.array(A)
B_gpu = cupy.array(B)
C_gpu = cupy.matmul(A, B)

%gpu_timeit cupy.matmul(A_gpu, B_gpu)
```

Now we will write our own CUDA kernel that does a matmul, and let's see if we can beat cupy!
We give you the kernel code:
```cpp
kernel_code = r"""
__global__ void matmul(int n, float* A, float* B, float* C) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    float sum = 0;
    for (int k = 0; k < n; k++) {
      sum += A[i * n + k] * B[k * n + j];
    }
    C[i * n + j] += sum;
  }
}
"""
```

We will use Kernel Tuner to decide the optimal block size. For this, some things need to be set up first.

```python
kernel_name = "matmul"
problem_size = (n, n)  # total number of threads we want in the x and y dimension
arguments = [np.int32(n), A, B, np.zeros_like(C)]

# the answer contains the expected result for each input. Non-relevant inputs should be set to None
answer = [None, None, None, C]

# What values we want to tune
tune_params = dict()
tune_params["block_size_x"] = [32, 64]
tune_params["block_size_y"] = [1, 2, 4]

import kernel_tuner

kernel_tuner.tune_kernel(
    kernel_name,
    kernel_code,
    problem_size,
    arguments,
    tune_params,
    answer=answer,
    lang="cuda"
);
```

By default, Kernel Tuner will try all combinations of tuning parameters, and tell you the runtime of each combination.

Exercise: find the optimal block size.

Now we will see if we can improve the kernel itself, by adding a few more options and making them tunable.
First, we will add loop unrolling.

```python
kernel_code = r"""
__global__ void matmul(int n, float* A, float* B, float* C) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    float sum = 0;
#pragma unroll loop_unroll_factor
    for (int k = 0; k < n; k++) {
      sum += A[i * n + k] * B[k * n + j];
    }
    C[i * n + j] += sum;
  }
}
"""

kernel_name = "matmul"
problem_size = (n, n)  # total number of threads we want in the x and y dimension
arguments = [np.int32(n), A, B, np.zeros_like(C)]

# the answer contains the expected result for each input. Non-relevant inputs should be set to None
answer = [None, None, None, C]

# What values we want to tune
tune_params = dict()
tune_params["block_size_x"] = [8, 32]
tune_params["block_size_y"] = [8, 32]
tune_params["loop_unroll_factor"] = [1, 2]

import kernel_tuner

kernel_tuner.tune_kernel(
    kernel_name,
    kernel_code,
    problem_size,
    arguments,
    tune_params,
    answer=answer,
    lang="cuda"
);
```

Exercise: Try a range of loop unroll factors.

Now we will force the compiler to use the GPU cache.
```python
kernel_code = r"""
__global__ void matmul(int n, float* A, float* B, float* C) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    float sum = 0;
#pragma unroll loop_unroll_factor
    for (int k = 0; k < n; k++) {
#if use_ldg
      sum += __ldg(&A[i * n + k]) * __ldg(&B[k * n + j]);
#else
      sum += A[i * n + k] * B[k * n + j];
#endif
    }
    C[i * n + j] += sum;
  }
}
"""

kernel_name = "matmul"
problem_size = (n, n)  # total number of threads we want in the x and y dimension
arguments = [np.int32(n), A, B, np.zeros_like(C)]

# the answer contains the expected result for each input. Non-relevant inputs should be set to None
answer = [None, None, None, C]

# What values we want to tune
tune_params = dict()
tune_params["block_size_x"] = [16, 32, 64]
tune_params["block_size_y"] = [2, 4, 8]
tune_params["loop_unroll_factor"] = [1, 10, 250, 1000]
tune_params["use_ldg"] = [0, 1]

import kernel_tuner

kernel_tuner.tune_kernel(
    kernel_name,
    kernel_code,
    problem_size,
    arguments,
    tune_params,
    answer=answer,
    lang="cuda"
);
```

The search space can grow very quickly. Kernel Tuner supports multiple search algorithms
```python
kernel_code = r"""
__global__ void matmul(int n, float* A, float* B, float* C) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    float sum = 0;
#pragma unroll loop_unroll_factor
    for (int k = 0; k < n; k++) {
#if use_ldg
      sum += __ldg(&A[i * n + k]) * __ldg(&B[k * n + j]);
#else
      sum += A[i * n + k] * B[k * n + j];
#endif
    }
    C[i * n + j] += sum;
  }
}
"""

kernel_name = "matmul"
problem_size = (n, n)  # total number of threads we want in the x and y dimension
arguments = [np.int32(n), A, B, np.zeros_like(C)]

# the answer contains the expected result for each input. Non-relevant inputs should be set to None
answer = [None, None, None, C]

# What values we want to tune
tune_params = dict()
tune_params["block_size_x"] = [16, 32, 64]
tune_params["block_size_y"] = [2, 4, 8]
tune_params["loop_unroll_factor"] = [1, 10, 250, 1000]
tune_params["use_ldg"] = [0, 1]

import kernel_tuner

kernel_tuner.tune_kernel(
    kernel_name,
    kernel_code,
    problem_size,
    arguments,
    tune_params,
    answer=answer,
    lang="cuda",
    strategy="genetic_algorithm"
);
```

### Energy Efficiency

Tunable code.
```python
import numpy as np
import kernel_tuner

kernel_name = "matmul"
kernel_code = """
__global__ void matmul(int n, float* A, float* B, float* C) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    float sum = 0;
    for (int k = 0; k < n; k++) {
      sum += A[i * n + k] * B[k * n + j];
    }
    C[i * n + j] = sum;
  }
}
"""

tune_params = dict()
tune_params["block_size_x"] = [8, 16, 32, 64]
tune_params["block_size_y"] = [1, 2, 4, 8, 16, 32]

n = 1024
A = np.random.rand(n, n).astype("float32")
B = np.random.rand(n, n).astype("float32")
C = np.matmul(A, B)

problem_size = (n, n)
arguments = [np.int32(n), A, B, np.zeros_like(C)]
answer = [None, None, None, C]

# Run the tuner!
kernel_tuner.tune_kernel(
    kernel_name,
    kernel_code,
    problem_size,
    arguments,
    tune_params,
    answer=answer,
    lang="cuda",
);
```
So far we tuned for performance, faster is better, and we looked at execution time. But execution time may be misleading.
A fairer measurement is the number of operations that the GPU is executing per second. 
FLOP/s is the number of floating point operations executed per second.

The number of operations for the matrix multiply is the following.
```python
gflop = 2 * n * n * n * 1e-9
```

This function computes the number of GFLOP/s of the matrix multiply.
```python
def get_gflops(result):
    return gflop / (result["time"] * 1e-3)
```

And now we can pass this metric to Kernel Tuner, so we can have GFLOP/s added to the results.
```python
metrics = dict()
metrics["GFLOP/s"] = get_gflops

kernel_tuner.tune_kernel(
    kernel_name,
    kernel_code,
    problem_size,
    arguments,
    tune_params,
    answer=answer,
    lang="cuda",
    metrics=metrics,
);
```

In modern GPUs there is a sensor that continuously monitors the power, for Nvidia GPUs we can measure the power in the following way, and add these values to Kernel Tuner.
```python
from kernel_tuner.observers.nvml import NVMLObserver

observer = NVMLObserver(["nvml_power", "nvml_energy"])

def get_power(result):
    return result["nvml_power"]

metrics["Power"] = get_power

kernel_tuner.tune_kernel(
    kernel_name,
    kernel_code,
    problem_size,
    arguments,
    tune_params,
    answer=answer,
    lang="cuda",
    metrics=metrics,
    observers=[observer],
);
```

Now that we can measure power, we can also compute energy efficiency in operations per second per watt: GFLOP/J.
```python
def get_efficiency(result):
    return gflop / result["nvml_energy"]

metrics["GFLOP/J"] = get_efficiency

kernel_tuner.tune_kernel(
    kernel_name,
    kernel_code,
    problem_size,
    arguments,
    tune_params,
    answer=answer,
    lang="cuda",
    metrics=metrics,
    observers=[observer],
);
```

We can plot the results and look at the performance/efficiency correlation.
```python
import matplotlib.pyplot as plt
%matplotlib inline

results, env = kernel_tuner.tune_kernel(
    kernel_name,
    kernel_code,
    problem_size,
    arguments,
    tune_params,
    answer=answer,
    lang="cuda",
    metrics=metrics,
    observers=[observer],
);

performance = [result["GFLOP/s"] for result in results]
efficiency = [result["GFLOP/J"] for result in results]

plt.scatter(efficiency, performance)
plt.xlabel("GFLOP/J")
plt.ylabel("GFLOP/s")
plt.show()
```

## Questions

-  Which memory stores blockIdx, blockDim, threadIdx values? How fast is their access?
    - You can assume these are stored in "special" registers with fast access. In old CUDA versions, `threadIdx` was stored in local registers and `blockIdx`, `blockDim` etc. were stored in shared memory.
- How do you debug CUDA code? I guess you can't print from within a thread? So say I want to see if the modulo division gives expected results to calculate the primes. Can I see that in any way?
    - You can actually use `printf` in your kernel! https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#formatted-output
    - However, note that all threads print their output at the same time then you might get thousands of lines of output. Typically, you would use something like `if (myitem===0) printf(...);` to limit the output.
    - Additionally, you can also use the CUDA debugger (called `cuda-gdb`): https://docs.nvidia.com/cuda/cuda-gdb/index.html

## 📚 Resources

* [Kernel Float: Unlocking Mixed-Precision GPU Programming](https://dl.acm.org/doi/10.1145/3779120)
* [CUDA Python documentation](https://nvidia.github.io/cuda-python/cuda-core/latest/index.html)
* [HIP Python documentation](https://rocm.docs.amd.com/projects/hip-python/en/latest/)
* [Kernel Tuner documentation](https://kerneltuner.github.io/kernel_tuner/stable/)
* [Kernel Tuner tutorials](https://github.com/KernelTuner/kernel_tuner_tutorial)
* [Tuning the Tuner: Introducing Hyperparameter Optimization for Auto-Tuning](https://ieeexplore.ieee.org/document/11181511)
* Power Measurement Tookit (PMT): https://git.astron.nl/RD/pmt
