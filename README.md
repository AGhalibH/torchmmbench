# torchmmbench

A simple PyTorch benchmark script to measure **Matrix Multiplication performance (TFLOPS)** across different devices:

* CPU
* GPU (NVIDIA/AMD/INTEL ARC)

It evaluates throughput using large matrix multiplications and reports performance across multiple data types.

---

## Features

* Benchmark **NxN matrix multiplication**
* Supports:

  * `float32`
  * `float16`
  * `bfloat16` (if the GPU supported)
* Works on:

  * CPU
  * CUDA (NVIDIA GPUs)
  * ROCm (AMD GPUs)
  * Intel XPU (INTEL ARC GPUs)
* Reports:

  * Average execution time (ms)
  * Throughput (TFLOPS)
* Configurable:

  * Matrix size
  * Iterations
  * Warmup runs
  * CPU threads

---

## How It Works

The benchmark computes:

```
C = A × B
```

Where:

* `A` and `B` are randomly generated NxN matrices

Performance is calculated using:

```
FLOPs = 2 × N³
TFLOPS = FLOPs / time
```

---

## Installation

### 1. Clone the repository
Make sure you have [Git](https://git-scm.com/) installed:

```bash
git clone https://github.com/AGhalibH/torchmmbench.git
cd torchmmbench
```

### 2. Install dependencies

Make sure you have [UV](https://github.com/astral-sh/uv) installed:

```bash
# Create Python Virtual Environment with UV
uv venv --python=3.12 --seed --clear ./venv

# Activate the Virtual Environment
source ./venv/bin/activate     # For Linux
call ./venv/Scripts/activate   # For Windows

# Install Pytorch based on your Hardware
uv pip install torch torchvision torchaudio
```

> ⚠️ Install the correct PyTorch build depending on your hardware (CUDA / ROCm / XPU / CPU)

| Device Type | Pytorch Setup  |
| ----------- | -------------- |
| CPU         | [Windows / Linux](https://pytorch.org/get-started/locally/) |
| NVIDIA GPU  | [Windows / Linux](https://pytorch.org/get-started/locally/) |
| AMD GPU     | [Windows](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) / [Linux](https://pytorch.org/get-started/locally/) |
| Intel GPU   | [Windows / Linux](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html) |

---

## Usage

```bash
python torchmmbench.py [OPTIONS]
```

### Arguments

| Argument    | Description                              |
| ----------- | ---------------------------------------- |
| `--device`  | `cpu`, `cuda`, or `xpu` (default: `cpu`) |
| `--size`    | Matrix size N (NxN) (default: `8192`)    |
| `--iters`   | Benchmark iterations (default: `10`)     |
| `--warmup`  | Warmup iterations (default: `3`)         |
| `--threads` | CPU threads (CPU only)                   |

---

## Example

```bash
python torchmmbench.py --device cuda --size 8192 --iters 10
```

### Sample Output

```
=======================================================
PyTorch MatMul (Matrix Multiplication) Benchmark
=======================================================
Configuration:
  OS          : Linux
  Device      : cuda
  Device Name : AMD Radeon RX 6600 XT | gfx1030
  PyTorch Ver : 2.10.0+rocm7.12.0a20260303
  Matrix Size : 8192 x 8192
  Iterations  : 10
  Warmup      : 3
=======================================================

Results:
=======================================================
| DTYPE      | AVG TIME (ms)   | THROUGHPUT (TFLOPS)  |
=======================================================
| float32    | 129.23          | 8.51                 |
| float16    | 70.89           | 15.51                |
| bfloat16   | 256.37          | 4.29                 |
=======================================================
```

---

## Interpreting Results

* **Lower time (ms)** = faster execution
* **Higher TFLOPS** = better performance
* `float16` is usually fastest on GPUs due to tensor core acceleration
* `bfloat16` performance depends heavily on hardware support

---

## Tips for Accurate Benchmarking

* Close other heavy applications
* Use larger matrix sizes (e.g. 8192 or higher)
* Run multiple iterations
* Ensure proper GPU drivers are installed
* Use release builds of PyTorch (not debug)

---

## Supported Devices

| Device Type     | Backend        |
| --------------- | -------------- |
| CPU             | Native PyTorch |
| NVIDIA GPU      | CUDA           |
| AMD GPU         | ROCm           |
| INTEL ARC GPU   | XPU            |

---

## Future Improvements

* [ ] Add multi-GPU benchmarking
* [ ] Add CSV/JSON export
* [ ] Add plotting (performance graphs)
* [ ] Add mixed precision benchmarking
* [ ] Add batch GEMM support

---

## Contributing

Contributions are welcome!

Feel free to:

* Open issues
* Submit pull requests
* Suggest features

---

## License

MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

---
