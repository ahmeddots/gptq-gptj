# GPTQ

### The GPT-J implementation is still untested, so let me know if it works or if you run into errors!

This repository contains the code for the paper [GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://arxiv.org/abs/2210.17323). 
The current release includes the following features:

* An efficient implementation of the GPTQ algorithm: `gptq.py`
* Compressing all models from the OPT and BLOOM families to 2/3/4 bits, including weight grouping: `opt.py`, `bloom.py`, `gptj.py,` `zeroShot/`
* Evaluating the perplexity of quantized models on several language generation tasks: `opt.py`, `bloom.py`, `gpt-j.py`
* Evaluating the performance of quantized models on several ZeroShot tasks: `zeroShot/`
* A 3-bit quantized matrix full-precision vector product CUDA kernel: `quant_cuda_kernel.cu`, `quant_cuda.cpp`, `setup_cuda.py`
* Benchmarking code for individual matrix-vector products and for language generation with quantized models: `test_kernel.py`, `opt.py`

## Dependencies

* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.21.2
* `datasets`: tested on v1.17.0
* (to run 3-bit kernels: setup for compiling PyTorch CUDA extensions, see also https://pytorch.org/tutorials/advanced/cpp_extension.html, tested on CUDA 11.4)

All experiments were run on a single 80GB NVIDIA A100. However, most experiments will work on a GPU with a lot less memory as well

## Installation
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
git clone https://github.com/AlpinDale/gptq-gptj && cd gptq-gptj
pip install -r requirements.txt
```

## Language Generation

### GPT-J
```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python gptj.py EleutherAI/gpt-j-6b c4
# Run RTN baseline and compute results
CUDA_VISIBLE_DEVICES=0 python gptj.py EleutherAI/gpt-j-6b c4 --wbits 4 --nearest
# Run GPTQ and compute results
CUDA_VISIBLE_DEVICES=0 python gptj.py EleutherAI/gpt-j-6b c4 --wbits 4 [--groupsize 1024]
````

## CUDA Kernels

```
# Install kernels
python setup_cuda.py install

# Benchmark performance for for GPT-J 6B
CUDA_VISIBLE_DEVICES=0 python test_kernel.py

# Save compressed model
CUDA_VISIBLE_DEVICES=0 python gptj.py EleutherAI/gpt-j-6b c4 --wbits 4 --groupsize 128 --save gpt-j-6b-4bit.pt
# (Optionally) save as compressed `.safetensors` model
CUDA_VISIBLE_DEVICES=0 python gptj.py EleutherAI/gpt-j-6b c4 --wbits 4 --groupsize 128 --save_safetensors gpt-j-6b-4bit.safetensors

# Benchmark generating a 2048 token sequence with the saved model
CUDA_VISIBLE_DEVICES=0 python gptj.py EleutherAI/gpt-j-6b c4 --wbits 4 --groupsize 128 --load gpt-j-6b-4bit.pt --benchmark 2048 --check
# Benchmark FP16 baseline, note that the model will be split across all listed GPUs. Do only `0` if you have only one GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python gptj.py EleutherAI/gpt-j-6b c4 --benchmark 2048 --checl

# Inference with the saved model
CUDA_VISIBLE_DEVICES=0 python gptj-inference.py EleutherAI/gpt-j-6b --wbits 4 --groupsize 128 --load gpt-j-6b-4bit.pt --text "Hello Pygmalion!"
```

## ZeroShot

*Not implemented for GPT-J yet.*

See `zeroShot/` folder.



## Cite

If you found this work useful, please consider citing:

```
@article{frantar-gptq,
  title={{GPTQ}: Accurate Post-training Compression for Generative Pretrained Transformers}, 
  author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
  year={2022},
  journal={arXiv preprint arXiv:2210.17323}
}
```
