# Builds the shim binary bdrs_shim against the upstream DROT headers in ../src.
# The target is the visible GPU's compute capability (sm_80 on an A100), or
# sm_80 when nvidia-smi reports none; NVCC_ARCH overrides it.

NVCC      ?= nvcc
GPU_ARCH  := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
               | head -1 | tr -d '. \r' | grep -E '^[0-9]+$$')
ifeq ($(GPU_ARCH),)
GPU_ARCH  := 80
endif
NVCC_ARCH ?= -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
NVCC_OPT  ?= -O3
INC       ?= -I ../src
OUT       ?= bdrs_shim

.PHONY: all clean

all: $(OUT)

$(OUT): bdrs_shim.cu
	$(NVCC) $(NVCC_OPT) $(NVCC_ARCH) $(INC) -std=c++17 -o $@ $<

clean:
	$(RM) $(OUT)
