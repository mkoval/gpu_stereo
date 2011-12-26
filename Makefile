CXX  = g++
NVCC = nvcc
LD   = g++
CUDA_PATH  = /usr/local/cuda
TARGET     = stereo
CXXFLAGS  := $(shell pkg-config --cflags opencv) -I$(CUDA_PATH)/include -pedantic -Wall -O3 -fopenmp
NVCCFLAGS := -O3
LDFLAGS   := $(CXXFLAGS) -L$(CUDA_PATH)/lib $(shell pkg-config --libs opencv) -lm -lopencv_gpu -lcuda -lcudart
OBJECTS    = stereo.cpp.o util.cpp.o bm_cpu.cpp.o bm_gpu.cpp.o bm_gpu.cu.o

ifeq ($(shell uname), Darwin)
LDFLAGS   += -Xlinker -rpath /usr/local/cuda/lib -flto
NVCCFLAGS += --machine=64
endif

.PHONY: all clean
.SECONDARY:

all: $(TARGET)

clean:
	@echo "CLEAN"
	@$(RM) $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	@echo "LD $@"
	@$(LD) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

%.cpp.o: %.cpp Makefile
	@echo "CXX $@"
	@$(CXX) $(CXXFLAGS) -c -o $@ $<

# Override make's implicit %: %.o rule.
%.cu:
%.cu.o: %.cu Makefile
	@echo "NVCC $@"
	@$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# vim: noet ts=4 sw=4
