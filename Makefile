CXX  = clang++
NVCC = nvcc
LD   = clang++
CUDA_PATH  = /usr/local/cuda
TARGET     = stereo
CXXFLAGS  := $(shell pkg-config --cflags opencv) -I$(CUDA_PATH)/include -pedantic -Wall -O3 -flto
NVCCFLAGS := -O3 --machine=64
LDFLAGS   := $(CXXFLAGS) -L$(CUDA_PATH)/lib $(shell pkg-config --libs opencv) -lopencv_gpu -lcuda -lcudart
OBJECTS    = stereo.cpp.o bm_cpu.cpp.o bm_gpu.cu.o

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

%.cu.o: %.cu Makefile
	@echo "NVCC $@"
	@$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# vim: noet ts=4 sw=4
