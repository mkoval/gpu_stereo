CXX  = clang++
NVCC = nvcc
LD   = clang++
TARGET     = stereo
CXXFLAGS  := $(shell pkg-config --cflags opencv) -I/usr/local/cuda/include -Wall -O3
NVCCFLAGS := -O3 --machine=64
LDFLAGS   := $(shell pkg-config --libs opencv) -lopencv_gpu -L/usr/local/cuda/lib -lcuda -lcudart
OBJECTS    = stereo.cpp.o bm_cpu.cpp.o bm_gpu.cu.o

MAKEFLAGS += -R

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
