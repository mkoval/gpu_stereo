CXX  = clang++
NVCC = nvcc
LD   = clang++
TARGET     = stereo
CXXFLAGS  := $(shell pkg-config --cflags opencv) -Wall -O3 -march=core2 -funroll-loops -Wno-system-headers
NVCCFLAGS :=
LDFLAGS   := $(shell pkg-config --libs opencv) -lopencv_gpu
OBJECTS    = stereo.cc.o util.cc.o

.PHONY: all clean
.SECONDARY:

all: $(TARGET)

clean:
	@echo "CLEAN"
	@$(RM) $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	@echo "LD $@"
	@$(LD) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

%.cc.o: %.cc
	@echo "CXX $@"
	@$(CXX) $(CXXFLAGS) -c -o $@ $^

%.cu.o: %.cu
	@echo "NVCC $@"
	@$(NVCC) $(NVCCFLAGS) -c -o $@ $^

# vim: noet ts=4 sw=4
