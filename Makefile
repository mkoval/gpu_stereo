CXX  = clang++
NVCC = nvcc
LD   = clang++
TARGET     = stereo
CXXFLAGS  := $(shell pkg-config --cflags opencv) -Wall -O3 -march=core2 -flto
NVCCFLAGS :=
LDFLAGS   := $(shell pkg-config --libs opencv) -lopencv_gpu
OBJECTS    = stereo.cpp.o bm_cpu.cpp.o

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
