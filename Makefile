TARGET  = stereo
CFLAGS  = `pkg-config --cflags opencv` -Wall -Wno-system-headers
LDFLAGS = `pkg-config --libs opencv` -lopencv_gpu
OBJECTS = stereo.cc.o

.PHONY: all clean
.SECONDARY:

all: $(TARGET)

clean:
	$(RM) $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $^

%.cc.o: %.cc
	$(CXX) $(CFLAGS) -c -o $@ $^
