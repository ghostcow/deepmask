CC=g++
CFLAGS=`pkg-config --cflags opencv`
LDFLAGS=
SOURCES=OpenCV_ViolaJones.cpp
LIBS=`pkg-config --libs opencv`
EXECUTABLE=Release/OpenCV_ViolaJones

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE):
	$(CC) $(CFLAGS) $(SOURCES) $< -o $@ $(LIBS)
