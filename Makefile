CXXFLAGS = -fopenmp -std=c++0x -O3 -Wall -DBOOST_UBLAS_NDEBUG

OBJS =		main.o MNIST.o

LIBS = -lgomp

TARGET =	War


$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)


all: $(TARGET) $(TESTS)

clean:
	rm -f $(OBJS) $(TARGET)
