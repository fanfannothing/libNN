CXXFLAGS = -Wall -std=c++0x -O3 -DBOOST_UBLAS_NDEBUG

OBJS =		main.o MNIST.o

LIBS =

TARGET =	War


$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)


all: $(TARGET) $(TESTS)

clean:
	rm -f $(OBJS) $(TARGET)
