CXXFLAGS = -fopenmp -std=c++0x -O3 -Wall -DBOOST_UBLAS_NDEBUG

NVCC = nvcc
NVCCFLAGS = --gpu-architecture=compute_20 --compiler-options "${CXXFLAGS}"

OBJS =		main.o MNIST.o

LIBS = -lgomp

TARGET =	War


$(TARGET): $(OBJS)
	$(NVCC) ${NVCCFLAGS} -o $(TARGET) $(OBJS) $(LIBS)

%.o: %.cpp
	${NVCC} ${NVCCFLAGS} -c $< -o $@

all: $(TARGET) $(TESTS)

clean:
	rm -f $(OBJS) $(TARGET)
