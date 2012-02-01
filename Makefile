CXXFLAGS = -fopenmp -std=c++0x -O3 -Wall -DBOOST_UBLAS_NDEBUG -DLIBNNCUDA -I/usr/local/cuda/include -g

NVCC = nvcc
NVCCFLAGS = --gpu-architecture=compute_20 -DLIBNNCUDA

OBJS =		main.o MNIST.o ActivactionFunctionTanh.o BackpropagationCU.o test_cuda.o

LIBS = -lgomp
NVLIBS = -lcublas

TARGET =	War


$(TARGET): $(OBJS)
	$(NVCC) ${NVCCFLAGS} -o $(TARGET) $(OBJS) $(LIBS) ${NVLIBS}

%.o: %.cpp
	${CXX} ${CXXFLAGS} -c $< -o $@

%.o: %.cu
	${NVCC} ${NVCCFLAGS} -c $< -o $@

all: $(TARGET) $(TESTS)

clean:
	rm -f $(OBJS) $(TARGET)
