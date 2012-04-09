CXXFLAGS = -fopenmp -std=c++0x -O3 -Wall -DBOOST_UBLAS_NDEBUG -I/usr/local/cuda/include `libpng-config --cflags`

NVCC = nvcc
NVCCFLAGS = --gpu-architecture=compute_20

OBJS =		main.o DataSetMNIST.o DataSetLETOR.o DataSetStockFLRank.o DataSetStockFLRankCV.o ActivationFunctionTanh.o BackpropagationCU.o test_cuda.o test_RankNet.o test_LambdaRank.o test_BAM.o vivian.o gladys.o poi.o test_stock.o

LIBS = -lgomp `libpng-config --ldflags` 
NVLIBS = -lcublas

TARGET =	War

RankNetStockCV10: $(OBJS)
	${NVCC} ${NVCCFLAGS} -o RankNetStockCV10 test_stock.o DataSetMNIST.o DataSetLETOR.o DataSetStockFLRank.o DataSetStockFLRankCV.o ActivationFunctionTanh.o BackpropagationCU.o test_cuda.o test_RankNet.o test_LambdaRank.o test_BAM.o vivian.o gladys.o poi.o $(LIBS) ${NVLIBS}

$(TARGET): $(OBJS)
	$(NVCC) ${NVCCFLAGS} -o $(TARGET) $(OBJS) $(LIBS) ${NVLIBS}

%.o: %.cpp
	${CXX} ${CXXFLAGS} -c $< -o $@

%.o: %.cu
	${NVCC} ${NVCCFLAGS} -c $< -o $@

all: $(TARGET) $(TESTS)

test:
	mkdir alpha
	cd alpha
	mkdir beta

data: data-mnist data-letor

data-mnist:
	${MAKE} -C mnist

data-letor:
	${MAKE} -C letor

data-sparsenet:
	${MAKE} -C sparsenet

clean:
	rm -f $(OBJS) $(TARGET)

