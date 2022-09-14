CC = ${ONEAPI_ROOT}/compiler/2022.0.1/linux/bin/icpx
CCFLAGS = -std=c++20 -O3 -march=native -qopt-report=1 -DMKL_ILP64 -fp-model=fast -fno-alias
MKL_LIBS = -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

STEINROOT = ${PROJECTROOT}/stein
STEINCPP = $(STEINROOT)/code/c++/ilib
dscal : dscal.cpp
	$(CC) -c $(CCFLAGS) dscal.cpp -I${MKLROOT}/include -o dscal.o
	$(CC) dscal.o -L${MKLROOT}/lib/intel64 $(MKL_LIBS) -o dscal

matrix : test_matrix.cpp
	$(CC) -c $(CCFLAGS) test_matrix.cpp -I${MKLROOT}/include -I${STEINCPP}/include -o test_matrix.o
	$(CC) test_matrix.o -L${MKLROOT}/lib/intel64 $(MKL_LIBS) -o test_matrix

vsl : test_vsl.cpp
	$(CC) -c $(CCFLAGS) test_vsl.cpp -I${MKLROOT}/include -I${STEINCPP}/include -o test_vsl.o
	$(CC) test_vsl.o -L${MKLROOT}/lib/intel64 $(MKL_LIBS) -o test_vsl

mvn : test_mvn.cpp
	$(CC) -c $(CCFLAGS) test_mvn.cpp -I${MKLROOT}/include -I${STEINCPP}/include -I${ONEAPI_ROOT}/compiler/latest/linux/compiler/include -o test_mvn.o
	$(CC) test_mvn.o -L${MKLROOT}/lib/intel64 $(MKL_LIBS) -o test_mvn

opt : test_opt.cpp
	$(CC) -c $(CCFLAGS) test_opt.cpp -I${MKLROOT}/include -I${STEINCPP}/include -o test_opt.o
	$(CC) test_opt.o -L${MKLROOT}/lib/intel64 $(MKL_LIBS) -o test_opt

clean :
	rm -f dscal.o dscal test_matrix.o test_matrix mvn mvn.o

