CXX = icpc
CXXFLAGS = -O3 -openmp 
MICFLAGS  = -opt-prefetch-distance=8 -opt-streaming-stores always
HOSTFLAGS = -opt-prefetch-distance=8

OBJECTS = \
	Transpose-sp-CPU.o Transpose-dp-CPU.o\
	Main-sp-CPU.o Main-dp-CPU.o\
	Transpose-sp-MIC.o Transpose-dp-MIC.o\
	Main-sp-MIC.o Main-dp-MIC.o

TARGETS = runme-sp-CPU runme-dp-CPU runme-sp-MIC runme-dp-MIC

%-sp-CPU.o:
	$(CXX) $(CXXFLAGS) $(HOSTFLAGS) -DSINGLE -c -o $@ $*.cc

%-dp-CPU.o:
	$(CXX) $(CXXFLAGS) $(HOSTFLAGS) -DDOUBLE -c -o $@ $*.cc

%-sp-MIC.o:
	$(CXX) $(CXXFLAGS) $(MICFLAGS) -mmic -DSINGLE -c -o $@ $*.cc

%-dp-MIC.o:
	$(CXX) $(CXXFLAGS) $(MICFLAGS) -mmic -DDOUBLE -c -o $@ $*.cc

%-sp-CPU:
	$(CXX) $(CXXFLAGS) -o $@ *sp-CPU.o

%-dp-CPU:
	$(CXX) $(CXXFLAGS) -o $@ *dp-CPU.o

%-sp-MIC:
	$(CXX) $(CXXFLAGS) -mmic -o $@ *sp-MIC.o

%-dp-MIC:
	$(CXX) $(CXXFLAGS) -mmic -o $@ *dp-MIC.o

all: $(OBJECTS) $(TARGETS)

clean: 
	rm -f runme-* *.o
