CFLAGS = -c -Wall -DDEBUG -g  -O2 -std=gnu++14 -DBOOST_NO_AUTO_PTR
LDFLAGS =  -lm -lstdc++ -lppl -lgmp -lgmpxx -lshogun -lm -ldl -llpsolve55 -lopenblas -lgsl  -lgurobi_g++5.2 -lgurobi91

GUROBI_PATH = /usr/local/bin/gurobi910
SHOGUN_PATH = ../../miniconda3
EIGEN_PATH = /usr/include/eigen3
LP_SOLVE_PATH = /usr/lib/lp_solve

COMMON_SOURCES = $(filter-out src/main.cpp, $(wildcard src/*/*.cpp))
TARGET_SOURCES = src/main.cpp
TEST_SOURCES = $(wildcard tests/*.cpp) $(wildcard tests/*/*.cpp)
COMMON_OBJECTS = $(COMMON_SOURCES:.cpp=.o)
TARGET_OBJECTS = $(TARGET_SOURCES:.cpp=.o)
TEST_OBJECTS = $(TEST_SOURCES:.cpp=.o)
EXECUTABLE = run
TEST_EXECUTABLE = test

INC_DIRS=include/models include/algorithms include/utils external/eigen-3.3.8
INC_DIRS += $(GUROBI_PATH)/linux64/include
INC_DIRS += $(SHOGUN_PATH)/include
INC_DIRS += $(EIGEN_PATH)

LDFLAGS += -L$(GUROBI_PATH)/linux64/lib
LDFLAGS += -L$(SHOGUN_PATH)/lib
LDFLAGS += -L$(LP_SOLVE_PATH)

# include directories needed for volesti
INC_DIRS += external/volesti/include
INC_DIRS += external/volesti/external/LPsolve_src/run_headers
INC_DIRS += external/volesti/include/generators
INC_DIRS += external/volesti/external/minimum_ellipsoid
INC_DIRS += external/volesti/include/volume
INC_DIRS += external/volesti/include/convex_bodies
INC_DIRS += external/volesti/include/annealing
INC_DIRS += external/volesti/include/samplers
INC_DIRS += external/volesti/include/lp_oracles
INC_DIRS += external/volesti/include/misc

INC_PARAMS=$(foreach d, $(INC_DIRS), -I$d)

.PHONY: all target tests

all: target tests

target: $(EXECUTABLE)

tests: $(TEST_EXECUTABLE)

$(EXECUTABLE): $(COMMON_OBJECTS) $(TARGET_OBJECTS) 
	$(CC)  $(INC_PARAMS) $^ $(LDFLAGS) -o $@

$(TEST_EXECUTABLE): $(COMMON_OBJECTS) $(TEST_OBJECTS)
	$(CC)  $(INC_PARAMS) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(COMMON_OBJECTS) $(TEST_OBJECTS) $(TARGET_OBJECTS)

.cpp.o:
	$(CC) $(CFLAGS) $(INC_PARAMS) $< -o $@
