#ifndef dirichlet_distribution
#define dirichlet_distribution
#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <unordered_map>
#include <algorithm>
#include "state.h"
#include "belief.h"
#include "utils.h"

class DirichletDistribution
{
private:
    std::unordered_map<State, float, StateHash> stateCounts;

public:
    DirichletDistribution() {};
    DirichletDistribution(std::unordered_map<State, float, StateHash> priorStateCounts_);
    void observe(State state);
    std::unordered_map<State, float, StateHash> getExpectedDistribution() const;
    std::unordered_map<State, float, StateHash> sampleDistribution() const;

    
};

#endif
