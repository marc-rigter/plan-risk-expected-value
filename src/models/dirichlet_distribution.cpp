#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <unordered_map>
#include <algorithm>
#include "state.h"
#include "belief.h"
#include "utils.h"
#include "dirichlet_distribution.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>

DirichletDistribution::DirichletDistribution(std::unordered_map<State, float, StateHash> priorStateCounts_)
: stateCounts(priorStateCounts_)
{

}

void DirichletDistribution::observe(State s){
    if(stateCounts.count(s) != 0){
        stateCounts[s] += 1.0;
    }else{
        std::cerr << "Error: State observed not in prior support of Dirichlet." << std::endl;
        std::exit(-1);
    }
}

std::unordered_map<State, float, StateHash>  DirichletDistribution::sampleDistribution() const{
    std::vector<double> counts;
    for(auto pair : stateCounts){
        counts.push_back(pair.second);
    }

    std::vector<double> sample;
    int n = counts.size();
    sample.reserve(n);

    // Allocate random number generator
    gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_ran_dirichlet(r, n, &counts[0], &sample[0]);

    std::unordered_map<State, float, StateHash> probs;
    int i = 0;
    for(auto pair : stateCounts){
        probs[pair.first] = sample[i];
    }
    return probs;
}

std::unordered_map<State, float, StateHash> DirichletDistribution::getExpectedDistribution() const{
    float totalCount = 0.0;
    std::unordered_map<State, float, StateHash> successorProbs;
    for(auto pair : stateCounts){
        totalCount += pair.second;
    }

    for(auto pair : stateCounts){
        successorProbs[pair.first] = pair.second / totalCount;
    }
    return successorProbs;
}
