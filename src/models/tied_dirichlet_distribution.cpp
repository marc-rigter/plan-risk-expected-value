#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <unordered_map>
#include <algorithm>
#include <sys/time.h>
#include "state.h"
#include "belief.h"
#include "utils.h"
#include "tied_dirichlet_distribution.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>


TiedDirichletDistribution::TiedDirichletDistribution(std::unordered_map<std::string, float> priorPseudoStateCounts_)
: pseudoStateCounts(priorPseudoStateCounts_)
{

}

void TiedDirichletDistribution::observe(std::string pseudoState){
    if(pseudoStateCounts.count(pseudoState) != 0){
        pseudoStateCounts[pseudoState] += 1.0;
    }else{
        std::cout << "Pseudo state referenced: " << pseudoState << std::endl;
        std::cerr << "Error: Pseudo state observed not in prior support of Dirichlet." << std::endl;
        std::exit(-1);
    }
}

std::unordered_map<std::string, float>  TiedDirichletDistribution::sampleDistribution() const{
    std::vector<double> counts;

    for(auto pair : pseudoStateCounts){
        counts.push_back(pair.second);
    }

    std::vector<double> sample;
    int n = counts.size();
    sample.reserve(n);

    // Allocate random number generator
    gsl_rng_env_setup();
    gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
    struct timeval tv;
    gettimeofday(&tv,0);
    unsigned long mySeed = tv.tv_sec + tv.tv_usec;
    gsl_rng_set(r, mySeed);
    gsl_ran_dirichlet(r, n, &counts[0], &sample[0]);

    std::unordered_map<std::string, float> probs;
    int i = 0;
    for(auto pair : pseudoStateCounts){
        probs[pair.first] = sample[i];
        i++;
    }
    return probs;
}

std::unordered_map<std::string, float> TiedDirichletDistribution::getExpectedDistribution() const{
    std::unordered_map<std::string, float> successorProbs;
    float totalCount = 0.0;
    for(auto pair : pseudoStateCounts){
        totalCount += pair.second;
    }

    for(auto pair : pseudoStateCounts){
        successorProbs[pair.first] = pair.second / totalCount;
    }
    return successorProbs;
}
