#ifndef tied_dirichlet_distribution
#define tied_dirichlet_distribution
#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <unordered_map>
#include <algorithm>
#include "state.h"
#include "belief.h"
#include "utils.h"

class TiedDirichletDistribution
{
private:
    std::unordered_map<std::string, float> pseudoStateCounts;

public:
    TiedDirichletDistribution() {};
    TiedDirichletDistribution(std::unordered_map<std::string, float> priorPseudoStateCounts_);
    void observe(std::string pseudoState);
    std::unordered_map<std::string, float> getExpectedDistribution() const;
    std::unordered_map<std::string, float> sampleDistribution() const;

    TiedDirichletDistribution* clone () const        // Virtual constructor (copying)
    {
      return new TiedDirichletDistribution(*this);
    }

    TiedDirichletDistribution(const TiedDirichletDistribution &b2) {
        pseudoStateCounts = b2.pseudoStateCounts;
    }
};

#endif
