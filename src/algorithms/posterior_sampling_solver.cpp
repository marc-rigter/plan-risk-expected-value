/* This file implements MCTS search using the UCB selection criterion */

#include <iostream>
#include <unordered_map>
#include "mdp.h"
#include "multimodel_mdp.h"
#include "state.h"
#include "value_iteration.h"
#include "posterior_sampling_solver.h"


PosteriorSamplingSolver::PosteriorSamplingSolver(int resampleInterval_):
resampleInterval(resampleInterval_)
{
    step = resampleInterval;
}

/* Get the next action choice by using the optimal policy for a posterior MDP
sample. A new MDP is sampled from the posterior at the interval given by
the resampleInterval attribute. */
std::string PosteriorSamplingSolver::getNextAction(MultiModelMDP& mmdp, State currentState, int horizon){

    // if we have reached the resampling interval sample a new model and
    // compute a new corresponding optimal policy
    if(step == resampleInterval){
        std::shared_ptr<MDP> pMDP = mmdp.sampleModel();
        std::unordered_map<State, float, StateHash> value;
        VI vi;
        std::tie(value, currentPolicy) = vi.valueIteration(*pMDP, true);
        step = 0;
    }
    step++;

    return currentPolicy[currentState];
}
