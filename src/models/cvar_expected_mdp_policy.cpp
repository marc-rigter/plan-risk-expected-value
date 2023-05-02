#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <unordered_map>
#include <algorithm>
#include "state.h"
#include "belief.h"
#include "utils.h"
#include "mdp.h"
#include "cvar_expected_mdp_policy.h"
#include "value_iteration.h"

CvarExpectedMDPPolicy::CvarExpectedMDPPolicy(std::shared_ptr<Belief> b) {
    std::cout << "Computing expected MDP cvar rollout policy..." << std::endl;
    env.start();
    env.set(GRB_IntParam_OutputFlag, 0);

    // update expected MDP
    expectedMDP = b->getExpectedMDP();

    int numInterpPts = 20;
    pSolver = std::make_shared<CvarValueIteration>(numInterpPts);
    pSolver->valueIteration(*expectedMDP);
    std::cout << "Done." << std::endl;

    cvarPolicy = pSolver->getCvarPolicy();
    cvarPerturbationPolicy = pSolver->getCvarPerturbationPolicy();
}

void CvarExpectedMDPPolicy::updateRolloutPolicy(std::shared_ptr<Belief> b){
    return;
}


std::string CvarExpectedMDPPolicy::getRolloutAction(State s, float alpha, int horizon, std::shared_ptr<Belief> pCurrentBelief){
    std::string action;


    // hack as at final state the cvar policy is undefined
    std::map<std::string, std::string> pol = cvarPolicy[s];
    if(pol.size() == 0){
        action = expectedMDP->getEnabledActions(s)[0];
        return action;

    }else{
        std::string closestAlpha;
        float maxDiff = 1.0;
        for(auto kv : pol){
            float diff = std::stof(kv.first) - alpha;
            if(std::fabs(diff) < maxDiff){
                maxDiff = std::fabs(diff);
                closestAlpha = kv.first;
            }
            if(diff > 0.0){
                break;
            }
        }

        action = cvarPolicy[s][closestAlpha];
        return action;
    }
}

std::unordered_map<State, float, StateHash> CvarExpectedMDPPolicy::getRolloutPerturbation(State s, std::string action, float alpha, int horizon, std::shared_ptr<Belief> pCurrentBelief){
    std::unordered_map<State, float, StateHash> expectedMDPperturbation;
    std::tie(action, expectedMDPperturbation) = pSolver->getOptimalAction(
                                            expectedMDP,
                                            s,
                                            alpha,
                                            env
                                    );

    std::unordered_map<State, float, StateHash> probs = expectedMDP->getTransitionProbs(s, action);
    std::unordered_map<State, float, StateHash> perturbedProbs;
    for(auto kv : expectedMDPperturbation){
        perturbedProbs[kv.first] = probs[kv.first] * expectedMDPperturbation[kv.first];
    }

    std::unordered_map<State, float, StateHash> beliefProbs = pCurrentBelief->getBeliefTransitionProbs(s, action);
    std::unordered_map<State, float, StateHash> beliefMDPPerturbation;
    for(auto kv : perturbedProbs){
        beliefMDPPerturbation[kv.first] = kv.second/beliefProbs[kv.first];
    }

    return beliefMDPPerturbation;
}
