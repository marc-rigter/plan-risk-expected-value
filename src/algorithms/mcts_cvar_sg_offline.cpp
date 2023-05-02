/* This file implements MCTS search using the UCB selection criterion */

#include <iostream>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include "mdp.h"
#include "state.h"
#include "mcts_cvar_sg.h"
#include "mcts_cvar_sg_offline.h"
#include "cvar_chance_node.h"
#include "cvar_adv_decision_node.h"
#include "bamdp_cvar_decision_node.h"
#include "bamdp_cvar_adv_decision_node.h"
#include "bamdp_rollout_policy.h"
#include "utils.h"

CvarMCTSOffline::CvarMCTSOffline(
    std::shared_ptr<MDP> pMDP_,
    std::shared_ptr<CvarDecisionNode> rootNode_,
    int horizon_,
    std::string optim_,
    float biasMultiplier_,
    float wideningParam_,
    std::string wideningStrategy_,
    std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy_,
    std::string rootAction_
) :
CvarMCTS(biasMultiplier_, wideningParam_, wideningStrategy_),
pTemplateMDP(pMDP_),
rootNode(rootNode_),
horizon(horizon_),
optim(optim_),
rolloutPolicy(rolloutPolicy_),
rootAction(rootAction_)
{
    alpha = rootNode->getAlpha();
    initState = rootNode->getState();
}


void CvarMCTSOffline::runTrials(
        int batchSize,
        int iterations)
{

    int it = 0;
    while(it < iterations){
        int trial = 0;
        std::vector<float> returns;
        while(trial < batchSize){
            float ret = this->trial(
                        pTemplateMDP,
                        rootNode,
                        horizon,
                        rolloutPolicy,
                        rootAction
                    );
            returns.push_back(ret);
            trial++;
        }

        // compute the empirical estimate of the VaR
        std::vector<float> returnsSorted = returns;
        std::sort(returnsSorted.begin(), returnsSorted.end());
        int varIndex = (int)(std::round(returns.size() * alpha)) - 1;
        float cvarSum = 0.0;
        for(int i = 0; i <= varIndex; i++){
            cvarSum += returnsSorted[i];
        }
        std::cout << "Iteration: " << it << ", Batch empirical cvar: " << cvarSum/(varIndex + 1) << std::endl;
        it++;
    }

}

CvarGameHist CvarMCTSOffline::executeEpisode(std::shared_ptr<MDP> pTrueMDP){
    CvarGameHist history = executeEpisode(
        pTrueMDP,
        rootNode,
        horizon,
        0,
        0,
        0,
        optim,
        rolloutPolicy
    );

    return history;
}
