/* This file implements MCTS search using the UCB selection criterion */

#include <iostream>
#include <cmath>
#include <unordered_map>
#include "mdp.h"
#include "multimodel_mdp.h"
#include "state.h"
#include "hist.h"
#include "bamcp_maxprob_solver.h"

void BAMCPMaxProbSolver::updateNodes(
        std::vector<MCTSDecisionNode*>&  decisionNodesToBackup,
        std::vector<MCTSChanceNode*>& chanceNodesToBackup,
        std::vector<float>& rewards)
{
    MCTSDecisionNode* pCurrentDecisionNode;
    MCTSChanceNode* pCurrentChanceNode;

    // get the appropriate return which corresponds to 1 for meeting threshold
    // and 0 for failing the threshold.
    float backupVal = getReturn(rewards);

    // backup values through the tree
    while(chanceNodesToBackup.size() != 0){

        // move backwards through list to backup
        pCurrentChanceNode = chanceNodesToBackup.back();
        pCurrentDecisionNode = decisionNodesToBackup.back();

        // update values along the path
        pCurrentChanceNode->updateNode(backupVal);
        pCurrentDecisionNode->updateNode(backupVal);

        // remove values from vectors to backup
        chanceNodesToBackup.pop_back();
        decisionNodesToBackup.pop_back();
    }
}

/* Return the total value corresponding to the vector of rewards collected.
*/
float BAMCPMaxProbSolver::getReturn(
    std::vector<float>& rewards
){
    float totalReturn = 0.0;
    for(auto rew : rewards){
        totalReturn += rew;
    }

    float backupVal;
    if(totalReturn + rewardThisEpisode >= minReward){
        backupVal = 1.0;
    }else{
        backupVal = 0.0;
    }
    return backupVal;
}
