/* This file implements MCTS search using the UCB selection criterion */

#include <iostream>
#include <cmath>
#include <unordered_map>
#include "mdp.h"
#include "multimodel_mdp.h"
#include "state.h"
#include "mcts_decision_node.h"
#include "hist.h"
#include "bamcp_threshold_solver.h"


void BAMCPThresholdSolver::updateNodes(
        std::vector<MCTSDecisionNode*>&  decisionNodesToBackup,
        std::vector<MCTSChanceNode*>& chanceNodesToBackup,
        std::vector<float>& rewards)
{
    MCTSDecisionNode* pCurrentDecisionNode;
    MCTSChanceNode* pCurrentChanceNode;

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
float BAMCPThresholdSolver::getReturn(
    std::vector<float>& rewards
){
    float totalReturn = 0.0;
    for(auto rew : rewards){
        totalReturn += rew;
    }

    // compute the gap between the total reward for the episode (reward incurred
    // so far to reach this state plus return thereafter) and the threshold
    float diff;
    diff = (totalReturn + rewardThisEpisode) - minReward;

    // if we exceed the threshold then we just set the return to 0. otherwise
    // we incurr a (negative) reward which is the deficit to the cost threshold.
    float backupVal;
    if(diff >= 0.0){
        backupVal = 0.0;
    }else{
        backupVal = diff;
    }
    return backupVal;
}
