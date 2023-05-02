#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include <algorithm>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "mcts_decision_node.h"
#include <random>


MCTSChanceNode::MCTSChanceNode(std::shared_ptr<MDP> pMDP_, const State state_, const std::string action_):
ChanceNode(pMDP_, state_, action_)
{
    observations = 0;
    totalReward = 0.0;
}

float MCTSChanceNode::getCurrentValue(){
    if(observations == 0){
        return 0.0;
    }else{
        return totalReward/observations;
    }
}

int MCTSChanceNode::getObservations(){
    return observations;
}

std::unordered_map<State, std::shared_ptr<MCTSDecisionNode>, StateHash> MCTSChanceNode::getChildren(){
    return children;
}

/* selects a random successor state according to the transition probabilities
associated with the state action pair.

Args:
    pSampledMDP (optional): optional argument of shared pointer to an MDP.
        If this argument is specified, the MDP being pointed to is used for
        the transition function to sample the successor. Otherwise the MDP
        originally associated with the node is used.
*/
State MCTSChanceNode::selectRandomOutcome(std::shared_ptr<MDP> pSampledMDP){

    State nextState;
    if(pSampledMDP == nullptr){
        nextState = pMDP->sampleSuccessor(s, action);
    }else{
        nextState = pSampledMDP->sampleSuccessor(s, action);
    }

    // if first time visiting this state create a node for the state.
    if(std::find(visited.begin(), visited.end(), nextState) == visited.end()){
        children[nextState] = std::make_shared<MCTSDecisionNode>(pMDP, nextState);
        visited.push_back(nextState);
    }

    return nextState;
}

/* Update the estimate of the current value at this node and increment the
visitation count. */
void MCTSChanceNode::updateNode(float trialReturn){
    totalReward += trialReturn;
    visit();
}

/* Updates the counts of times this node has been visited.*/
void MCTSChanceNode::visit(){
    observations++;
}
