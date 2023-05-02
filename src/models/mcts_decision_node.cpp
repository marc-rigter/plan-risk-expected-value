#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include <random>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "mcts_chance_node.h"
#include "mcts_decision_node.h"
#include "decision_node.h"


float MIN_BIAS = 10.0;

MCTSDecisionNode::MCTSDecisionNode(std::shared_ptr<MDP> pMDP_, const State state_):
DecisionNode(pMDP_, state_)
{
    observations = 0;
    totalReward = 0.0;
    notTakenActions = possibleActions;
}

int MCTSDecisionNode::getObservations(){
    return observations;
}

MCTSDecisionNode::MCTSDecisionNode(){

}

std::unordered_map<std::string, std::shared_ptr<MCTSChanceNode>> MCTSDecisionNode::getChildren(){
    return children;
}

std::vector<std::string> MCTSDecisionNode::getNotTakenActions(){
    return notTakenActions;
}

float MCTSDecisionNode::getCurrentValue(){
    if(observations == 0){
        return 0.0;
    }else{
        return totalReward/observations;
    }
}

/* Returns boolean defining whether this node is a leaf. The node is a leaf
if any of the actions have not been taken.
*/
bool MCTSDecisionNode::isFullyExpanded(){
    if(notTakenActions.size() == 0){
        return true;
    }else{
        return false;
    }
}

/* Returns whether or not an action has been expanded from this node */
bool MCTSDecisionNode::isExpanded(){
    if(notTakenActions.size() == possibleActions.size()){
        return false;
    }else{
        return true;
    }
}

/* Returns the action with the highest estimated Q value.

Returns:
    string: action with best Q value
*/
std::string MCTSDecisionNode::getBestAction(){
    std::string bestAction;
    int mostVisits = -1;
    int visits;

    for(auto pair : children){
        visits = pair.second->getObservations();
        if(visits > mostVisits){
            mostVisits = visits;
            bestAction = pair.first;
        }
    }
    return bestAction;
}

/* Expand decision node by choosing action which has not been chosen yet
and generating a new child node. This should only

Returns:
    action corresponding to the expanded child node.
*/
std::string MCTSDecisionNode::expand(){
    std::string action;

    // ucb is initialised by taking every action once. if any actions have not
    // been sampled choose one of them randomly.
    int numNotTaken = notTakenActions.size();
    int actionIndex;
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, numNotTaken-1);
    actionIndex = uni(rng);

    action = notTakenActions[actionIndex];
    notTakenActions.erase(notTakenActions.begin()+actionIndex);
    children[action] = std::make_shared<MCTSChanceNode>(pMDP, s, action);

    return action;
}

/* Pick an action using the UCB1 formula.

Returns:
    An action selected according to UCB.
*/
std::string MCTSDecisionNode::selectActionUCB(float biasFactor){

    // select the action which has the highest upper confidence bound.
    float bias = std::max(MIN_BIAS, std::fabs(getCurrentValue()*biasFactor));
    float maxUcb = -std::numeric_limits<float>::max();
    std::shared_ptr<MCTSChanceNode> pChildNode;
    std::string maxAction;
    std::string action;
    int childObs;
    float ucb;
    for(auto pair : children){
        action = pair.first;
        pChildNode = children[action];
        childObs = pChildNode->getObservations();

        ucb = bias * sqrt(log(observations)/childObs);
        ucb += pChildNode->getCurrentValue();

        if(ucb > maxUcb){
            maxUcb = ucb;
            maxAction = action;
        }
    }

    return maxAction;
}

/* Update the estimated value of this node and increment the visitation count*/
void MCTSDecisionNode::updateNode(float trialReturn){
    totalReward += trialReturn;
    visit();
}

/* Increment the number of times this node has been visited. */
void MCTSDecisionNode::visit(){
    observations++;
}
