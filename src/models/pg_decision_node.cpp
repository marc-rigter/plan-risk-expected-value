#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include <random>
#include <algorithm>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "pg_decision_node.h"


PGDecisionNode::PGDecisionNode(std::shared_ptr<MDP> pMDP_, const State state_):
DecisionNode(pMDP_, state_)
{
    for(auto act : pMDP_->getEnabledActions(state_)){
        actionWeights[act] = 1.0;
    }
}

std::unordered_map<std::string, std::shared_ptr<PGChanceNode>> PGDecisionNode::getChildren(){
    return children;
}

std::shared_ptr<PGChanceNode> PGDecisionNode::createChild(std::string action){
    return std::make_shared<PGChanceNode>(pMDP, s, action);
}

std::unordered_map<std::string, float> PGDecisionNode::getActionWeights(){
    return actionWeights;
}

void PGDecisionNode::setActionWeight(std::string action, float value){
    actionWeights[action] = value;
}

std::string PGDecisionNode::selectActionSoftmax(){
    float weightSum = 0.0;
    std::vector<float> weights;
    for(auto pair : actionWeights){
        weightSum += std::exp(pair.second);
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<std::string> actions;
    std::vector<float> probs;
    actions.reserve(actionWeights.size());
    probs.reserve(actionWeights.size());

    for(auto kv : actionWeights) {
        actions.push_back(kv.first);
        probs.push_back(std::exp(kv.second)/weightSum);
    }

    // sample a state index according to perturbed distribution
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int ind = dist(gen);
    std::string action = actions.at(ind);

    // if the action hasn't been taken create a child node
    if(std::find(actionsTaken.begin(), actionsTaken.end(), action) == actionsTaken.end()){
        children[action] = createChild(action);
        actionsTaken.push_back(action);
    }

    return action;
}
