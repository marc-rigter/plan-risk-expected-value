#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "decision_node.h"
#include <random>

DecisionNode::DecisionNode(std::shared_ptr<MDP> pMDP_, const State state_):
pMDP(pMDP_), s(state_)
{
    possibleActions = pMDP->getEnabledActions(s);
}

State DecisionNode::getState(){
    return s;
}

DecisionNode::DecisionNode(){

}
