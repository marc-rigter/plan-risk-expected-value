#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "chance_node.h"



ChanceNode::ChanceNode(std::shared_ptr<MDP> pMDP_, const State state_, const std::string action_):
pMDP(pMDP_), s(state_), action(action_)
{
    localReward = pMDP->getReward(s, action);
}

std::string ChanceNode::getAction(){
    return action;
}

State ChanceNode::getState(){
    return s;
}

float ChanceNode::getLocalReward(){
    return localReward;
}
