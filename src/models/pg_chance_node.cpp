#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "pg_chance_node.h"
#include "pg_decision_node.h"

PGChanceNode::PGChanceNode(std::shared_ptr<MDP> pMDP_, const State state_, const std::string action_):
ChanceNode(pMDP_, state_, action_)
{
}


State PGChanceNode::sampleSuccessor(std::shared_ptr<MDP> pSampledMDP){

    State nextState;
    nextState = pSampledMDP->sampleSuccessor(s, action);

    // if first time visiting this state create a node for the state.
    if(std::find(visited.begin(), visited.end(), nextState) == visited.end()){
        children[nextState] = std::make_shared<PGDecisionNode>(pMDP, nextState);
        visited.push_back(nextState);
    }

    return nextState;
}

std::unordered_map<State, std::shared_ptr<PGDecisionNode>, StateHash> PGChanceNode::getChildren(){
    return children;
}
