#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include <random>
#include <algorithm>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "cvar_chance_node.h"
#include "cvar_decision_node.h"

float CvarChanceNode::getAlpha(){
    return alpha;
};

/* selects a random outcome of this chance node with the adversarial perturb
ation applied.*/
State CvarChanceNode::sampleSuccessor(){
    return pMDP->samplePerturbedSuccessor(s, action, advPerturbation);
}

/* selects a random outcome of this chance node with the adversarial perturb
ation applied and adds a child node if that successor has not been visited.*/
State CvarChanceNode::sampleSuccessorAddChild(){
    State nextState;
    nextState = sampleSuccessor();

    // if first time visiting this state create a node for the state.
    if(std::find(visited.begin(), visited.end(), nextState) == visited.end()){
        cvarChildren[nextState] = createChild(nextState);
        visited.push_back(nextState);
    }

    return nextState;
}

/* return a child node corresponding to a sucessor state. */
std::shared_ptr<CvarDecisionNode> CvarChanceNode::createChild(State nextState){
    float perturbationApplied = advPerturbation[nextState];
    float newAlpha = alpha*perturbationApplied;
    if(newAlpha > 1.0){
        newAlpha = 1.0;
    }
    return std::make_shared<CvarDecisionNode>(pMDP, nextState, newAlpha);
}

/* selects a random outcome of this chance node without the adversarial perturb
ation applied.*/
State CvarChanceNode::sampleUnperturbedSuccessor(){
    return pMDP->sampleSuccessor(s, action);
}

/* Add a child node at the successor state if it does not exist. */
void CvarChanceNode::addChild(State nextState){

    // if first time visiting this state create a node for the state.
    if(std::find(visited.begin(), visited.end(), nextState) == visited.end()){
        cvarChildren[nextState] = createChild(nextState);
        visited.push_back(nextState);
    }
}


std::unordered_map<State, std::shared_ptr<CvarDecisionNode>, StateHash> CvarChanceNode::getCvarChildren(){
    return cvarChildren;
}

std::unordered_map<State, float, StateHash> CvarChanceNode::getPerturbation(){
    return advPerturbation;
}
