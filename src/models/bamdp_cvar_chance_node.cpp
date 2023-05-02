#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "bamdp_cvar_chance_node.h"
#include "bamdp_cvar_decision_node.h"


/* selects a random outcome of this chance node with the adversarial perturb
ation applied.*/
State BamdpCvarChanceNode::sampleSuccessor(){
    return b->samplePerturbedSuccessor(s, action, advPerturbation, nominalTransProbs);
}

/* return a child node corresponding to a sucessor state. */
std::shared_ptr<CvarDecisionNode> BamdpCvarChanceNode::createChild(State nextState){
    float perturbationApplied = advPerturbation[nextState];
    float newAlpha = alpha*perturbationApplied;
    if(newAlpha > 1.0){
        newAlpha = 1.0;
    }
    std::shared_ptr<Belief> newBelief = getNewBelief(s, action, nextState);
    return std::make_shared<BamdpCvarDecisionNode>(pMDP, newBelief, nextState, newAlpha);
}

/* Make a copy of the belief from this node and return the updated belief
according to the transition observed. Note that this does not modify
the belief at this node. */
std::shared_ptr<Belief> BamdpCvarChanceNode::getNewBelief(State s, std::string action, State nextState){

    // make a copy of the belief from this existing node
    std::shared_ptr<Belief> newBelief(b->clone());

    // update the belief according to the transition observed
    newBelief->updateBelief(s, action, nextState);
    return newBelief;
}

/* selects a random outcome of this chance node without the adversarial perturb
ation applied.*/
State BamdpCvarChanceNode::sampleUnperturbedSuccessor(){
    return b->sampleSuccessor(s, action, nominalTransProbs);
}
