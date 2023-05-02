#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <random>
#include <memory>
#include <values.h>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "bamdp_cvar_adv_decision_node.h"
#include "bamdp_cvar_chance_node.h"

/* Get the unperturbed successor probabilites for executing the state action
pair associated with this adversary decision node. As this node represents a
belief in a Bayes-Adaptive MDP, the transition probabilities depend on
the current belief state associated with this node.

Args:
    None

Returns:
    the unperturbed successor probabilities for executing this state action
    pair in the Belief MDP.
*/
std::unordered_map<State, float, StateHash> BamdpCvarAdvDecisionNode::getSuccessorProbs(){
    if(!successorProbsComputed){
        successorProbsComputed = true;
        successorProbs = b->getBeliefTransitionProbs(s, action);
    }
    return successorProbs;
}

std::shared_ptr<CvarChanceNode> BamdpCvarAdvDecisionNode::createChild(std::unordered_map<State, float, StateHash> perturbationMapping){
    if(!successorProbsComputed){
        successorProbsComputed = true;
        successorProbs = b->getBeliefTransitionProbs(s, action);
    }
    return std::make_shared<BamdpCvarChanceNode>(pMDP, b, s, alpha, action, perturbationMapping, successorProbs);
}

/* Expand heuristic perturbation according to a rollout policy */
int BamdpCvarAdvDecisionNode::expandHeuristicPerturbation(std::shared_ptr<BamdpRolloutPolicy> rolloutPolicy, int stepsRemaining){
    std::unordered_map<State, float, StateHash> perturbationMapping;
    int actionInd;

    // if there is little perturbation budget left or only one successor then
    // just return null perturbation
    if(doNotExpand()){
        actionInd = getNoPerturbationAction();

    // otherwise we can expand a new random action
    }else{
        perturbationMapping = rolloutPolicy->getRolloutPerturbation(
            s,
            action,
            alpha,
            stepsRemaining,
            b);

        actionInd = addChildChanceNode(perturbationMapping);
    }
    return actionInd;
}
