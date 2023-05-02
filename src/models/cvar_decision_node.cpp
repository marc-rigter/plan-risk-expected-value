#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "mcts_chance_node.h"
#include "cvar_decision_node.h"
#include "cvar_adv_decision_node.h"
#include <random>

float MIN_BIAS_AGENT = 5.0;

/* Expand an action for the agent in the Cvar SG. The only difference compared
to expansion in standard MCTS is that the child is an adversary decision node.
TODO: fix code reuse with standard MCTS */
std::string CvarDecisionNode::expand(){
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
    children[action] = createChild(action);
    return action;
}

/* return a child node corresponding to a sucessor state. */
std::shared_ptr<CvarAdvDecisionNode> CvarDecisionNode::createChild(std::string action){
    return std::make_shared<CvarAdvDecisionNode>(pMDP, s, alpha, action);
}

void CvarDecisionNode::setAlpha(float alpha_){
    alpha = alpha_;
}

std::unordered_map<std::string, std::shared_ptr<CvarAdvDecisionNode>>  CvarDecisionNode::getAdvChildren(){
    return children;
}

/* Returns the action with the highest estimated Q value.

Returns:
    string: action with best Q value
*/
std::string CvarDecisionNode::getBestAction(){
    std::string bestAction;
    float highestValue = -std::numeric_limits<float>::max();
    float value;

    for(auto pair : children){
        value = pair.second->getCurrentValue();
        std::cout << "action: " << pair.first << "value est. " << pair.second->getCurrentValue() << std::endl;
        if(value > highestValue){
            highestValue = value;
            bestAction = pair.first;
        }
    }
    std::cout << std::endl;
    return bestAction;
}


/* print one random possible path from this root node according to
the best action choices in the tree. */
void CvarDecisionNode::printPossiblePath(){
    State nextState;
    State currentState = s;
    std::string agentAction = getBestAction();
    std::shared_ptr<CvarAdvDecisionNode> advNode = children[agentAction];

    // loop until we reach a node without expanded actions
    while(agentAction != ""){

        // print current state and best agent action
        std::cout << "state: " << currentState << " best action: " << agentAction << std::endl;

        // print best adversary perturbation
        int bestAdvActionInd = advNode->getBestAdvAction();
        std::unordered_map<State, float, StateHash> pert = advNode->getPerturbationMapping(bestAdvActionInd);
        std::cout << "Adversary perturbation: ";
        for(auto pair : pert){
            std::cout << "state: " << pair.first << " delta: " << pair.second << "; ";
        }
        std::cout << std::endl;

        // get corresponding chance node
        std::shared_ptr<CvarChanceNode> pCurrentChanceNode;
        pCurrentChanceNode = advNode->getChanceChildren()[bestAdvActionInd];

        // sampe random next state
        nextState = pCurrentChanceNode->sampleSuccessorAddChild();
        std::cout << "Random next state: " << nextState << std::endl << std::endl;

        // update current agent decision node
        std::shared_ptr<CvarDecisionNode> pCurrentAgentNode;
        pCurrentAgentNode = pCurrentChanceNode->getCvarChildren()[nextState];
        currentState = nextState;

        // best best agent action and next adversary node.
        agentAction = pCurrentAgentNode->getBestAction();
        advNode = pCurrentAgentNode->getAdvChildren()[agentAction];
    }
}

/* Pick an action using the UCB1 formula.

Returns:
    An action selected according to UCB.
*/
std::string CvarDecisionNode::selectActionUCB(float biasFactor){

    // select the action which has the highest upper confidence bound.
    float bias = std::max(MIN_BIAS_AGENT, std::fabs(getCurrentValue()*biasFactor));
    float maxUcb = -std::numeric_limits<float>::max();
    std::shared_ptr<CvarAdvDecisionNode> pChildNode;
    std::string maxAction;
    std::string action;
    int childObs;
    float ucb;
    for(auto pair : children){

        action = pair.first;
        pChildNode = pair.second;
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
