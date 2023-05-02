#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "mcts_chance_node.h"
#include "bamdp_cvar_decision_node.h"
#include "bamdp_cvar_adv_decision_node.h"
#include <random>



/* return a child node corresponding to a sucessor state. */
std::shared_ptr<CvarAdvDecisionNode> BamdpCvarDecisionNode::createChild(std::string action){
    return std::make_shared<BamdpCvarAdvDecisionNode>(pMDP, b, s, alpha, action);
}

std::shared_ptr<Belief> BamdpCvarDecisionNode::getBelief(){
    return b;
}
