#include <iostream>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include "state.h"
#include "cvar_decision_node.h"
#include "cvar_game_hist.h"
#include "cvar_adv_decision_node.h"
#include "cvar_chance_node.h"
#include "mdp_examples.h"
#include "utils.h"

void CvarGameHist::addTransition(std::shared_ptr<CvarDecisionNode> pAgentNode, std::string action, int advActionInd, float reward){

    // add standard information for path
    totalReturn += reward;
    State s = pAgentNode->getState();
    float alpha = pAgentNode->getAlpha();
    std::shared_ptr<CvarAdvDecisionNode> pCurrentAdvNode = pAgentNode->getAdvChildren()[action];
    std::unordered_map<State, float, StateHash> advPerturbation = pCurrentAdvNode->getPerturbationMapping(advActionInd);
    std::unordered_map<State, float, StateHash> transitionProbs;
    cvarGamePath.push_back(std::make_tuple (s, action, alpha, transitionProbs, advPerturbation, reward));

    // add all of the actions which have been expanded
    std::unordered_map<int, std::shared_ptr<CvarChanceNode>> chanceNodes = pCurrentAdvNode->getChanceChildren();
    std::vector<std::shared_ptr<CvarChanceNode>> expandedNodes;
    for(auto chanceNode : chanceNodes){
        expandedNodes.push_back(chanceNode.second);
    }
    pathExpandedPerturbations.push_back(expandedNodes);
    agentNodesVisited.push_back(pAgentNode);
}

void CvarGameHist::printPath(bool verbose){
    std::cout << "############ Episode Path: ############" << std::endl;
    int step = 0;
    for(auto tuple : cvarGamePath){
        std::cout << std::endl;
        std::cout << "## Step " << step << " ##" << std::endl;
        printTuple(tuple);
        if(verbose){
            std::cout << std::endl;
            printAgentActions(agentNodesVisited[step]);
            printExpandedPerturbationActions(pathExpandedPerturbations[step]);
            std::cout << std::endl;
        }

        std::vector<std::string> sf = std::get<0>(tuple).getStateFactors();
        step++;
    }

    std::cout << "Total Return: " << totalReturn << "\n" << std::endl;
}

void CvarGameHist::printTuple(stage st){
    std::cout<< "State: " << std::get<0>(st) << " Alpha: " << std::get<2>(st)  << " Agent Action: " << std::get<1>(st) << ", Reward: " << std::get<5>(st) << std::endl;
    std::cout << "Adversary perturbation: ";
    for(auto pair : std::get<4>(st)){
        std::cout  << pair.first << " delta: " << pair.second << "; ";
    }
    std::cout << std::endl;
}


void CvarGameHist::printAgentActions(std::shared_ptr<CvarDecisionNode> agentNode){
    for(auto pair : agentNode->getAdvChildren()){
        std::cout << "Action: " << pair.first << ", Value Estimate: " << pair.second->getCurrentValue() << ", Number of visits:" << pair.second->getObservations() << std::endl;
        std::cout << "Transition probabilities: ";
        for(auto kv : pair.second->getSuccessorProbs()){
            std::cout << kv.first << " prob: " << kv.second << "; ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void CvarGameHist::printExpandedPerturbationActions(std::vector<std::shared_ptr<CvarChanceNode>> expandedNodes){
    for(auto node : expandedNodes){
        std::cout << "Expanded perturbation: ";
        std::unordered_map<State, float, StateHash> pert = node->getPerturbation();
        for(auto pair : pert){
            std::cout << pair.first << " delta: " << pair.second << "; ";
        }
        std::cout << "Value estimate: " << node->getCurrentValue();
        std::cout << ", Number of visits: " << node->getObservations();
        std::cout << std::endl;
    }
}
