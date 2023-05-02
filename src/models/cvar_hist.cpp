#include <iostream>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include "state.h"
#include "cvar_decision_node.h"
#include "cvar_hist.h"
#include "mdp_examples.h"
#include "utils.h"

void CvarHist::addTransition(
        State s,
        std::string action,
        float alpha,
        std::unordered_map<State, float, StateHash> transitionProbs,
        std::unordered_map<State, float, StateHash> perturbation,
        float reward
){
    totalReturn += reward;
    stages.push_back(std::make_tuple(s, action, alpha, transitionProbs, perturbation, reward));
}

void CvarHist::printPath(bool verbose){
    std::cout << "############ Episode Path: ############" << std::endl;
    int step = 0;
    for(auto tuple : stages){
        std::cout << std::endl;
        std::cout << "## Step " << step << " ##" << std::endl;
        printTuple(tuple);

        step++;
        std::vector<std::string> sf = std::get<0>(tuple).getStateFactors();
    }

    std::cout << "Total Return: " << totalReturn << "\n" << std::endl;
}

void CvarHist::printPathToFile(std::string fileName){

    std::ofstream myfile;
    myfile.open(fileName, std::ios_base::app);

    myfile << "############ Episode Path: ############" << std::endl;
    int step = 0;
    for(auto tuple : stages){
        myfile << std::endl;
        myfile << "## Step " << step << " ##" << std::endl;
        myfile << "State: " << std::get<0>(tuple) << " Alpha: " << std::get<2>(tuple)  << " Agent Action: " << std::get<1>(tuple) << ", Reward: " << std::get<5>(tuple) << std::endl;
        step++;
        std::vector<std::string> sf = std::get<0>(tuple).getStateFactors();
    }

    myfile << "Total Return: " << totalReturn << "\n" << std::endl;
    myfile.close();
}

void CvarHist::printTuple(stage st){
    std::cout<< "State: " << std::get<0>(st) << " Alpha: " << std::get<2>(st)  << " Agent Action: " << std::get<1>(st) << ", Reward: " << std::get<5>(st) << std::endl;
    std::cout << "Transition probs: ";
    for(auto pair : std::get<3>(st)){
        std::cout  << pair.first << " prob: " << pair.second << "; ";
    }
    std::cout << std::endl;

    std::cout << "Adversary perturbation: ";
    for(auto pair : std::get<4>(st)){
        std::cout  << pair.first << " delta: " << pair.second << "; ";
    }
    std::cout << std::endl;
}
