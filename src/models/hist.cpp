#include <iostream>
#include <cmath>
#include <unordered_map>
#include "state.h"
#include "hist.h"

std::string getNewHistory(State prevHistoryState, std::string action, State nextState){
    std::string history = prevHistoryState.getValue("history");
    history.append("-").append(action).append("|").append(nextState.toString());
    return history;
}

Hist::Hist(){
    totalReturn = 0.0;
}

float Hist::getTotalReturn(){
    return totalReturn;
}

void Hist::addTransition(State s, std::string action, float reward){
    totalReturn += reward;
    path.push_back(std::make_tuple(s, action, reward));
}

void Hist::printPath(){
    std::cout << "Episode Path: " << std::endl;
    for(auto tup : path){
        std::cout<< "State: " << std::get<0>(tup) << " Action: " << std::get<1>(tup) << ", Reward: " << std::get<2>(tup) << std::endl;
    }
    std::cout << "Total Return: " << totalReturn << "\n" << std::endl;
}
