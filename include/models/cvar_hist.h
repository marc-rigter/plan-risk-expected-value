#ifndef cvar_hist
#define cvar_hist
#include <string>
#include <iostream>
#include "state.h"
#include "hist.h"

class CvarHist : public Hist
{
private:
    typedef std::tuple<State, std::string, float, std::unordered_map<State, float, StateHash>, std::unordered_map<State, float, StateHash>, float> stage;
    std::vector<stage> stages;

public:
    CvarHist() : Hist() {};
    void addTransition(
            State s,
            std::string action,
            float alpha,
            std::unordered_map<State, float, StateHash> transitionProbs,
            std::unordered_map<State, float, StateHash> perturbation,
            float reward);
    void printPath(bool verbose = true);
    void printPathToFile(std::string fileName);
    void printTuple(stage st);
};

#endif
