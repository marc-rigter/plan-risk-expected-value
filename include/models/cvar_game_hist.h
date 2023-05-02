#ifndef cvar_game_history
#define cvar_game_history
#include <string>
#include <iostream>
#include "state.h"
#include "hist.h"
#include "cvar_decision_node.h"
#include "cvar_chance_node.h"

class CvarGameHist : public Hist
{
private:
    typedef std::tuple<State, std::string, float, std::unordered_map<State, float, StateHash>, std::unordered_map<State, float, StateHash>, float> stage;
    std::vector<stage> cvarGamePath;
    std::vector<std::vector<std::shared_ptr<CvarChanceNode>>> pathExpandedPerturbations;
    std::vector<std::shared_ptr<CvarDecisionNode>> agentNodesVisited;

public:
    CvarGameHist() : Hist() {};
    void addTransition(std::shared_ptr<CvarDecisionNode> pAgentNode, std::string action, int advActionInd, float reward);
    void printTuple(stage);
    void printExpandedPerturbationActions(std::vector<std::shared_ptr<CvarChanceNode>> expandedNodes);
    void printAgentActions(std::shared_ptr<CvarDecisionNode> agentNode);
    void printPath(bool verbose = false);
};

#endif
