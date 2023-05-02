/* This file implements tests for the MCTS implementation. */

#include <iostream>
#include <string>
#include "catch.h"
#include "mdp_examples.h"
#include "cvar_decision_node.h"
#include "mcts_cvar_sg.h"
#include "utils.h"
#include "cvar_game_hist.h"
#include "cvar_adv_decision_node.h"



TEST_CASE("Cvar Decision node observations"){
    std::shared_ptr<MDP> pTestMDP = makeSimpleMDP(2, 2, 2.0);
    State myState = pTestMDP->enumerateStates()[0];
    CvarDecisionNode rootNode(pTestMDP, myState, 0.5);
    std::string actionChosen;

    REQUIRE(rootNode.getState() == myState);
    REQUIRE(rootNode.getObservations() == 0);

    rootNode.updateNode(5.0);
    REQUIRE(rootNode.getObservations() == 1);

    rootNode.updateNode(7.0);
    REQUIRE(cmpf(rootNode.getCurrentValue(), 6.0));
}

TEST_CASE("Cvar Decision node children"){
    std::shared_ptr<MDP> pTestMDP = makeSimpleMDP(2, 2, 2.0);
    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = "1";
    stateMap["y"] = "1";
    State myState(stateMap);
    CvarDecisionNode rootNode(pTestMDP, myState, 0.5);
    std::string actionChosen;

    actionChosen = rootNode.expand();
    actionChosen = rootNode.expand();
    actionChosen = rootNode.expand();
    actionChosen = rootNode.expand();
    std::unordered_map<std::string, std::shared_ptr<CvarAdvDecisionNode>> rootChildren = rootNode.getAdvChildren();

    REQUIRE(rootChildren["up"]->getAction() == "up");
    REQUIRE(rootChildren["up"]->getState() == myState);
    REQUIRE(rootChildren["right"]->getState() == myState);
}

TEST_CASE("Check random actions for adversary"){
    std::shared_ptr<MDP> pTestMDP = makeSimpleMDP(2, 2, 2.0);
    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = "1";
    stateMap["y"] = "1";
    State myState(stateMap);
    CvarDecisionNode rootNode(pTestMDP, myState, 0.5);
    std::string actionChosen;

    actionChosen = rootNode.expand();
    actionChosen = rootNode.expand();
    actionChosen = rootNode.expand();
    actionChosen = rootNode.expand();

    std::unordered_map<std::string, std::shared_ptr<CvarAdvDecisionNode>> rootChildren = rootNode.getAdvChildren();
    std::shared_ptr<CvarAdvDecisionNode> advNode = rootChildren["left"];

    std::unordered_map<State, float, StateHash> probs = pTestMDP->getTransitionProbs(myState, "left");
    advNode->expandRandomPerturbation();
    advNode->expandRandomPerturbation();
    advNode->expandRandomPerturbation();
    advNode->expandRandomPerturbation();

    for(int i = 0; i < 4; i++){
        std::unordered_map<State, float, StateHash> advPerturbation;
        advPerturbation = advNode->getPerturbationMapping(i);

        float sum = 0.0;
        for(auto pair : advPerturbation){
            sum += pair.second * probs[pair.first];
        }
        REQUIRE(cmpf(sum, 1.0));
    }
}
