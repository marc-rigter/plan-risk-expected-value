/* This file implements tests for the MCTS implementation. */

#include <iostream>
#include <string>
#include "catch.h"
#include "mdp_examples.h"
#include "mcts_chance_node.h"
#include "mcts_decision_node.h"
#include "mcts.h"
#include "utils.h"

TEST_CASE("Decision node observations"){
    std::shared_ptr<MDP> pTestMDP = makeSimpleMDP(2, 2, 2.0);
    State myState = pTestMDP->enumerateStates()[0];
    MCTSDecisionNode rootNode(pTestMDP, myState);
    std::string actionChosen;

    REQUIRE(rootNode.getState() == myState);
    REQUIRE(rootNode.getObservations() == 0);

    rootNode.updateNode(5.0);
    REQUIRE(rootNode.getObservations() == 1);
}

TEST_CASE("Decision node children"){
    std::shared_ptr<MDP> pTestMDP = makeSimpleMDP(2, 2, 2.0);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = "1";
    stateMap["y"] = "1";
    State myState(stateMap);
    MCTSDecisionNode rootNode(pTestMDP, myState);
    std::string actionChosen;

    actionChosen = rootNode.expand();
    actionChosen = rootNode.expand();
    actionChosen = rootNode.expand();
    actionChosen = rootNode.expand();
    std::unordered_map<std::string, std::shared_ptr<MCTSChanceNode>> rootChildren = rootNode.getChildren();

    REQUIRE(rootChildren["up"]->getAction() == "up");
    REQUIRE(rootChildren["up"]->getState() == myState);
    REQUIRE(rootChildren["right"]->getState() == myState);
}

TEST_CASE("Fully expanded decision node"){
    std::shared_ptr<MDP> pTestMDP = makeSimpleMDP(2, 2, 2.0);
    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = "1";
    stateMap["y"] = "1";
    State myState(stateMap);
    MCTSDecisionNode rootNode(pTestMDP, myState);
    std::string actionChosen;

    REQUIRE(!rootNode.isFullyExpanded());
    actionChosen = rootNode.expand();
    REQUIRE(!rootNode.isFullyExpanded());
    actionChosen = rootNode.expand();
    actionChosen = rootNode.expand();
    actionChosen = rootNode.expand();
    REQUIRE(rootNode.isFullyExpanded());
}

TEST_CASE("Chance node attributes"){
    std::shared_ptr<MDP> pTestMDP = makeSimpleMDP(2, 2, 2.0);
    State myState = pTestMDP->enumerateStates()[0];
    MCTSChanceNode cNode(pTestMDP, myState, "up");

    REQUIRE(cNode.getAction() == "up");
    REQUIRE(cNode.getState() == myState);
}

TEST_CASE("Chance node sampling"){
    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = "0";
    stateMap["y"] = "0";
    State state1(stateMap);
    stateMap["x"] = "0";
    stateMap["y"] = "1";
    State state2(stateMap);

    std::shared_ptr<MDP> pTestMDP = makeSimpleMDP(2, 2, 2.0);
    MCTSChanceNode cNode(pTestMDP, state1, "up");

    State successor = cNode.selectRandomOutcome();
    REQUIRE(((successor == state1) || (successor == state2)));

    std::unordered_map<State, std::shared_ptr<MCTSDecisionNode>, StateHash> children = cNode.getChildren();
    REQUIRE(children[successor]->getState() == successor);
}

TEST_CASE("MCTS"){
    std::shared_ptr<MDP> pTestMDP = makeSimpleMDP(2, 2, 2.0);
    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = "2";
    stateMap["y"] = "2";
    State state1(stateMap);
    MCTS m(2.0);
    std::shared_ptr<MCTSDecisionNode> root = m.runMCTS(pTestMDP, state1, 10, 10);
    REQUIRE(cmpf(root->getCurrentValue(), -1.0));

    stateMap["x"] = "2";
    stateMap["y"] = "1";
    State state2(stateMap);
    std::shared_ptr<MCTSDecisionNode> root2 = m.runMCTS(pTestMDP, state2, 10, 100);
    REQUIRE(root2->getBestAction() == "up");

    stateMap["x"] = "1";
    stateMap["y"] = "1";
    State state3(stateMap);
    std::shared_ptr<MCTSDecisionNode> root3 = m.runMCTS(pTestMDP, state3, 10, 3000);
    REQUIRE(root3->getBestAction() == "right");
}
