/* This file implements tests for the MDP class. */

#include <unordered_map>
#include <iostream>
#include "catch.h"
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "value_iteration.h"
#include "mdp_examples.h"


TEST_CASE("Enabled actions"){
    MDP testMDP = *makeSimpleMDP(2, 2, 2.0);
    std::vector<std::string> enabledActions;
    std::unordered_map<std::string, std::string> stateMap1;

    stateMap1["x"] = "1";
    stateMap1["y"] = "1";
    State state1(stateMap1);
    enabledActions = testMDP.getEnabledActions(state1);

    REQUIRE(enabledActions.size() == 4);
    REQUIRE(std::find(enabledActions.begin(), enabledActions.end(), "up") != enabledActions.end());
    REQUIRE(std::find(enabledActions.begin(), enabledActions.end(), "right") != enabledActions.end());
    REQUIRE(std::find(enabledActions.begin(), enabledActions.end(), "left") != enabledActions.end());
    REQUIRE(std::find(enabledActions.begin(), enabledActions.end(), "down") != enabledActions.end());

    stateMap1["x"] = "0";
    stateMap1["y"] = "0";
    State state2(stateMap1);
    enabledActions = testMDP.getEnabledActions(state2);
    REQUIRE(enabledActions.size() == 2);
}

TEST_CASE("Enumerate states"){
    MDP testMDP = *makeSimpleMDP(2, 2, 2.0);
    int numStates;

    numStates = testMDP.enumerateStates().size();
    REQUIRE(numStates == 10);
}

TEST_CASE("Get rewards"){
    MDP testMDP = *makeSimpleMDP(2, 2, 2.0);
    float reward;

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = "0";
    stateMap["y"] = "1";
    State state1(stateMap);

    reward = testMDP.getReward(state1, "up");
    REQUIRE(cmpf(reward, -1.0));

    stateMap["x"] = "-1";
    stateMap["y"] = "-1";
    State state2(stateMap);
    reward = testMDP.getReward(state2, "up");
    REQUIRE(cmpf(reward, 0.0));
}

TEST_CASE("Transition probs"){
    MDP testMDP = *makeSimpleMDP(2, 2, 2.0);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = "0";
    stateMap["y"] = "0";
    State state1(stateMap);

    stateMap["x"] = "0";
    stateMap["y"] = "1";
    State state2(stateMap);

    std::unordered_map<State, float, StateHash> probs;
    probs = testMDP.getTransitionProbs(state1, "up");

    REQUIRE(cmpf(probs[state1], 0.2));
    REQUIRE(cmpf(probs[state2], 0.8));
}
