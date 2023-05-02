/* This file implements tests for the MDP class. */

#include <unordered_map>
#include <iostream>
#include "catch.h"
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "mdp_examples.h"
#include "finite_mdp_belief.h"

TEST_CASE("Finite MDP belief update"){
    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights;
    std::unordered_map<std::shared_ptr<MDP>, std::pair<int, int>> mdpGoals;
    for(int x = 0; x < 3; x++){
        for(int y = 0; y < 3; y++){
            std::shared_ptr<MDP> pMDP = makeSimpleMDP(x, y, 0.0);

            // don't allow the goal state to be placed at the initial state
            if(x == 0 && y == 0){
                mdpWeights[pMDP] = 0.0;
                continue;
            }
            mdpWeights[pMDP] = 0.125;
            mdpGoals[pMDP] = std::make_pair(x, y);
        }
    }
    FiniteMDPBelief b(mdpWeights);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = "0";
    stateMap["y"] = "1";
    State currentState(stateMap);

    stateMap["x"] = "1";
    stateMap["y"] = "1";
    State nextState(stateMap);

    b.updateBelief(currentState, "right", nextState);
    std::unordered_map<std::shared_ptr<MDP>, float> updatedWeights = b.getWeights();
    for(auto pair : mdpGoals){
        int xGoal = pair.second.first;
        int yGoal = pair.second.second;

        if(xGoal == 0 && yGoal == 1){
            REQUIRE(cmpf(updatedWeights[pair.first], 0.0));
        }else if(xGoal == 0 && yGoal == 0){
            REQUIRE(cmpf(updatedWeights[pair.first], 0.0));
        }else{
            REQUIRE(cmpf(updatedWeights[pair.first], 0.142857));
        }
    }

    FiniteMDPBelief b2(mdpWeights);
    b2.updateBelief(currentState, "right", currentState);
    updatedWeights = b2.getWeights();
    for(auto pair : mdpGoals){
        int xGoal = pair.second.first;
        int yGoal = pair.second.second;

        if(xGoal == 0 && yGoal == 1){
            REQUIRE(cmpf(updatedWeights[pair.first], 0.0));
        }else if(xGoal == 0 && yGoal == 0){
            REQUIRE(cmpf(updatedWeights[pair.first], 0.0));
        }else{
            REQUIRE(cmpf(updatedWeights[pair.first], 0.142857));
        }
    }

    FiniteMDPBelief b3(mdpWeights);
    stateMap["x"] = "-1";
    stateMap["y"] = "-1";
    State terminalState(stateMap);
    b3.updateBelief(currentState, "right", terminalState);
    updatedWeights = b3.getWeights();
    for(auto pair : mdpGoals){
        int xGoal = pair.second.first;
        int yGoal = pair.second.second;

        if(xGoal == 0 && yGoal == 1){
            REQUIRE(cmpf(updatedWeights[pair.first], 1.0));
        }else{
            REQUIRE(cmpf(updatedWeights[pair.first], 0.0));

        }
    }
}

TEST_CASE("Belief transition probs"){
    std::shared_ptr<MDP> testMDP1 = makeBettingMDP(0.5, 5);
    std::shared_ptr<MDP> testMDP2 = makeBettingMDP(0.8, 5);
    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights;
    mdpWeights[testMDP1] = 0.4;
    mdpWeights[testMDP2] = 0.6;
    FiniteMDPBelief b(mdpWeights);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State state(stateMap);

    stateMap["t"] = "1";
    stateMap["money"] = "13";
    State stateWin(stateMap);

    stateMap["t"] = "1";
    stateMap["money"] = "7";
    State stateLose(stateMap);

    std::unordered_map<State, float, StateHash> transitionProbs = b.getBeliefTransitionProbs(state, "3");

    REQUIRE(cmpf(transitionProbs[stateWin], 0.68));
    REQUIRE(cmpf(transitionProbs[stateLose], 0.32));
}
