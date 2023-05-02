/* This file implements tests for solving BAMDPs with MCTS */

#include <iostream>
#include <string>
#include "mdp.h"
#include "catch.h"
#include "multimodel_mdp.h"
#include "mdp_examples.h"
#include "mcts_decision_node.h"
#include "hist.h"
#include "utils.h"
#include "posterior_sampling_solver.h"
#include "bamcp_solver.h"


TEST_CASE("Belief update"){
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
    MultiModelMDP mmdp(mdpWeights);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = "0";
    stateMap["y"] = "1";
    State currentState(stateMap);

    stateMap["x"] = "1";
    stateMap["y"] = "1";
    State nextState(stateMap);

    mmdp.updateBelief(currentState, "right", nextState);
    std::unordered_map<std::shared_ptr<MDP>, float> updatedWeights = mmdp.getWeights();
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

    MultiModelMDP mmdp2(mdpWeights);
    mmdp2.updateBelief(currentState, "right", currentState);
    updatedWeights = mmdp2.getWeights();
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

    MultiModelMDP mmdp3(mdpWeights);
    stateMap["x"] = "-1";
    stateMap["y"] = "-1";
    State terminalState(stateMap);
    mmdp3.updateBelief(currentState, "right", terminalState);
    updatedWeights = mmdp3.getWeights();
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

TEST_CASE("Bayes Adaptive MCTS"){
    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights;
    std::unordered_map<std::shared_ptr<MDP>, std::pair<int, int>> mdpGoals;
    std::shared_ptr<MDP> pTrueModel;
    int y = 2;
    for(int x = 0; x < 3; x++){
        std::shared_ptr<MDP> pMDP = makeSimpleMDP(x, y, 0.0);

        // don't allow the goal state to be placed at the initial state
        if(x == 0 && y == 0){
            mdpWeights[pMDP] = 0.0;
            continue;
        }
        if(x == 0 && y == 2){
            pTrueModel = pMDP;
        }
        mdpWeights[pMDP] = 1.0/3;
        mdpGoals[pMDP] = std::make_pair(x, y);
    }

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = "0";
    stateMap["y"] = "0";
    State initState(stateMap);
    int trials = 3000;
    int horizon = 10;
    BAMCPSolver solver;
    MultiModelMDP mmdp = MultiModelMDP(mdpWeights);
    std::shared_ptr<MCTSDecisionNode> node = solver.runMCTS(mmdp, initState, horizon, trials);
    //REQUIRE(node->getBestAction() == "up");
}
