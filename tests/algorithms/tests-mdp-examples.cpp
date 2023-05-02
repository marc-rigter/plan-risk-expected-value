/* This file implements tests for solving BAMDPs with MCTS */

#include <iostream>
#include <string>
#include "catch.h"
#include "multimodel_mdp.h"
#include "mdp_examples.h"
#include "ssp_mdp_examples.h"
#include "domains.h"
#include "value_iteration.h"
#include "utils.h"

TEST_CASE("real_traffic"){
    bool maximise = false; // cost minimisation

    std::shared_ptr<MDP> pMDP;
    State initState;

    std::string dataset = "tests/datasets/am_district7_avoid_highways.csv";

    int startID = 762398; // Buena park
    int goalID = 770599; // Sylmar
    std::tie(initState, pMDP) = sspRealTrafficDomain(dataset, startID, goalID, true, false);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["id"] = "763669";
    stateMap["cost"] = "0.0";
    State s(stateMap);

    stateMap["id"] = "770599";
    stateMap["cost"] = "0.0";
    State goalState(stateMap);

    std::vector<std::string> enabledActions = pMDP->getEnabledActions(s);

    REQUIRE(std::find(enabledActions.begin(), enabledActions.end(), "767494") != enabledActions.end());
    REQUIRE(std::find(enabledActions.begin(), enabledActions.end(), "759566") != enabledActions.end());
    REQUIRE(std::find(enabledActions.begin(), enabledActions.end(), "767495") != enabledActions.end());

    std::vector<std::string> enabledActionsGoal = pMDP->getEnabledActions(goalState);

    REQUIRE(enabledActionsGoal.size() == 1);
    REQUIRE(enabledActionsGoal.front() == "end");

    std::unordered_map<State, float, StateHash> transProbsGoal = pMDP->getTransitionProbs(goalState, "end");
    REQUIRE(cmpf(transProbsGoal[goalState], 1.0));

    stateMap["id"] = "767494";
    stateMap["cost"] = "10.1833";
    State sNext(stateMap);
    std::unordered_map<State, float, StateHash> transProbs = pMDP->getTransitionProbs(s, "767494");

    REQUIRE(cmpf(transProbs[sNext], 1.0));

    transProbs = pMDP->getTransitionProbs(s, "759566");
    stateMap["id"] = "759566";
    stateMap["cost"] = "10.8422";
    State sNext1(stateMap);

    stateMap["id"] = "759566";
    stateMap["cost"] = "9.22363";
    State sNext2(stateMap);

    stateMap["id"] = "759566";
    stateMap["cost"] = "8.02554";
    State sNext3(stateMap);

    stateMap["id"] = "759566";
    stateMap["cost"] = "7.10291";
    State sNext4(stateMap);

    REQUIRE(transProbs.size() == 4);
    REQUIRE(cmpf(transProbs[sNext1], 0.0243056));
    REQUIRE(cmpf(transProbs[sNext2], 0.336806));
    REQUIRE(cmpf(transProbs[sNext3], 0.267361));
    REQUIRE(cmpf(transProbs[sNext4], 0.371528));

    stateMap["id"] = "759566";
    stateMap["cost"] = "0.0";
    State baseState(stateMap);

    transProbs = pMDP->getTransitionProbs(sNext1, "cost");
    REQUIRE(cmpf(transProbs[baseState], 1.0));

    transProbs = pMDP->getTransitionProbs(sNext2, "cost");
    REQUIRE(cmpf(transProbs[baseState], 1.0));

    transProbs = pMDP->getTransitionProbs(sNext3, "cost");
    REQUIRE(cmpf(transProbs[baseState], 1.0));

    float reward = pMDP->getReward(s, "759566");
    REQUIRE(cmpf(reward, 0.0));

    reward = pMDP->getReward(sNext1, "cost");
    REQUIRE(cmpf(reward, std::stof(sNext1.getValue("cost"))));

    reward = pMDP->getReward(sNext2, "cost");
    REQUIRE(cmpf(reward, std::stof(sNext2.getValue("cost"))));

    reward = pMDP->getReward(sNext3, "cost");
    REQUIRE(cmpf(reward, std::stof(sNext3.getValue("cost"))));
}


TEST_CASE("Betting game example 1"){
    MDP testMDP = *makeBettingMDP(1.0, 5);
    std::unordered_map<State, float, StateHash> value;
    std::unordered_map<State, std::string, StateHash> policy;

    VI vi;
    std::tie(value, policy) = vi.valueIteration(testMDP, true);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State state(stateMap);
    REQUIRE(cmpf(value[state], 35.0));
}

TEST_CASE("Betting game example 2"){
    MDP testMDP = *makeBettingMDP(0.8, 5);
    std::unordered_map<State, float, StateHash> value;
    std::unordered_map<State, std::string, StateHash> policy;

    VI vi;
    std::tie(value, policy) = vi.valueIteration(testMDP, true);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State state(stateMap);
    REQUIRE(cmpf(value[state], 24.6016));
}

TEST_CASE("Betting game example 3"){
    MDP testMDP = *makeBettingMDP(0., 5);
    std::unordered_map<State, float, StateHash> value;
    std::unordered_map<State, std::string, StateHash> policy;

    VI vi;
    std::tie(value, policy) = vi.valueIteration(testMDP, true);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State state(stateMap);
    REQUIRE(cmpf(value[state], 10.0));
}

TEST_CASE("Medical decision making"){
    int numDays = 7;
    int seed = numDays;
    MDP testMDP = *makeMedicalMDP(numDays, seed);
    std::unordered_map<State, float, StateHash> value;
    std::unordered_map<State, std::string, StateHash> policy;

    VI vi;
    std::tie(value, policy) = vi.valueIteration(testMDP, true);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["health"] = "5";
    State state(stateMap);
    REQUIRE(cmpf(value[state], 4.39669));
}


// TEST_CASE("TrafficMDP"){
//     int horizon = 8;
//     MDP testMDP = *trafficMDP(horizon);
//     std::unordered_map<State, float, StateHash> value;
//     std::unordered_map<State, std::string, StateHash> policy;
//     std::tie(value, policy) = valueIteration(testMDP, true);
//
//     std::unordered_map<std::string, std::string> stateMap;
//     stateMap["t"] = "0";
//     stateMap["x"] = "1";
//     stateMap["y"] = "0";
//     stateMap["min"] = "0";
//     State state(stateMap);
//     std::cout << value[state] << std::endl;
// }
