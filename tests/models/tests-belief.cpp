/* This file implements tests for the MDP class. */

#include <unordered_map>
#include <iostream>
#include "catch.h"
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "mdp_examples.h"
#include "finite_mdp_belief.h"
#include "value_iteration.h"

TEST_CASE("Convert finite mdp belief to bamdp"){
    int stages = 2;
    std::shared_ptr<MDP> templateMDP = makeBettingMDP(1.0, stages);
    std::shared_ptr<MDP> testMDP1 = makeBettingMDP(0.5, stages);
    std::shared_ptr<MDP> testMDP2 = makeBettingMDP(0.8, stages);
    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights;
    mdpWeights[testMDP1] = 0.4;
    mdpWeights[testMDP2] = 0.6;
    FiniteMDPBelief b(mdpWeights);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State initState(stateMap);

    std::shared_ptr<MDP> pBAMDP = b.toBamdp(templateMDP, initState, stages+1);

    std::unordered_map<State, float, StateHash> value;
    std::unordered_map<State, std::string, StateHash> policy;

    VI vi;
    std::tie(value, policy) = vi.valueIteration(*pBAMDP, true);

    std::unordered_map<std::string, std::string> stateMap2;
    stateMap2["history"] = initState.toString();
    stateMap2["t"] = "0";
    State s(stateMap2);

    REQUIRE(cmpf(value[s], 13.6));
}

TEST_CASE("Convert dirichlet belief to bamdp"){
    int stages = 2;
    std::vector<int> actions{0, 1, 2, 3, 4, 5};
    std::shared_ptr<MDP> templateMDP = makeBettingMDP(1.0, stages, actions);

    // initial belief
    float priorWinCount = 2.0;
    float priorLoseCount = 1.0;
    std::shared_ptr<FullyTiedDirichletBelief> pBelief = getBettingGameBelief(
            priorWinCount,
            priorLoseCount,
            stages,
            templateMDP);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State initState(stateMap);

    std::shared_ptr<MDP> pBAMDP = pBelief->toBamdp(templateMDP, initState, stages+1);

    std::unordered_map<State, float, StateHash> value;
    std::unordered_map<State, std::string, StateHash> policy;

    VI vi;
    std::tie(value, policy) = vi.valueIteration(*pBAMDP, true);

    std::unordered_map<std::string, std::string> stateMap2;
    stateMap2["history"] = initState.toString();
    stateMap2["t"] = "0";
    State s(stateMap2);

    REQUIRE(cmpf(value[s], 13.3333));
}
