/* This file implements tests for value iteration */
#include <unordered_map>
#include <iostream>
#include "catch.h"
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "value_iteration.h"
#include "mdp_examples.h"

TEST_CASE("Value iteration"){
    MDP testMDP = *makeSimpleMDP(2, 2, 2.0);
    std::unordered_map<State, float, StateHash> value;
    std::unordered_map<State, std::string, StateHash> policy;

    VI vi;
    std::tie(value, policy) = vi.valueIteration(testMDP, true);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = "0";
    stateMap["y"] = "0";
    State statex0y0(stateMap);
    REQUIRE(cmpf(value[statex0y0], -6.0));
    REQUIRE(policy[statex0y0] == "up");

    stateMap["x"] = "1";
    stateMap["y"] = "0";
    State statex1y0(stateMap);
    REQUIRE(cmpf(value[statex1y0], -7.25));
    REQUIRE(policy[statex1y0] == "up");

    stateMap["x"] = "1";
    stateMap["y"] = "1";
    State statex1y1(stateMap);
    REQUIRE(cmpf(value[statex1y1], -3.5));
    REQUIRE(policy[statex1y1] == "right");
}
