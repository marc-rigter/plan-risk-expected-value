/* This file implements tests for the MDP class. */
#include <unordered_map>
#include <iostream>
#include "catch.h"
#include "state.h"


TEST_CASE("States with same values are equal") {
    std::unordered_map<std::string, std::string> stateMap1;
    stateMap1["x"] = "5";
    stateMap1["y"] = "9";
    State state1(stateMap1);

    std::unordered_map<std::string, std::string> stateMap2;
    stateMap2["x"] = "5";
    stateMap2["y"] = "9";
    State state2(stateMap2);

    REQUIRE(state1 == state2);
}
