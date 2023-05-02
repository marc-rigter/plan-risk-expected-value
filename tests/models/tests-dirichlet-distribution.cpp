/* This file implements tests for the MDP class. */

#include <unordered_map>
#include <iostream>
#include "catch.h"
#include "state.h"
#include "dirichlet_distribution.h"
#include "tied_dirichlet_distribution.h"

TEST_CASE("State dirichlet distribution"){

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["stage"] = "1";
    stateMap["money"] = "10";
    State state1(stateMap);

    stateMap["stage"] = "1";
    stateMap["money"] = "11";
    State state2(stateMap);

    stateMap["stage"] = "1";
    stateMap["money"] = "12";
    State state3(stateMap);

    std::unordered_map<State, float, StateHash> priorStateCounts;
    priorStateCounts[state1] = 1.0;
    priorStateCounts[state2] = 1.0;
    priorStateCounts[state3] = 1.0;

    DirichletDistribution dist(priorStateCounts);

    std::unordered_map<State, float, StateHash> successorProbs;
    successorProbs = dist.getExpectedDistribution();
    REQUIRE(cmpf(successorProbs[state1], 0.33333));
    REQUIRE(cmpf(successorProbs[state2], 0.33333));
    REQUIRE(cmpf(successorProbs[state3], 0.33333));

    dist.observe(state3);
    dist.observe(state3);
    dist.observe(state3);

    successorProbs = dist.getExpectedDistribution();
    REQUIRE(cmpf(successorProbs[state1], 0.166667));
    REQUIRE(cmpf(successorProbs[state2], 0.166667));
    REQUIRE(cmpf(successorProbs[state3], 0.666667));
}


TEST_CASE("Pseudo state dirichlet distribution"){

    std::unordered_map<std::string, float> priorPseudoStateCounts;
    priorPseudoStateCounts["left"] = 1.0;
    priorPseudoStateCounts["straight"] = 1.0;
    priorPseudoStateCounts["right"] = 1.0;

    TiedDirichletDistribution dist(priorPseudoStateCounts);

    std::unordered_map<std::string, float> successorProbs;
    successorProbs = dist.getExpectedDistribution();
    REQUIRE(cmpf(successorProbs["left"], 0.33333));
    REQUIRE(cmpf(successorProbs["straight"], 0.33333));
    REQUIRE(cmpf(successorProbs["right"], 0.33333));

    dist.observe("right");
    dist.observe("right");
    dist.observe("right");

    successorProbs = dist.getExpectedDistribution();
    REQUIRE(cmpf(successorProbs["left"], 0.166667));
    REQUIRE(cmpf(successorProbs["straight"], 0.166667));
    REQUIRE(cmpf(successorProbs["right"], 0.666667));
}
