/* This file implements tests for the MDP class. */

#include <unordered_map>
#include <iostream>
#include "catch.h"
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "mdp_examples.h"
#include "fully_tied_dirichlet_belief.h"
#include "mcts_bamdp_cvar_sg.h"

TEST_CASE("Betting game fully tied belief"){
    int stages = 5;
    std::shared_ptr<MDP> templateMDP = makeBettingMDP(1.0, stages);

    float priorWinCounts = 3.0;
    float priorLoseCounts = 1.0;

    std::shared_ptr<FullyTiedDirichletBelief> b = getBettingGameBelief(
            priorWinCounts,
            priorLoseCounts,
            stages,
            templateMDP);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "2";
    stateMap["money"] = "10";
    State initState(stateMap);

    stateMap["t"] = "3";
    stateMap["money"] = "10";
    State nextStep(stateMap);

    stateMap["t"] = "3";
    stateMap["money"] = "13";
    State stateWin(stateMap);

    stateMap["t"] = "3";
    stateMap["money"] = "7";
    State stateLose(stateMap);

    stateMap["t"] = "4";
    stateMap["money"] = "8";
    State initState2(stateMap);

    stateMap["t"] = "5";
    stateMap["money"] = "10";
    State stateWin2(stateMap);

    stateMap["t"] = "5";
    stateMap["money"] = "6";
    State stateLose2(stateMap);

    stateMap["t"] = "6";
    stateMap["money"] = "-1";
    State terminalState(stateMap);

    std::unordered_map<State, float, StateHash> probs = b->getBeliefTransitionProbs(initState, "3");
    REQUIRE(cmpf(probs[stateWin], 0.75));
    REQUIRE(cmpf(probs[stateLose], 0.25));

    probs = b->getBeliefTransitionProbs(initState2, "2");
    REQUIRE(cmpf(probs[stateWin2], 0.75));
    REQUIRE(cmpf(probs[stateLose2], 0.25));

    probs = b->getBeliefTransitionProbs(stateWin2, "end");
    REQUIRE(cmpf(probs[terminalState], 1.0));

    b->updateBelief(initState, "3", stateWin);
    b->updateBelief(initState, "3", stateWin);
    b->updateBelief(initState, "3", stateLose);

    probs = b->getBeliefTransitionProbs(initState, "3");
    REQUIRE(cmpf(probs[stateWin], 0.71428));
    REQUIRE(cmpf(probs[stateLose], 0.28572));

    probs = b->getBeliefTransitionProbs(initState2, "2");
    REQUIRE(cmpf(probs[stateWin2], 0.71428));
    REQUIRE(cmpf(probs[stateLose2], 0.28572));

    probs = b->getBeliefTransitionProbs(stateWin2, "end");
    REQUIRE(cmpf(probs[terminalState], 1.0));

    // observing a transition after the 0 action should result in no change to
    // the probabilites.
    b->updateBelief(initState, "0", nextStep);
    probs = b->getBeliefTransitionProbs(initState, "3");
    REQUIRE(cmpf(probs[stateWin], 0.71428));
    REQUIRE(cmpf(probs[stateLose], 0.28572));

    probs = b->getBeliefTransitionProbs(initState2, "2");
    REQUIRE(cmpf(probs[stateWin2], 0.71428));
    REQUIRE(cmpf(probs[stateLose2], 0.28572));

    probs = b->getBeliefTransitionProbs(stateWin2, "end");
    REQUIRE(cmpf(probs[terminalState], 1.0));

    // check the transition probabilities of the expected MDP
    std::shared_ptr<MDP> pExpectedMDP = b->getExpectedMDP();
    std::unordered_map<State, float, StateHash> expMDPProbs = pExpectedMDP->getTransitionProbs(initState, "3");
    REQUIRE(cmpf(expMDPProbs[stateWin], 0.71428));
    REQUIRE(cmpf(expMDPProbs[stateLose], 0.28572));

    expMDPProbs = pExpectedMDP->getTransitionProbs(initState2, "2");
    REQUIRE(cmpf(expMDPProbs[stateWin2], 0.71428));
    REQUIRE(cmpf(expMDPProbs[stateLose2], 0.28572));
}

TEST_CASE("Sample betting game fully tied belief"){
    int stages = 5;
    std::shared_ptr<MDP> templateMDP = makeBettingMDP(1.0, stages);

    float priorWinCounts = 3.0;
    float priorLoseCounts = 1.0;

    std::shared_ptr<FullyTiedDirichletBelief> b = getBettingGameBelief(
            priorWinCounts,
            priorLoseCounts,
            stages,
            templateMDP);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "2";
    stateMap["money"] = "10";
    State initState(stateMap);

    stateMap["t"] = "3";
    stateMap["money"] = "13";
    State stateWin(stateMap);

    stateMap["t"] = "3";
    stateMap["money"] = "11";
    State stateSmallWin(stateMap);

    stateMap["t"] = "3";
    stateMap["money"] = "7";
    State stateLose(stateMap);

    stateMap["t"] = "3";
    stateMap["money"] = "9";
    State stateSmallLose(stateMap);

    stateMap["t"] = "2";
    stateMap["money"] = "15";
    State initState2(stateMap);

    stateMap["t"] = "3";
    stateMap["money"] = "18";
    State stateWin2(stateMap);

    stateMap["t"] = "3";
    stateMap["money"] = "12";
    State stateLose2(stateMap);

    // check that the sampled probabilities are the same for all states and actions
    std::shared_ptr<MDP> pSample = b->sampleModel();
    std::unordered_map<State, float, StateHash> probs = pSample->getTransitionProbs(initState, "3");
    std::unordered_map<State, float, StateHash> probs2 = pSample->getTransitionProbs(initState, "1");
    std::unordered_map<State, float, StateHash> probs3 = pSample->getTransitionProbs(initState2, "3");

    REQUIRE(cmpf(probs[stateWin], probs2[stateSmallWin]));
    REQUIRE(cmpf(probs[stateWin], probs3[stateWin2]));
    REQUIRE(cmpf(probs[stateLose], probs2[stateSmallLose]));
    REQUIRE(cmpf(probs[stateLose], probs3[stateLose2]));
}

TEST_CASE("Mars rover fully tied belief"){
    int horizon = 5;
    std::shared_ptr<MDP> pRoverMDP = marsRoverMDP(horizon);
    bool fullyTied = true;
    float initCount = 1.0;
    std::shared_ptr<Belief> b = getMarsRoverBelief(pRoverMDP, horizon, initCount, fullyTied);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["x"] = "2";
    stateMap["y"] = "7";
    State sInit(stateMap);

    stateMap["t"] = "1";
    stateMap["x"] = "2";
    stateMap["y"] = "6";
    State sUp(stateMap);

    stateMap["t"] = "1";
    stateMap["x"] = "3";
    stateMap["y"] = "6";
    State sUpRight(stateMap);

    stateMap["t"] = "1";
    stateMap["x"] = "1";
    stateMap["y"] = "6";
    State sUpLeft(stateMap);

    stateMap["t"] = "1";
    stateMap["x"] = "3";
    stateMap["y"] = "7";
    State sRight(stateMap);

    stateMap["t"] = "2";
    stateMap["x"] = "5";
    stateMap["y"] = "5";
    State sInit2(stateMap);

    stateMap["t"] = "3";
    stateMap["x"] = "5";
    stateMap["y"] = "4";
    State sUp2(stateMap);

    stateMap["t"] = "3";
    stateMap["x"] = "6";
    stateMap["y"] = "4";
    State sUpRight2(stateMap);

    stateMap["t"] = "3";
    stateMap["x"] = "4";
    stateMap["y"] = "4";
    State sUpLeft2(stateMap);

    stateMap["t"] = "3";
    stateMap["x"] = "6";
    stateMap["y"] = "5";
    State sRight2(stateMap);

    stateMap["t"] = "2";
    stateMap["x"] = "1";
    stateMap["y"] = "2";
    State sGoal(stateMap);

    stateMap["t"] = "4";
    stateMap["x"] = "6";
    stateMap["y"] = "1";
    State sLastStep(stateMap);

    stateMap["t"] = "5";
    stateMap["x"] = "-1";
    stateMap["y"] = "-1";
    State sTerminal(stateMap);

    std::unordered_map<State, float, StateHash> probs = b->getBeliefTransitionProbs(sInit, "u");
    REQUIRE(cmpf(probs[sUp], 0.33333));
    REQUIRE(cmpf(probs[sUpRight], 0.33333));
    REQUIRE(cmpf(probs[sUpLeft], 0.33333));

    probs = b->getBeliefTransitionProbs(sInit2, "r");
    REQUIRE(cmpf(probs[sUpRight2], 0.33333));
    REQUIRE(cmpf(probs[sRight2], 0.33333));

    b->updateBelief(sInit, "u", sUpRight);
    b->updateBelief(sInit, "u", sUpRight);

    probs = b->getBeliefTransitionProbs(sInit, "u");
    REQUIRE(cmpf(probs[sUp], 0.2));
    REQUIRE(cmpf(probs[sUpRight], 0.6));
    REQUIRE(cmpf(probs[sUpLeft], 0.2));

    probs = b->getBeliefTransitionProbs(sInit2, "u");
    REQUIRE(cmpf(probs[sUp2], 0.2));
    REQUIRE(cmpf(probs[sUpRight2], 0.6));
    REQUIRE(cmpf(probs[sUpLeft2], 0.2));

    // observations of up action should also change probs for other actions
    probs = b->getBeliefTransitionProbs(sInit2, "r");
    REQUIRE(cmpf(probs[sUpRight2], 0.2));
    REQUIRE(cmpf(probs[sRight2], 0.2));

    probs = b->getBeliefTransitionProbs(sGoal, "end");
    REQUIRE(cmpf(probs[sTerminal], 1.0));

    probs = b->getBeliefTransitionProbs(sLastStep, "end");
    REQUIRE(cmpf(probs[sTerminal], 1.0));

    probs = b->getBeliefTransitionProbs(sTerminal, "end");
    REQUIRE(cmpf(probs[sTerminal], 1.0));
}
