/* This file implements tests for the MDP class. */

#include <unordered_map>
#include <iostream>
#include "catch.h"
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "mdp_examples.h"
#include "tied_dirichlet_belief.h"
#include "mcts_bamdp_cvar_sg.h"

std::shared_ptr<TiedDirichletBelief> getTiedDirichetBelief(std::shared_ptr<MDP> pMDP, int xGoal, int yGoal){
    std::unordered_map<std::string, float> priorPseudoStateCounts;
    priorPseudoStateCounts["move"] = 1.0;
    priorPseudoStateCounts["staySame"] = 1.0;

    std::vector<std::string> actionList{"up", "right", "down", "left"};
    std::vector<std::string> pseudoStates{"move", "staySame"};
    std::unordered_map<std::string, std::shared_ptr<TiedDirichletDistribution>> dirichletDistributions;

    for(std::string act : actionList){
        std::shared_ptr<TiedDirichletDistribution> dist = std::make_shared<TiedDirichletDistribution>(priorPseudoStateCounts);
        dirichletDistributions[act] = dist;
    }

    typedef std::unordered_map<std::string, State> successorMapping;
    std::shared_ptr<std::unordered_map<State, std::unordered_map<std::string, successorMapping>, StateHash>> pseudoStateMapping = std::make_shared<std::unordered_map<State, std::unordered_map<std::string, successorMapping>, StateHash>>();

    int xDelta = 0;
    int yDelta = 0;
    int xOld;
    int yOld;
    int xMax = 2;
    int yMax = 2;
    int xNew;
    int yNew;

    std::unordered_map<std::string, std::string> stateMap;
    for(State s : pMDP->enumerateStates()){
        std::unordered_map<std::string, successorMapping> actionMap;
        xOld = std::stoi(s.getValue("x"));
        yOld = std::stoi(s.getValue("y"));

        for(std::string act : pMDP->getEnabledActions(s)){
            successorMapping map;

            for(std::string outcome : pseudoStates){
                if(act == "up"){
                    xDelta = 0;
                    yDelta = 1;
                }else if(act == "right"){
                    xDelta = 1;
                    yDelta = 0;
                }else if(act == "down"){
                    xDelta = 0;
                    yDelta = -1;
                }else if(act== "left"){
                    xDelta = -1;
                    yDelta = 0;
                }

                if(xOld == xGoal && yOld == yGoal){
                    xNew = -1;
                    yNew = -1;
                    stateMap["x"] = std::to_string(xNew);
                    stateMap["y"] = std::to_string(yNew);
                    State sNext(stateMap);
                    map[outcome] = sNext;
                    continue;
                }

                if(xOld == -1 && yOld == -1){
                    stateMap["x"] = std::to_string(xOld);
                    stateMap["y"] = std::to_string(yOld);
                    State sNext(stateMap);
                    map[outcome] = sNext;
                }

                // if at the goal state the successor is always -1, -1
                if(outcome == "staySame"){
                    xNew = xOld;
                    yNew = yOld;
                }else{
                    xNew = xOld + xDelta;
                    yNew = yOld + yDelta;
                }

                if(xNew > xMax || yNew > yMax || xNew < 0 || yNew < 0){
                    continue;
                }

                stateMap["x"] = std::to_string(xNew);
                stateMap["y"] = std::to_string(yNew);
                State sNext(stateMap);
                map[outcome] = sNext;
            }

            actionMap[act] = map;
        }
        (*pseudoStateMapping)[s] = actionMap;
    }

    std::shared_ptr<TiedDirichletBelief> b = std::make_shared<TiedDirichletBelief>(pMDP, dirichletDistributions, pseudoStateMapping);
    return b;
}

TEST_CASE("Dirichlet belief update"){
    int goalX = 1;
    int goalY = 1;
    std::shared_ptr<MDP> pMDP = makeSimpleMDP(goalX, goalY, 0.0);
    std::unordered_map<std::string, std::string> stateMap;

    std::shared_ptr<TiedDirichletBelief> b = getTiedDirichetBelief(pMDP, goalX, goalY);

    stateMap["x"] = "0";
    stateMap["y"] = "0";
    State sInit(stateMap);

    stateMap["x"] = "0";
    stateMap["y"] = "1";
    State sUp(stateMap);

    stateMap["x"] = "1";
    stateMap["y"] = "0";
    State sRight(stateMap);

    stateMap["x"] = "1";
    stateMap["y"] = "0";
    State sInit2(stateMap);

    stateMap["x"] = "1";
    stateMap["y"] = "1";
    State sUp2(stateMap);

    stateMap["x"] = "2";
    stateMap["y"] = "0";
    State sRight2(stateMap);

    std::unordered_map<State, float, StateHash> probs = b->getBeliefTransitionProbs(sInit, "up");
    REQUIRE(cmpf(probs[sInit], 0.5));
    REQUIRE(cmpf(probs[sUp], 0.5));

    probs = b->getBeliefTransitionProbs(sInit2, "up");
    REQUIRE(cmpf(probs[sInit2], 0.5));
    REQUIRE(cmpf(probs[sUp2], 0.5));

    probs = b->getBeliefTransitionProbs(sInit, "right");
    REQUIRE(cmpf(probs[sInit], 0.5));
    REQUIRE(cmpf(probs[sRight], 0.5));

    probs = b->getBeliefTransitionProbs(sInit2, "right");
    REQUIRE(cmpf(probs[sInit2], 0.5));
    REQUIRE(cmpf(probs[sRight2], 0.5));

    b->updateBelief(sInit, "up", sUp);
    b->updateBelief(sInit, "up", sUp);

    probs = b->getBeliefTransitionProbs(sInit, "up");
    REQUIRE(cmpf(probs[sInit], 0.25));
    REQUIRE(cmpf(probs[sUp], 0.75));

    probs = b->getBeliefTransitionProbs(sInit2, "up");
    REQUIRE(cmpf(probs[sInit2], 0.25));
    REQUIRE(cmpf(probs[sUp2], 0.75));

    b->updateBelief(sInit, "right", sRight);

    probs = b->getBeliefTransitionProbs(sInit, "right");
    REQUIRE(cmpf(probs[sInit], 0.33333));
    REQUIRE(cmpf(probs[sRight], 0.66667));

    probs = b->getBeliefTransitionProbs(sInit2, "right");
    REQUIRE(cmpf(probs[sInit2], 0.33333));
    REQUIRE(cmpf(probs[sRight2], 0.66667));
}

TEST_CASE("Sample from dirichlet belief"){
    int goalX = 1;
    int goalY = 1;
    std::shared_ptr<MDP> pMDP = makeSimpleMDP(goalX, goalY, 0.0);
    std::unordered_map<std::string, std::string> stateMap;

    std::shared_ptr<TiedDirichletBelief> b = getTiedDirichetBelief(pMDP, goalX, goalY);

    stateMap["x"] = "0";
    stateMap["y"] = "0";
    State sInit(stateMap);

    stateMap["x"] = "0";
    stateMap["y"] = "1";
    State sUp(stateMap);

    stateMap["x"] = "1";
    stateMap["y"] = "0";
    State sRight(stateMap);

    stateMap["x"] = "1";
    stateMap["y"] = "0";
    State sInit2(stateMap);

    stateMap["x"] = "1";
    stateMap["y"] = "1";
    State sUp2(stateMap);

    stateMap["x"] = "2";
    stateMap["y"] = "0";
    State sRight2(stateMap);

    // check sampled probability sample
    std::shared_ptr<MDP> pSample = b->sampleModel();
    std::unordered_map<State, float, StateHash> probs = pSample->getTransitionProbs(sInit, "up");
    std::unordered_map<State, float, StateHash> probs2 = pSample->getTransitionProbs(sInit2, "up");
    REQUIRE(cmpf(probs[sInit], probs2[sInit2]));
    REQUIRE(cmpf(probs[sUp], probs2[sUp2]));


    probs = pSample->getTransitionProbs(sInit, "right");
    probs2 = pSample->getTransitionProbs(sInit2, "right");
    REQUIRE(cmpf(probs[sInit], probs2[sInit2]));
    REQUIRE(cmpf(probs[sRight], probs2[sRight2]));
}

TEST_CASE("BAMDP Dirichlet Belief"){
    int initX = 0;
    int initY = 0;
    int goalX = 1;
    int goalY = 1;
    std::shared_ptr<MDP> pMDP = makeSimpleMDP(goalX, goalY, 0.0);
    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = std::to_string(initX);
    stateMap["y"] = std::to_string(initY);
    State initState(stateMap);

    std::shared_ptr<TiedDirichletBelief> pBelief = getTiedDirichetBelief(pMDP, goalX, goalY);

    std::string strat = "bayesOpt";
    BamdpCvarMCTS m(1.0, 0.3, strat);
    std::shared_ptr<BamdpCvarDecisionNode> rootNode;

    rootNode = std::make_shared<BamdpCvarDecisionNode>(pMDP, pBelief, initState, 0.3);

    int horizon = 6;
    int burninTrials = 100;
    int agentTrials = 30;
    int advTrials = 40;

    std::shared_ptr<MDP> realMDP = pMDP;

    CvarGameHist history = m.executeEpisode(realMDP, rootNode, horizon, burninTrials, agentTrials, advTrials);
    bool verbose = false;
    history.printPath(verbose);
}

TEST_CASE("Mars rover belief"){
    int horizon = 5;
    std::shared_ptr<MDP> pRoverMDP = marsRoverMDP(horizon);
    bool fullyTied = false;
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

    probs = b->getBeliefTransitionProbs(sInit2, "r");
    REQUIRE(cmpf(probs[sUpRight2], 0.33333));
    REQUIRE(cmpf(probs[sRight2], 0.33333));

    b->updateBelief(sInit, "r", sRight);
    b->updateBelief(sInit, "r", sRight);
    b->updateBelief(sInit, "r", sRight);
    b->updateBelief(sInit, "r", sRight);

    probs = b->getBeliefTransitionProbs(sInit2, "r");
    REQUIRE(cmpf(probs[sUpRight2], 0.142857));
    REQUIRE(cmpf(probs[sRight2], 0.714285));

    probs = b->getBeliefTransitionProbs(sInit, "r");
    REQUIRE(cmpf(probs[sUpRight], 0.142857));
    REQUIRE(cmpf(probs[sRight], 0.714285));

    probs = b->getBeliefTransitionProbs(sGoal, "end");
    REQUIRE(cmpf(probs[sTerminal], 1.0));

    probs = b->getBeliefTransitionProbs(sLastStep, "end");
    REQUIRE(cmpf(probs[sTerminal], 1.0));

    probs = b->getBeliefTransitionProbs(sTerminal, "end");
    REQUIRE(cmpf(probs[sTerminal], 1.0));
}

// TEST_CASE("TrafficBelief"){
//     int horizon = 5;
//     std::shared_ptr<MDP> pTrafficMDP = trafficMDP(horizon);
//     std::vector<float> initCounts{1.0, 1.0, 1.0};
//     std::shared_ptr<Belief> b = getTrafficBelief(pTrafficMDP, horizon, initCounts);
//
//     std::tuple<std::unordered_map<std::string, int>, std::unordered_map<std::string, int>, std::unordered_map<std::string, int>, std::unordered_map<std::string, int>> outcomes = getTrafficOutcomes();
//     std::unordered_map<std::string, int> outcomesVeryQuiet = std::get<0>(outcomes);
//     std::unordered_map<std::string, int> outcomesQuiet = std::get<1>(outcomes);
//     std::unordered_map<std::string, int> outcomesAverage = std::get<2>(outcomes);
//     std::unordered_map<std::string, int> outcomesBusy = std::get<3>(outcomes);
//
//     std::unordered_map<std::string, std::string> stateMap;
//     stateMap["t"] = "0";
//     stateMap["min"] = "0";
//     stateMap["x"] = "1";
//     stateMap["y"] = "0";
//     State sInit(stateMap);
//
//     stateMap["t"] = "5";
//     stateMap["min"] = "-1";
//     stateMap["x"] = "-1";
//     stateMap["y"] = "-1";
//     State sTerminal(stateMap);
//
//     stateMap["t"] = "3";
//     stateMap["min"] = "5";
//     stateMap["x"] = "0";
//     stateMap["y"] = "5";
//     State sGoal(stateMap);
//
//     stateMap["t"] = "4";
//     stateMap["min"] = "5";
//     stateMap["x"] = "1";
//     stateMap["y"] = "2";
//     State sLastStep(stateMap);
//
//     stateMap["t"] = "1";
//     stateMap["min"] = std::to_string(outcomesAverage["fast"]);
//     stateMap["x"] = "1";
//     stateMap["y"] = "1";
//     State sInitUpFast(stateMap);
//
//     stateMap["t"] = "1";
//     stateMap["min"] = std::to_string(outcomesAverage["medium"]);
//     stateMap["x"] = "1";
//     stateMap["y"] = "1";
//     State sInitUpMedium(stateMap);
//
//     stateMap["t"] = "1";
//     stateMap["min"] = std::to_string(outcomesAverage["slow"]);
//     stateMap["x"] = "1";
//     stateMap["y"] = "1";
//     State sInitUpSlow(stateMap);
//
//     stateMap["t"] = "0";
//     stateMap["min"] = "0";
//     stateMap["x"] = "1";
//     stateMap["y"] = "1";
//     State sInit2(stateMap);
//
//     stateMap["t"] = "1";
//     stateMap["min"] = std::to_string(outcomesAverage["fast"]);
//     stateMap["x"] = "1";
//     stateMap["y"] = "2";
//     State sInit2UpFast(stateMap);
//
//     stateMap["t"] = "1";
//     stateMap["min"] = std::to_string(outcomesAverage["medium"]);
//     stateMap["x"] = "1";
//     stateMap["y"] = "2";
//     State sInit2UpMedium(stateMap);
//
//     stateMap["t"] = "1";
//     stateMap["min"] = std::to_string(outcomesAverage["slow"]);
//     stateMap["x"] = "1";
//     stateMap["y"] = "2";
//     State sInit2UpSlow(stateMap);
//
//     stateMap["t"] = "0";
//     stateMap["min"] = "0";
//     stateMap["x"] = "3";
//     stateMap["y"] = "0";
//     State sQuietInit(stateMap);
//
//     stateMap["t"] = "1";
//     stateMap["min"] = std::to_string(outcomesQuiet["fast"]);
//     stateMap["x"] = "2";
//     stateMap["y"] = "0";
//     State sQuietLeftFast(stateMap);
//
//     stateMap["t"] = "1";
//     stateMap["min"] = std::to_string(outcomesVeryQuiet["fast"]);
//     stateMap["x"] = "3";
//     stateMap["y"] = "1";
//     State sQuietUpFast(stateMap);
//
//     stateMap["t"] = "1";
//     stateMap["min"] = std::to_string(outcomesVeryQuiet["medium"]);
//     stateMap["x"] = "3";
//     stateMap["y"] = "1";
//     State sQuietUpMedium(stateMap);
//
//     stateMap["t"] = "1";
//     stateMap["min"] = std::to_string(outcomesVeryQuiet["slow"]);
//     stateMap["x"] = "3";
//     stateMap["y"] = "1";
//     State sQuietUpSlow(stateMap);
//
//     std::unordered_map<State, float, StateHash> probs = b->getBeliefTransitionProbs(sInit, "average_up");
//     REQUIRE(cmpf(probs[sInitUpFast], 0.33333));
//     REQUIRE(cmpf(probs[sInitUpMedium], 0.33333));
//     REQUIRE(cmpf(probs[sInitUpSlow], 0.33333));
//
//     b->updateBelief(sInit, "average_up", sInitUpFast);
//     b->updateBelief(sInit, "average_up", sInitUpMedium);
//     b->updateBelief(sInit, "average_up", sInitUpMedium);
//
//     probs = b->getBeliefTransitionProbs(sInit, "average_up");
//     REQUIRE(cmpf(probs[sInitUpFast], 0.33333));
//     REQUIRE(cmpf(probs[sInitUpMedium], 0.5));
//     REQUIRE(cmpf(probs[sInitUpSlow], 0.16666));
//
//     probs = b->getBeliefTransitionProbs(sInit2, "average_up");
//     REQUIRE(cmpf(probs[sInit2UpFast], 0.33333));
//     REQUIRE(cmpf(probs[sInit2UpMedium], 0.5));
//     REQUIRE(cmpf(probs[sInit2UpSlow], 0.16666));
//
//     probs = b->getBeliefTransitionProbs(sQuietInit, "very_quiet_up");
//     REQUIRE(cmpf(probs[sQuietUpFast], 0.33333));
//     REQUIRE(cmpf(probs[sQuietUpMedium], 0.33333));
//     REQUIRE(cmpf(probs[sQuietUpSlow], 0.33333));
//
//
//     b->updateBelief(sQuietInit, "very_quiet_up", sQuietUpFast);
//     probs = b->getBeliefTransitionProbs(sQuietInit, "very_quiet_up");
//     REQUIRE(cmpf(probs[sQuietUpFast], 0.5));
//     REQUIRE(cmpf(probs[sQuietUpMedium], 0.25));
//     REQUIRE(cmpf(probs[sQuietUpSlow], 0.25));
//
//     // COMMENTING THIS OUT AS I KEEP CHANGING ACTION CONFIGS
//     // // observing the quiet up action should also influence quiet left action.
//     // probs = b->getBeliefTransitionProbs(sQuietInit, "quiet_left");
//     // REQUIRE(cmpf(probs[sQuietLeftFast], 0.5));
//
//     probs = b->getBeliefTransitionProbs(sGoal, "end");
//     REQUIRE(cmpf(probs[sTerminal], 1.0));
//
//     probs = b->getBeliefTransitionProbs(sLastStep, "end");
//     REQUIRE(cmpf(probs[sTerminal], 1.0));
//
//     std::vector<std::string> actions;
//     actions = pTrafficMDP->getEnabledActions(sGoal);
//     REQUIRE(actions.size() == 1);
//
//     actions = pTrafficMDP->getEnabledActions(sLastStep);
//     REQUIRE(actions.size() == 1);
// }
