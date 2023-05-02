/* This file implements tests for value iteration */
#include <unordered_map>
#include <iostream>
#include "catch.h"
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "cvar_value_iteration.h"
#include "value_iteration.h"
#include "mdp_examples.h"
#include "cvar_hist.h"
#include "finite_mdp_belief.h"

TEST_CASE("Cvar Value iteration"){
    MDP testMDP = *makeBettingMDP(0.8, 3);
    std::unordered_map<State, float, StateHash> value;
    std::unordered_map<State, std::string, StateHash> policy;

    VI vi;
    std::tie(value, policy) = vi.valueIteration(testMDP, true);

    std::unordered_map<State, float, StateHash> cvarValue;
    int numInterpPts = 10;
    CvarValueIteration solv(numInterpPts);
    cvarValue = solv.valueIteration(testMDP);

    for(auto kv : value){
        State s = kv.first;
        std::unordered_map<std::string, std::string> stateMap = s.getStateMapping();

        // check that the values are the same as normal VI when alpha is 1.0
        stateMap["alpha"] = "1.000000";
        State augState(stateMap);
        REQUIRE(cmpf(cvarValue[augState], value[s], 0.01));

        // for low alpha the values should be same as current money as policy
        // plays zero
        if(std::stoi(stateMap["t"]) <= 2){
            stateMap["alpha"] = "0.029155";
            State augState2(stateMap);

            // need to divide cvar value function by alpha to get normal value
            REQUIRE(cmpf(cvarValue[augState2]/0.029155, (float)std::stoi(stateMap["money"]), 0.01));
        }
    }


    // generate betting markov chain where only bet is 3
    std::vector<int> bets{3};
    MDP testMC = *makeBettingMDP(0.7, 3, bets);
    CvarValueIteration solv2(numInterpPts);
    cvarValue = solv2.valueIteration(testMC);

    // check against ground truth cvar values for the resulting markov chain.
    // note that because VI uses interpolation need wide error bounds.
    std::unordered_map<std::string, std::string> stateMap2;
    stateMap2["t"] = "0";
    stateMap2["money"] = "10";

    stateMap2["alpha"] = "1.000000";
    REQUIRE(cmpf(cvarValue[State(stateMap2)], 13.6, 0.01));

    stateMap2["alpha"] = "0.900000";
    REQUIRE(cmpf(cvarValue[State(stateMap2)]/0.9, 13.0, 0.1));

    stateMap2["alpha"] = "0.674264";
    REQUIRE(cmpf(cvarValue[State(stateMap2)]/0.674264, 10.99, 0.2));

    stateMap2["alpha"] = "0.364159";
    REQUIRE(cmpf(cvarValue[State(stateMap2)]/0.364159, 8.99, 0.2));

    stateMap2["alpha"] = "0.115443";
    REQUIRE(cmpf(cvarValue[State(stateMap2)]/0.115443, 5.59, 0.2));
}

TEST_CASE("CVaR VI execution"){
    std::shared_ptr<MDP> pMDP = makeBettingMDP(0.8, 3);
    int numInterpPts = 10;
    CvarValueIteration solv(numInterpPts);
    std::unordered_map<State, float, StateHash> cvarValue = solv.valueIteration(*pMDP);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State initState(stateMap);

    stateMap["t"] = "1";
    stateMap["money"] = "15";
    State stateWin(stateMap);

    stateMap["t"] = "1";
    stateMap["money"] = "5";
    State stateLose(stateMap);

    std::string action;
    std::unordered_map<State, float, StateHash> perturbation;
    GRBEnv env = GRBEnv();
    env.start();
    env.set(GRB_IntParam_OutputFlag, 0);


    float alpha = 0.5;
    std::tie(action, perturbation) = solv.getOptimalAction(
                                            pMDP,
                                            initState,
                                            alpha,
                                            env);


    REQUIRE(action == "5");
    REQUIRE(cmpf(perturbation[stateLose], 2.0));
    REQUIRE(cmpf(perturbation[stateWin], 0.75));

    alpha = 0.05;
    std::tie(action, perturbation) = solv.getOptimalAction(
                                            pMDP,
                                            initState,
                                            alpha,
                                            env);
    REQUIRE(action == "0");


    CvarHist result = solv.executeEpisode(pMDP, pMDP, initState, alpha);
    result.printPath();
}

TEST_CASE("CVaR VI on BAMDP"){
    int stages = 2;
    std::vector<int> bets{5};
    std::shared_ptr<MDP> mc1 = makeBettingMDP(0.5, stages, bets);
    std::shared_ptr<MDP> mc2 = makeBettingMDP(0.8, stages, bets);
    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights;
    mdpWeights[mc1] = 0.4;
    mdpWeights[mc2] = 0.6;

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State initState(stateMap);

    FiniteMDPBelief pBelief(mdpWeights);
    std::shared_ptr<MDP> pBAMDP = pBelief.toBamdp(mc1, initState, stages+1);


    int numInterpPts = 10;
    CvarValueIteration solv(numInterpPts);
    std::unordered_map<State, float, StateHash> cvarValue = solv.valueIteration(*pBAMDP);

    std::unordered_map<std::string, std::string> bamdpStateMap;
    bamdpStateMap["history"] = initState.toString();
    bamdpStateMap["t"] = "0";

    bamdpStateMap["alpha"] = "1.000000";
    REQUIRE(cmpf(cvarValue[State(bamdpStateMap)], 13.6, 0.01));

    bamdpStateMap["alpha"] = "0.674264";
    REQUIRE(cmpf(cvarValue[State(bamdpStateMap)]/0.674264, 10.51, 0.1));

    bamdpStateMap["alpha"] = "0.364159";
    REQUIRE(cmpf(cvarValue[State(bamdpStateMap)]/0.364159, 6.60, 0.1));

    bamdpStateMap["alpha"] = "0.115443";
    REQUIRE(cmpf(cvarValue[State(bamdpStateMap)]/0.115443, 0.0, 0.1));
}

TEST_CASE("Bamdp Cvar VI example"){
    int stages = 2;
    std::shared_ptr<MDP> mdp1 = makeBettingMDP(0.2, stages);
    std::shared_ptr<MDP> mdp2 = makeBettingMDP(0.9, stages);
    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights;
    mdpWeights[mdp1] = 0.2;
    mdpWeights[mdp2] = 0.8;

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State initState(stateMap);

    FiniteMDPBelief pBelief(mdpWeights);
    std::shared_ptr<MDP> pBamdp = pBelief.toBamdp(mdp1, initState, stages+1);

    int numInterpPts = 10;
    CvarValueIteration solv(numInterpPts);
    std::unordered_map<State, float, StateHash> cvarValue = solv.valueIteration(*pBamdp);

    float initAlpha = 0.5;
    CvarHist result = solv.executeBamdpEpisode(pBamdp, mdp1, initState, initAlpha);
    result.printPath();
}

TEST_CASE("Bamdp Cvar VI example 2"){
    int stages = 2;
    std::shared_ptr<MDP> templateMDP = makeBettingMDP(1.0, stages);
    std::shared_ptr<MDP> trueMDP = makeBettingMDP(0.6, stages);

    float priorWinCounts = 0.8;
    float priorLoseCounts = 0.2;

    std::shared_ptr<FullyTiedDirichletBelief> b = getBettingGameBelief(
            priorWinCounts,
            priorLoseCounts,
            stages,
            templateMDP);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State initState(stateMap);

    std::shared_ptr<MDP> pBamdp = b->toBamdp(templateMDP, initState, stages+1);

    int numInterpPts = 10;
    CvarValueIteration solv(numInterpPts);
    std::unordered_map<State, float, StateHash> cvarValue = solv.valueIteration(*pBamdp);

    float initAlpha = 0.4;
    CvarHist result = solv.executeBamdpEpisode(pBamdp, trueMDP, initState, initAlpha);
    result.printPath();
}
