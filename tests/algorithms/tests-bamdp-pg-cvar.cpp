/* This file implements tests for the MCTS implementation. */

#include <iostream>
#include <string>
#include "catch.h"
#include "mdp_examples.h"
#include "pg_bamdp_cvar.h"
#include "pg_bamdp_cvar_approx.h"
#include "fully_tied_dirichlet_belief.h"
#include "hist.h"

TEST_CASE("BAMDP_PG"){
    int stages = 4;
    std::shared_ptr<MDP> templateMDP = makeBettingMDP(1.0, stages);

    float priorWinCounts = 3.0;
    float priorLoseCounts = 1.0;

    std::shared_ptr<FullyTiedDirichletBelief> pBelief = getBettingGameBelief(
            priorWinCounts,
            priorLoseCounts,
            stages,
            templateMDP);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State initState(stateMap);

    float lr = 0.001;
    int horizon = stages+1;
    int batchSize = 1000;
    int iterations = 20;
    float alpha = 0.2;

    BamdpCvarPG solver(lr, pBelief, initState, alpha, horizon);
    solver.runTrials(batchSize, iterations);
    std::shared_ptr<MDP> pTrueModel = pBelief->sampleModel();
    Hist h = solver.executeEpisode(pTrueModel);
    h.printPath();

    solver.runTrials(batchSize, iterations);
    pTrueModel = pBelief->sampleModel();
    h = solver.executeEpisode(pTrueModel);
    h.printPath();
}

TEST_CASE("BAMDP_PG_APPROX"){
    int stages = 4;
    std::shared_ptr<MDP> templateMDP = makeBettingMDP(1.0, stages);

    float priorWinCounts = 3.0;
    float priorLoseCounts = 1.0;

    std::shared_ptr<FullyTiedDirichletBelief> pBelief = getBettingGameBelief(
            priorWinCounts,
            priorLoseCounts,
            stages,
            templateMDP);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State initState(stateMap);

    float lr = 0.1;

    int horizon = stages+1;
    int batchSize = 500;
    int iterations = 10;
    float alpha = 0.2;

    bool initPolicy = false;
    BamdpCvarPGApprox solver(lr, pBelief, initState, alpha, horizon, initPolicy);

    solver.runTrials(batchSize, iterations);
    std::shared_ptr<MDP> pTrueModel = pBelief->sampleModel();
    Hist h = solver.executeEpisode(pTrueModel);
    h.printPath();

    solver.runTrials(batchSize, iterations);
    h = solver.executeEpisode(pTrueModel);
    h.printPath();
}
