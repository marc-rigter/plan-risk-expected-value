/* This file implements tests for solving BAMDPs with MCTS */

#include <iostream>
#include <string>
#include "catch.h"
#include "multimodel_mdp.h"
#include "mdp_examples.h"
#include "posterior_sampling_solver.h"


TEST_CASE("Posterior sampling"){
    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights;
    std::unordered_map<std::shared_ptr<MDP>, std::pair<int, int>> mdpGoals;
    std::shared_ptr<MDP> pTrueModel;
    int horizon = 15;

    std::shared_ptr<MDP> pMDP = makeSimpleMDP(2, 0, 0.0);
    pTrueModel = pMDP;
    mdpWeights[pMDP] = 0.5;

    pMDP = makeSimpleMDP(1, 0, 0.0);
    mdpWeights[pMDP] = 0.5;

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = "0";
    stateMap["y"] = "0";
    State initState(stateMap);
    MultiModelMDP mmdp = MultiModelMDP(mdpWeights);

    int stepsBeforeResample = 1;
    PosteriorSamplingSolver solver(stepsBeforeResample);
    REQUIRE(solver.getNextAction(mmdp, initState, horizon) == "right");
}
