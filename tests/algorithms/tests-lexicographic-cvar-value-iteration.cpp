/* This file implements tests for value iteration */
#include <unordered_map>
#include <iostream>
#include "catch.h"
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "cvar_value_iteration.h"
#include "cvar_lexicographic.h"
#include "value_iteration.h"
#include "worst_case_value_iteration.h"
#include "ssp_mdp_examples.h"
#include "cvar_hist.h"
#include "finite_mdp_belief.h"

TEST_CASE("Cvar_lexicographic"){
    std::shared_ptr<MDP> pMDP = testExample();
    bool maximise = false; // cost minimisation

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["state"] = "0";
    State state0(stateMap);
    stateMap["state"] = "1";
    State state1(stateMap);
    stateMap["state"] = "2";
    State state2(stateMap);
    stateMap["state"] = "3";
    State state3(stateMap);
    stateMap["state"] = "4";
    State state4(stateMap);
    stateMap["state"] = "5";
    State state5(stateMap);
    stateMap["state"] = "6";
    State state6(stateMap);
    stateMap["state"] = "7";
    State state7(stateMap);
    stateMap["state"] = "8";
    State state8(stateMap);

    std::unordered_map<State, float, StateHash> worstCaseValue;
    std::unordered_map<State, std::string, StateHash> worstCasePolicy;

    worstCaseVI vi;
    std::tie(worstCaseValue, worstCasePolicy) = vi.valueIteration(*pMDP, maximise);
    REQUIRE(cmpf(worstCaseValue[state0], 10.0));
    REQUIRE(cmpf(worstCaseValue[state3], 7.0));
    REQUIRE(worstCasePolicy[state3] == "C");

    int numInterpPts = 20;
    CvarValueIteration solver(numInterpPts);
    std::unordered_map<State, float, StateHash> cvarValue = solver.sspValueIteration(*pMDP, maximise);

    stateMap["state"] = "0";
    stateMap["alpha"] = "0.083298";
    REQUIRE(cmpf(cvarValue[State(stateMap)]/0.083298, 10.0, 1e-2));

    // int numVarInterpPts = 50;
    // std::unordered_map<State, float, StateHash> varValue = solver.sspGetVaR(state0, pMDP, maximise, numVarInterpPts);
    // float alpha = 0.09;
    // float varAtAlpha = solver.interpVarAlphaValue(varValue, state0, alpha);
    // REQUIRE(cmpf(varAtAlpha, 10.0));
    //
    // alpha = 0.11;
    // varAtAlpha = solver.interpVarAlphaValue(varValue, state0, alpha);
    // REQUIRE(cmpf(varAtAlpha, 8.0));

    float varAtAlpha = 10.0;
    int costInterpPts = 50;
    bool isSSP = true;
    CvarLexicographic lexSolver(costInterpPts, varAtAlpha, worstCaseValue, worstCasePolicy);
    lexSolver.computeLexicographicValue(pMDP, isSSP);

    std::string action = lexSolver.getOptimalAction(
                                        pMDP,
                                        state3,
                                        0.0
    );
    REQUIRE(action == "B");

    action = lexSolver.getOptimalAction(
                                        pMDP,
                                        state3,
                                        2.0
    );

    REQUIRE(action == "C");

    varAtAlpha = 8.0;
    CvarLexicographic lexSolver2(costInterpPts, varAtAlpha, worstCaseValue, worstCasePolicy);
    lexSolver2.computeLexicographicValue(pMDP, isSSP);

    action = lexSolver2.getOptimalAction(
                                        pMDP,
                                        state3,
                                        0.0
    );

    REQUIRE(action == "C");
}
