#ifndef run_algorithms
#define run_algorithms
#include <string>
#include "state.h"
#include "domains.h"
#include "bamdp_rollout_policy.h"
#include "cvar_value_iteration.h"

void runPGTabular(
    std::string folder,
    domain dom,
    std::vector<float> alphas,
    int evalRepeats,
    std::vector<int> evalTrials,
    int batchSize,
    float lr
);

void runMCTSOffline(
    std::string folder,
    domain dom,
    std::vector<float> alphas,
    int evalRepeats,
    std::vector<int> evalTrials,
    int batchSize,
    float bias,
    float widening,
    std::string strat,
    std::string optim
);

void runMCTS(
    std::string folder,
    domain dom,
    std::vector<float> alphas,
    int evalRepeats,
    std::vector<std::tuple<int, int, int>> trialsList,
    float bias,
    float widening,
    std::string strat,
    std::string optim,
    std::string rolloutPolName = "none"
);

void runPGApprox(
    std::string folder,
    domain dom,
    std::vector<float> alphas,
    int evalRepeats,
    std::vector<int> evalTrials,
    int batchSize,
    float lr,
    bool initCvarPolicy
);

void runBamdpCvarVI(
    std::string folder,
    domain dom,
    std::vector<float> alphas,
    int evalRepeats,
    int numInterpPts
);

void runExpectedMDPCvarVI(
    std::string folder,
    domain dom,
    std::vector<float> alphas,
    int evalRepeats,
    int numInterpPts
);

std::shared_ptr<CvarValueIteration> runSSPCvarVI(
    std::string folder,
    std::shared_ptr<MDP> pMDP,
    State initState,
    std::vector<float> alphas,
    int evalRepeats,
    int numInterpPts,
    bool maximise,
    std::string name = "",
    bool isSSP = true
);

void runSSPCvarLexicographicVI(
    std::string folder,
    std::shared_ptr<MDP> pMDP,
    State initState,
    std::vector<float> alphas,
    int evalRepeats,
    int cvarInterpPts,
    int costInterpPts,
    bool maximise,
    std::string name = "",
    std::shared_ptr<CvarValueIteration> solver = NULL,
    bool isSSP = true
);

#endif
