#include <unordered_map>
#include <iostream>
#include <random>
#include "catch.h"
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "domains.h"
#include "mdp_examples.h"
#include "finite_mdp_belief.h"
#include "ssp_mdp_examples.h"

domain bettingGame(int horizon){
    int stages = horizon - 1;
    std::vector<int> actions{0, 1, 2, 5, 10};
    std::shared_ptr<MDP> templateMDP = makeBettingMDP(1.0, stages, actions);
    float priorWinCounts = 10.0/11;
    float priorLoseCounts = 1.0/11;

    std::shared_ptr<FullyTiedDirichletBelief> pBelief = getBettingGameBelief(
            priorWinCounts,
            priorLoseCounts,
            stages,
            templateMDP);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State initState(stateMap);

    return std::make_tuple(horizon, pBelief, templateMDP, initState);
}

domain bettingGameSmall(){
    int horizon = 7;
    return bettingGame(horizon);
}

domain bettingGameLarge(){
    int horizon = 11;
    return bettingGame(horizon);
}

domain medicalDomain(int horizon){
    int nSamples = 15;
    int days = horizon;

    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights;
    std::shared_ptr<MDP> pTrueModel;
    std::shared_ptr<MDP> templateMDP = makeMedicalMDP(days, 0);

    for(int n = 0; n < nSamples; n++){
        int seed = n;
        std::shared_ptr<MDP> pMDP = makeMedicalMDP(days, seed);
        mdpWeights[pMDP] = 1.0/nSamples;
    }

    std::shared_ptr<FiniteMDPBelief> pBelief = std::make_shared<FiniteMDPBelief>(mdpWeights);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["health"] = "10";
    State initState(stateMap);

    return std::make_tuple(horizon, pBelief, templateMDP, initState);
}

domain medicalDomainSmall(){
    int horizon = 5;
    return medicalDomain(horizon);
}

domain medicalDomainLarge(){
    int horizon = 10;
    return medicalDomain(horizon);
}

domain marsRover(int horizon){
    std::shared_ptr<MDP> templateMDP = marsRoverMDP(horizon);
    bool fullyTied = true;
    float initCount = 0.5;
    std::shared_ptr<Belief> pBelief = getMarsRoverBelief(templateMDP, horizon, initCount, fullyTied);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["x"] = "3";
    stateMap["y"] = "9";
    State initState(stateMap);

    return std::make_tuple(horizon, pBelief, templateMDP, initState);
}

domain trafficDomain(){
    int horizon = 10;
    std::shared_ptr<MDP> templateMDP = trafficMDP(horizon);
    std::vector<float> initCounts{1.0, 1.0, 0.4};
    std::shared_ptr<Belief> pBelief = getTrafficBelief(templateMDP, horizon, initCounts);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["min"] = "0";
    stateMap["x"] = "0";
    stateMap["y"] = "0";
    State initState(stateMap);

    return std::make_tuple(horizon, pBelief, templateMDP, initState);
}

domain trafficDomainRefactored(){
    int horizon = 10;
    bool refactored = true;
    std::shared_ptr<MDP> templateMDP = trafficMDPRefactored(horizon);
    std::vector<float> initCounts{1.0, 1.0, 0.4};
    std::shared_ptr<Belief> pBelief = getTrafficBelief(templateMDP, horizon, initCounts, refactored);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["min"] = "0";
    stateMap["x"] = "0";
    stateMap["y"] = "0";
    State initState(stateMap);

    // note that the horizon for this one has to be doubled for the search algorithms
    // as for each step in "real" time there is another step used to accumulate the cost
    horizon *= 2;
    return std::make_tuple(horizon, pBelief, templateMDP, initState);
}

domain marsRoverSmall(){
    int horizon = 5;
    return marsRover(horizon);
}

domain marsRoverLarge(){
    int horizon = 12;
    return marsRover(horizon);
}

std::tuple<State, std::shared_ptr<MDP>> sspTestDomain(){
    std::shared_ptr<MDP> pMDP = testExample();
    std::unordered_map<std::string, std::string> stateMap;
    stateMap["state"] = "0";
    State initState(stateMap);

    return std::make_tuple(initState, pMDP);
}

std::tuple<State, std::shared_ptr<MDP>> sspSyntheticTrafficDomain(int size){
    std::shared_ptr<MDP> pMDP = trafficSSP(size);
    std::unordered_map<std::string, std::string> stateMap;
    stateMap["x"] = "0";
    stateMap["y"] = "0";
    stateMap["cost"] = "0.0";
    State initState(stateMap);

    return std::make_tuple(initState, pMDP);
}

std::tuple<State, std::shared_ptr<MDP>> sspRealTrafficDomain(
    std::string adjacencyMatrixFile,
    int startID,
    int goalID,
    bool includeHighways,
    bool addNoise
){
    std::shared_ptr<MDP> pMDP = realTrafficSSP(
        adjacencyMatrixFile,
        startID,
        goalID,
        includeHighways,
        addNoise
    );
    std::unordered_map<std::string, std::string> stateMap;
    stateMap["id"] = std::to_string(startID);
    stateMap["cost"] = "0.0";
    State initState(stateMap);

    return std::make_tuple(initState, pMDP);
}

std::tuple<State, std::shared_ptr<MDP>> bettingWithJackpotDomain(
){
    int stages = 10;
    std::shared_ptr<MDP> pMDP = makeBettingWithJackpotMDP(stages);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "5";
    State initState(stateMap);

    return std::make_tuple(initState, pMDP);
}

std::tuple<State, std::shared_ptr<MDP>> deepSeaTreasureDomain(
){
    int horizon = 16;
    std::shared_ptr<MDP> pMDP = deepSeaTreasureMDP(horizon);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["x"] = "0";
    stateMap["y"] = "1";
    State initState(stateMap);

    return std::make_tuple(initState, pMDP);
}

std::tuple<State, std::shared_ptr<MDP>> inventoryDomain(){
    int horizon = 10;
    int maxInventory = 20;
    std::shared_ptr<MDP> pMDP = makeInventoryRandomWalkMDP(horizon, maxInventory);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["inventory"] = "0";
    stateMap["demand"] = "10";
    State initState(stateMap);

    return std::make_tuple(initState, pMDP);
}
