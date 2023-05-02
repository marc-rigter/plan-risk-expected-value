/* This file implements tests for the MCTS implementation. */

#include <iostream>
#include <string>
#include "catch.h"
#include "mdp_examples.h"
#include "bamdp_cvar_decision_node.h"
#include "bamdp_cvar_adv_decision_node.h"
#include "mcts_bamdp_cvar_sg.h"
#include "finite_mdp_belief.h"
#include "belief.h"
#include "utils.h"
#include "mcts_cvar_sg_offline.h"
#include "cvar_hist.h"

TEST_CASE("BAMDP CVAR check root belief constant"){
    std::shared_ptr<MDP> testMDP1 = makeBettingMDP(0.5, 4);
    std::shared_ptr<MDP> testMDP2 = makeBettingMDP(0.8, 4);
    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights;
    mdpWeights[testMDP1] = 0.4;
    mdpWeights[testMDP2] = 0.6;
    std::shared_ptr<FiniteMDPBelief> pBelief = std::make_shared<FiniteMDPBelief>(mdpWeights);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State state(stateMap);

    BamdpCvarMCTS m(0.5, 0.3);
    std::shared_ptr<BamdpCvarDecisionNode> rootNode;
    rootNode = std::make_shared<BamdpCvarDecisionNode>(testMDP1, pBelief, state, 0.1);
    int horizon = 5;
    int numTrials = 10;

    m.cvarMCTS(testMDP1, rootNode, horizon, numTrials);
    std::shared_ptr<Belief> beliefAfterRollouts = rootNode->getBelief();
    mdpWeights = dynamic_cast<FiniteMDPBelief*>(beliefAfterRollouts.get())->getWeights();

    REQUIRE(cmpf(mdpWeights[testMDP1], 0.4));
    REQUIRE(cmpf(mdpWeights[testMDP2], 0.6));
}

TEST_CASE("Check belief updates and transition probs"){
    std::shared_ptr<MDP> testMDP1 = makeBettingMDP(0.5, 4);
    std::shared_ptr<MDP> testMDP2 = makeBettingMDP(0.8, 4);
    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights;
    mdpWeights[testMDP1] = 0.4;
    mdpWeights[testMDP2] = 0.6;
    std::shared_ptr<FiniteMDPBelief> pBelief = std::make_shared<FiniteMDPBelief>(mdpWeights);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State state(stateMap);

    stateMap["t"] = "1";
    stateMap["money"] = "12";
    State stateWin(stateMap);

    stateMap["t"] = "1";
    stateMap["money"] = "8";
    State stateLose(stateMap);

    std::string strat = "bayesOpt";
    BamdpCvarMCTS m(0.5, 0.0, strat);
    std::shared_ptr<BamdpCvarDecisionNode> rootNode;
    rootNode = std::make_shared<BamdpCvarDecisionNode>(testMDP1, pBelief, state, 0.1);
    int horizon = 5;
    int numTrials = 100;

    m.cvarMCTS(testMDP1, rootNode, horizon, numTrials);

    // get children after action of 2 played in root
    std::shared_ptr<CvarAdvDecisionNode> advNode = rootNode->getAdvChildren()["2"];

    // check the transition probabilities according to belief at the root
    std::unordered_map<State, float, StateHash> successorProbs;
    successorProbs = advNode->getSuccessorProbs();
    REQUIRE(cmpf(successorProbs[stateWin], 0.68));
    REQUIRE(cmpf(successorProbs[stateLose], 0.32));

    // get chance node for first adversary action played
    std::shared_ptr<CvarChanceNode> chanceNode = advNode->getChanceChildren()[0];

    // get decision nodes after either of the random outcomes
    std::unordered_map<State, std::shared_ptr<CvarDecisionNode>, StateHash> decisionNodeMap = chanceNode->getCvarChildren();

    for(auto pair : decisionNodeMap){
        std::shared_ptr<Belief> beliefUpdate = dynamic_cast<BamdpCvarDecisionNode*>(pair.second.get())->getBelief();
        mdpWeights = dynamic_cast<FiniteMDPBelief*>(beliefUpdate.get())->getWeights();
        std::unordered_map<std::string, std::shared_ptr<CvarAdvDecisionNode>> advs = pair.second->getAdvChildren();

        if(pair.first == stateWin){

            // check updated belief
            REQUIRE(cmpf(mdpWeights[testMDP1], 0.29411));
            REQUIRE(cmpf(mdpWeights[testMDP2], 0.70588));

            // check transition probability distribution for belief
            for(auto pairInner : advs){
                if(pairInner.first == "0"){
                    continue;
                }
                stateMap["t"] = "2";
                stateMap["money"] = std::to_string(12 + std::stoi(pairInner.first));
                State stateWinWin(stateMap);

                stateMap["t"] = "2";
                stateMap["money"] = std::to_string(12 - std::stoi(pairInner.first));
                State stateWinLose(stateMap);

                std::unordered_map<State, float, StateHash> successorProbs;
                successorProbs = pairInner.second->getSuccessorProbs();
                REQUIRE(cmpf(successorProbs[stateWinWin], 0.71176));
                REQUIRE(cmpf(successorProbs[stateWinLose], 0.28824));
            }

        // if the first stage lost check correct belief probabilities for second
        // stage
        }else if(pair.first == stateLose){

            // check updated belief
            REQUIRE(cmpf(mdpWeights[testMDP1], 0.625));
            REQUIRE(cmpf(mdpWeights[testMDP2], 0.375));

            // check transition probability distribution for belief
            for(auto pairInner : advs){
                if(pairInner.first == "0"){
                    continue;
                }
                stateMap["t"] = "2";
                stateMap["money"] = std::to_string(8 + std::stoi(pairInner.first));
                State stateLoseWin(stateMap);

                stateMap["t"] = "2";
                stateMap["money"] = std::to_string(8 - std::stoi(pairInner.first));
                State stateLoseLose(stateMap);
                std::unordered_map<State, float, StateHash> successorProbs;
                successorProbs = pairInner.second->getSuccessorProbs();
                REQUIRE(cmpf(successorProbs[stateLoseWin], 0.6125));
                REQUIRE(cmpf(successorProbs[stateLoseLose], 0.3875));
            }
        }
    }
}

TEST_CASE("BAMDP_CVAR"){

    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights;
    std::shared_ptr<MDP> pMDP;
    std::shared_ptr<MDP> pMDP2;
    std::shared_ptr<MDP> pTrueModel;

    float prob = 0.2;
    pMDP2 = makeBettingMDP(prob, 3);
    mdpWeights[pMDP2] = 0.1;

    prob = 0.6;
    pMDP = makeBettingMDP(prob, 3);
    mdpWeights[pMDP] = 0.1;

    prob = 0.8;
    pMDP = makeBettingMDP(prob, 3);
    mdpWeights[pMDP] = 0.4;

    prob = 1.0;
    pMDP = makeBettingMDP(prob, 3);
    mdpWeights[pMDP] = 0.4;

    std::shared_ptr<FiniteMDPBelief> pBelief = std::make_shared<FiniteMDPBelief>(mdpWeights);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State state(stateMap);

    std::string strat = "bayesOpt";
    BamdpCvarMCTS m(0.1, 0.2, strat);
    std::shared_ptr<BamdpCvarDecisionNode> rootNode;
    rootNode = std::make_shared<BamdpCvarDecisionNode>(pMDP, pBelief, state, 0.3);
    int horizon = 4;
    int burninTrials = 1000;
    int agentTrials = 300;
    int advTrials = 300;

    std::shared_ptr<MDP> realMDP =  pMDP2;
    CvarGameHist history = m.executeEpisode(realMDP, rootNode, horizon, burninTrials, agentTrials, advTrials);
    bool verbose = false;
    history.printPath(verbose);

}

TEST_CASE("BAMDP_CVAR_MED"){

    std::unordered_map<std::shared_ptr<MDP>, float> mdpWeights;
    std::shared_ptr<MDP> pMDP;

    int nSamples = 10;
    int days = 7;
    for(int n = 0; n < nSamples; n++){
        int seed = n;
        pMDP = makeMedicalMDP(days, seed);
        mdpWeights[pMDP] = 1.0/nSamples;
    }

    std::shared_ptr<FiniteMDPBelief> pBelief = std::make_shared<FiniteMDPBelief>(mdpWeights);

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["health"] = "5";
    State state(stateMap);

    std::string strat = "bayesOpt";
    BamdpCvarMCTS m(0.1, 0.2, strat);
    std::shared_ptr<BamdpCvarDecisionNode> rootNode;
    rootNode = std::make_shared<BamdpCvarDecisionNode>(pMDP, pBelief, state, 0.3);
    int horizon = days + 1;
    int burninTrials = 100;
    int agentTrials = 30;
    int advTrials = 40;

    std::shared_ptr<MDP> realMDP = pBelief->sampleModel();
    CvarGameHist history = m.executeEpisode(realMDP, rootNode, horizon, burninTrials, agentTrials, advTrials);
    bool verbose = false;
    history.printPath(verbose);
}

TEST_CASE("MCTS_CVAR_offline"){
    int stages = 2;
    std::shared_ptr<MDP> pTrueModel;
    std::shared_ptr<MDP> pTemplateModel;
    pTemplateModel = makeBettingMDP(1.0, stages);
    pTrueModel = makeBettingMDP(0.2, stages);

    // initial belief
    float priorWinCount = 0.8;
    float priorLoseCount = 0.2;
    std::shared_ptr<FullyTiedDirichletBelief> pBelief = getBettingGameBelief(
            priorWinCount,
            priorLoseCount,
            stages,
            pTemplateModel);

    TiedDirichletDistribution priorDist = pBelief->getDirichletDistribution();

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = "0";
    stateMap["money"] = "10";
    State initState(stateMap);

    std::string strat = "bayesOpt";
    std::shared_ptr<BamdpCvarDecisionNode> rootNode;
    float alpha = 0.5;
    rootNode = std::make_shared<BamdpCvarDecisionNode>(pTemplateModel, pBelief, initState, alpha);
    int horizon = 4;


    std::string lexicographicOptimisation = "worst_case";

    float bias = 2.0;
    float widening = 0.25;

    CvarMCTSOffline m(
        pTemplateModel,
        rootNode,
        horizon,
        lexicographicOptimisation,
        bias,
        widening,
        strat
    );

    int batchSize = 500;
    int iterations = 10;
    m.runTrials(batchSize, iterations);
    CvarGameHist history = m.executeEpisode(pTrueModel);
    bool verbose = true;
    history.printPath(verbose);

    m.runTrials(batchSize, iterations);
    history = m.executeEpisode(pTrueModel);
    history.printPath(verbose);
}
