#include <unordered_map>
#include <iostream>
#include <random>
#include <fstream>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <cctype>
#include <string>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "domains.h"
#include "pg_bamdp_cvar.h"
#include "pg_bamdp_cvar_approx.h"
#include "mcts_cvar_sg_offline.h"
#include "mcts_bamdp_cvar_sg.h"
#include "bamdp_cvar_decision_node.h"
#include "cvar_value_iteration.h"
#include "agent_expected_mdp_policy.h"
#include "cvar_expected_mdp_policy.h"
#include "worst_case_value_iteration.h"


void runPGTabular(
    std::string folder,
    domain dom,
    std::vector<float> alphas,
    int evalRepeats,
    std::vector<int> evalTrials,
    int batchSize,
    float lr
){
    int horizon = std::get<0>(dom);
    std::shared_ptr<Belief> pBelief = std::get<1>(dom);
    std::shared_ptr<MDP> templateMDP = std::get<2>(dom);
    State initState = std::get<3>(dom);

    std::string runFileName = folder;
    std::time_t result = std::time(nullptr);
    std::string time = std::asctime(std::localtime(&result));
    time.erase(std::remove(time.begin(), time.end(), ' '), time.end());
    runFileName.append("/pg_discrete_").append(time).append("_lr_").append(std::to_string(lr)).append(".csv");
    std::ofstream myfile;
    myfile.open(runFileName);



    for(float alpha : alphas){
        int trialsDone = 0;
        BamdpCvarPG solver(lr, pBelief, initState, alpha, horizon);
        std::chrono::duration<float> trainingTime(0.0);

        for(int numTrials : evalTrials){

            // time training simulations
            auto begin = std::chrono::high_resolution_clock::now();
            int trialsToDo = numTrials - trialsDone;
            int batches = (int)std::round((float)trialsToDo/batchSize);
            solver.runTrials(batchSize, batches);
            trialsDone += batches * batchSize;
            auto end = std::chrono::high_resolution_clock::now();
            trainingTime += end - begin;

            // first two columns are alpha value and number of trials done
            myfile << alpha << ",";
            myfile << trialsDone << ",";
            std::vector<std::chrono::duration<float>> epTimes;
            for(int i = 0; i < evalRepeats; i++){
                std::shared_ptr<MDP> pTrueModel = pBelief->sampleModel();
                auto epStart = std::chrono::high_resolution_clock::now();
                Hist h = solver.executeEpisode(pTrueModel);
                auto epEnd = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float> epTime = epEnd - epStart;
                epTimes.push_back(epTime);
                h.printPath();
                myfile << h.getTotalReturn() << ",";
            }
            myfile << "\n";
            myfile << "time, " << trainingTime.count() << ",";
            for(auto ep : epTimes){
                myfile << ep.count() << ",";
            }
            myfile << "\n";
            std::cout << "Trials Complete: " << numTrials << std::endl;
        }
    }
    myfile.close();
}

void runPGApprox(
    std::string folder,
    domain dom,
    std::vector<float> alphas,
    int evalRepeats,
    std::vector<int> evalTrials,
    int batchSize,
    float lr,
    bool initCvarPolicy
){
    int horizon = std::get<0>(dom);
    std::shared_ptr<Belief> pBelief = std::get<1>(dom);
    std::shared_ptr<MDP> templateMDP = std::get<2>(dom);
    State initState = std::get<3>(dom);

    std::string runFileName = folder;
    std::time_t result = std::time(nullptr);
    std::string time = std::asctime(std::localtime(&result));
    time.erase(std::remove(time.begin(), time.end(), ' '), time.end());
    runFileName.append("/pg_approx_").append(time).append("_lr_").append(std::to_string(lr)).append(".csv");
    std::ofstream myfile;
    myfile.open(runFileName);



    for(float alpha : alphas){
        int trialsDone = 0;
        BamdpCvarPGApprox solver(lr, pBelief, initState, alpha, horizon, initCvarPolicy);
        std::chrono::duration<float> trainingTime(0.0);

        for(int numTrials : evalTrials){

            // time training simulations
            auto begin = std::chrono::high_resolution_clock::now();
            int trialsToDo = numTrials - trialsDone;
            int batches = (int)std::round((float)trialsToDo/batchSize);
            solver.runTrials(batchSize, batches);
            trialsDone += batches * batchSize;
            auto end = std::chrono::high_resolution_clock::now();
            trainingTime += end - begin;

            // first two columns are alpha value and number of trials done
            myfile << alpha << ",";
            myfile << trialsDone << ",";
            std::vector<std::chrono::duration<float>> epTimes;
            for(int i = 0; i < evalRepeats; i++){
                auto epStart = std::chrono::high_resolution_clock::now();
                Hist h = solver.executeEpisode();
                auto epEnd = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float> epTime = epEnd - epStart;
                epTimes.push_back(epTime);
                myfile << h.getTotalReturn() << ",";
            }
            myfile << "\n";
            myfile << "time, " << trainingTime.count() << ",";
            for(auto ep : epTimes){
                myfile << ep.count() << ",";
            }
            myfile << "\n";
            std::cout << "Trials Complete: " << numTrials << std::endl;
        }
    }
    myfile.close();
}

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
){
    int horizon = std::get<0>(dom);
    std::shared_ptr<Belief> pBelief = std::get<1>(dom);
    std::shared_ptr<MDP> templateMDP = std::get<2>(dom);
    State initState = std::get<3>(dom);

    std::string runFileName = folder;
    std::time_t result = std::time(nullptr);
    std::string time = std::asctime(std::localtime(&result));
    time.erase(std::remove(time.begin(), time.end(), ' '), time.end());
    runFileName.append("/mcts_offline_").append(time).append("_bias_").append(std::to_string(bias));
    runFileName.append("_widening_").append(std::to_string(widening));
    runFileName.append("_optim_").append(optim);
    runFileName.append(".csv");

    std::ofstream myfile;
    myfile.open(runFileName);
    bool verbose = true;

    for(float alpha : alphas){
        int trialsDone = 0;

        std::shared_ptr<BamdpCvarDecisionNode> rootNode;
        rootNode = std::make_shared<BamdpCvarDecisionNode>(
            templateMDP,
            pBelief,
            initState,
            alpha
        );

        CvarMCTSOffline solver(
            templateMDP,
            rootNode,
            horizon,
            optim,
            bias,
            widening,
            strat
        );

        for(int numTrials : evalTrials){
            int trialsToDo = numTrials - trialsDone;
            int batches = (int)std::round((float)trialsToDo/batchSize);
            solver.runTrials(batchSize, batches);
            trialsDone += batches * batchSize;

            // first two columns are alpha value and number of trials done
            myfile << alpha << ",";
            myfile << trialsDone << ",";
            for(int i = 0; i < evalRepeats; i++){
                std::shared_ptr<MDP> pTrueModel = pBelief->sampleModel();
                CvarGameHist h = solver.executeEpisode(pTrueModel);
                h.printPath(verbose);
                myfile << h.getTotalReturn() << ",";
            }
            myfile << "\n";
        }
    }
    myfile.close();
}


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
    std::string rolloutPolName
){
    int horizon = std::get<0>(dom);
    std::shared_ptr<Belief> pBelief = std::get<1>(dom);
    std::shared_ptr<MDP> templateMDP = std::get<2>(dom);
    State initState = std::get<3>(dom);

    std::string runFileName = folder;
    std::time_t result = std::time(nullptr);
    std::string time = std::asctime(std::localtime(&result));
    time.erase(std::remove(time.begin(), time.end(), ' '), time.end());
    runFileName.append("/mcts_online_").append(time).append("_rollout_").append(rolloutPolName).append("_bias_").append(std::to_string(bias));
    runFileName.append("_widening_").append(std::to_string(widening));
    runFileName.append("_expansionstrat_").append(strat);
    runFileName.append("_optim_").append(optim);
    runFileName.append(".csv");

    // time preprocessing
    auto begin = std::chrono::high_resolution_clock::now();

    std::shared_ptr<BamdpRolloutPolicy> rolloutPol;
    if(rolloutPolName == "expected_value"){
        rolloutPol = std::make_shared<AgentExpectedMDPPolicy>(pBelief);
    }else if(rolloutPolName == "cvar"){
        rolloutPol = std::make_shared<CvarExpectedMDPPolicy>(pBelief);
    }else if(rolloutPolName == "none"){
        rolloutPol = NULL;
    }else{
        throw "Invalid rollout policy type";
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> preprocessTime = end - begin;

    std::ofstream myfile;
    myfile.open(runFileName);
    myfile.close();
    bool verbose = true;

    for(float alpha : alphas){

        for(auto tup : trialsList){
            int burnInTrials = std::get<0>(tup);
            int agentTrials = std::get<1>(tup);
            int advTrials = std::get<2>(tup);

            myfile.open(runFileName, std::ios_base::app);
            myfile << alpha << ",";
            myfile << "burn_" << burnInTrials << "agent_" << agentTrials << "adv_" << advTrials << ",";
            myfile.close();

            std::vector<std::chrono::duration<float>> epTimes;
            for(int j = 0; j < evalRepeats;  j++){
                std::cout << "Episode: " << j << std::endl;
                BamdpCvarMCTS solver(bias, widening, strat);
                std::shared_ptr<BamdpCvarDecisionNode> rootNode;
                rootNode = std::make_shared<BamdpCvarDecisionNode>(
                    templateMDP,
                    pBelief,
                    initState,
                    alpha
                );

                std::shared_ptr<MDP> pTrueModel = pBelief->sampleModel();

                auto epStart = std::chrono::high_resolution_clock::now();
                CvarGameHist history = solver.executeEpisode(
                                            pTrueModel,
                                            rootNode,
                                            horizon,
                                            burnInTrials,
                                            agentTrials,
                                            advTrials,
                                            optim,
                                            rolloutPol
                                        );

                auto epEnd = std::chrono::high_resolution_clock::now();
                history.printPath(verbose);
                myfile.open(runFileName, std::ios_base::app);
                myfile << history.getTotalReturn() << ",";
                myfile.close();
                std::chrono::duration<float> epTime = epEnd - epStart;
                epTimes.push_back(epTime);
            }
            myfile.open(runFileName, std::ios_base::app);
            myfile << "\n";
            myfile << "time, " << preprocessTime.count() << ",";
            for(auto ep : epTimes){
                myfile << ep.count() << ",";
            }
            myfile << "\n";
            myfile.close();
        }
    }
    myfile.close();
}

void runBamdpCvarVI(
    std::string folder,
    domain dom,
    std::vector<float> alphas,
    int evalRepeats,
    int numInterpPts
){
    int horizon = std::get<0>(dom);
    std::shared_ptr<Belief> pBelief = std::get<1>(dom);
    std::shared_ptr<MDP> templateMDP = std::get<2>(dom);
    State initState = std::get<3>(dom);

    std::string runFileName = folder;
    std::time_t result = std::time(nullptr);
    std::string time = std::asctime(std::localtime(&result));
    time.erase(std::remove(time.begin(), time.end(), ' '), time.end());
    runFileName.append("/cvar_vi_bamdp_").append(time);
    runFileName.append(".csv");

    std::ofstream myfile;
    myfile.open(runFileName);

    // time preprocessing
    auto begin = std::chrono::high_resolution_clock::now();
    std::shared_ptr<MDP> pBamdp = pBelief->toBamdp(templateMDP, initState, horizon);

    CvarValueIteration solver(numInterpPts);
    solver.valueIteration(*pBamdp);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> preprocessTime = end - begin;

    for(auto alpha : alphas){
        myfile << std::to_string(alpha) << ",";
        myfile << "0" << ",";

        std::vector<std::chrono::duration<float>> epTimes;
        for(int j = 0; j < evalRepeats;  j++){
            std::cout << "Episode: " << j << std::endl;
            std::shared_ptr<MDP> pTrueModel = pBelief->sampleModel();
            auto epStart = std::chrono::high_resolution_clock::now();
            CvarHist history = solver.executeBamdpEpisode(pBamdp, pTrueModel, initState, alpha);
            auto epEnd = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> epTime = epEnd - epStart;
            epTimes.push_back(epTime);
            history.printPath(true);
            myfile << history.getTotalReturn() << ",";
        }
        myfile << "\n";
        myfile << "time, " << preprocessTime.count() << ",";
        for(auto ep : epTimes){
            myfile << ep.count() << ",";
        }
        myfile << "\n";
    }
    myfile.close();
}

void runExpectedMDPCvarVI(
    std::string folder,
    domain dom,
    std::vector<float> alphas,
    int evalRepeats,
    int numInterpPts
){
    std::shared_ptr<Belief> pBelief = std::get<1>(dom);
    std::shared_ptr<MDP> templateMDP = std::get<2>(dom);
    State initState = std::get<3>(dom);

    std::string runFileName = folder;
    std::time_t result = std::time(nullptr);
    std::string time = std::asctime(std::localtime(&result));
    time.erase(std::remove(time.begin(), time.end(), ' '), time.end());
    runFileName.append("/cvar_vi_expected_mdp").append(time);
    runFileName.append(".csv");

    std::ofstream myfile;
    myfile.open(runFileName);

    // time preprocessing
    auto begin = std::chrono::high_resolution_clock::now();
    std::shared_ptr<MDP> pExpectedMDP = pBelief->getExpectedMDP();

    CvarValueIteration solver(numInterpPts);
    solver.valueIteration(*pExpectedMDP);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> preprocessTime = end - begin;

    for(auto alpha : alphas){
        myfile << std::to_string(alpha) << ",";
        myfile << "0" << ",";

        std::vector<std::chrono::duration<float>> epTimes;
        for(int j = 0; j < evalRepeats;  j++){
            std::cout << "Episode: " << j << std::endl;
            std::shared_ptr<MDP> pTrueModel = pBelief->sampleModel();
            auto epStart = std::chrono::high_resolution_clock::now();
            CvarHist history = solver.executeEpisode(pExpectedMDP, pTrueModel, initState, alpha);
            auto epEnd = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> epTime = epEnd - epStart;
            epTimes.push_back(epTime);
            history.printPath();
            myfile << history.getTotalReturn() << ",";
        }
        myfile << "\n";
        myfile << "time, " << preprocessTime.count() << ",";
        for(auto ep : epTimes){
            myfile << ep.count() << ",";
        }
        myfile << "\n";
    }
    myfile.close();
}

std::shared_ptr<CvarValueIteration> runSSPCvarVI(
    std::string folder,
    std::shared_ptr<MDP> pMDP,
    State initState,
    std::vector<float> alphas,
    int evalRepeats,
    int numInterpPts,
    bool maximise,
    std::string name,
    bool isSSP
){
    std::string runFileName = folder;
    std::time_t result = std::time(nullptr);
    std::string time = std::asctime(std::localtime(&result));
    time.erase(std::remove(time.begin(), time.end(), ' '), time.end());
    time.erase(std::remove(time.begin(), time.end(), '\n'), time.end());
    runFileName.append("/cvar_vi-worst_case").append("_").append(name).append("_").append(time);
    std::string pathFileName = runFileName;
    runFileName.append(".csv");
    pathFileName.append("_paths.txt");

    std::ofstream myfile;
    myfile.open(runFileName);

    std::ofstream pathfile;
    int pathsToSave = 50;
    pathfile.open(pathFileName);
    pathfile.close();

    // time preprocessing
    auto begin = std::chrono::high_resolution_clock::now();

    std::cout << "Computing worst-case value function..." << std::endl;
    std::unordered_map<State, float, StateHash> worstCaseValue;
    std::unordered_map<State, std::string, StateHash> worstCasePolicy;
    worstCaseVI vi;
    std::tie(worstCaseValue, worstCasePolicy) = vi.valueIteration(*pMDP, maximise);

    std::shared_ptr<CvarValueIteration> solver = std::make_shared<CvarValueIteration>(numInterpPts);
    std::unordered_map<State, float, StateHash> cvarValue;

    if(isSSP){
        cvarValue = solver->sspValueIteration(*pMDP, maximise);
    }else{
        cvarValue = solver->valueIteration(*pMDP, maximise);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> preprocessTime = end - begin;

    for(auto alpha : alphas){
        myfile << std::to_string(alpha) << ",";
        myfile << "0" << ",";

        std::vector<std::chrono::duration<float>> epTimes;
        for(int j = 0; j < evalRepeats;  j++){
            std::cout << "Episode: " << j << std::endl;
            auto epStart = std::chrono::high_resolution_clock::now();
            CvarHist history = solver->sspExecuteEpisode(pMDP, initState, alpha, worstCasePolicy, maximise);
            auto epEnd = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> epTime = epEnd - epStart;

            epTimes.push_back(epTime);
            history.printPath();
            if(j < pathsToSave){
                history.printPathToFile(pathFileName);
            }
            myfile << history.getTotalReturn() << ",";
        }
        myfile << "\n";
        myfile << "time, " << preprocessTime.count() << ",";
        for(auto ep : epTimes){
            myfile << ep.count() << ",";
        }
        myfile << "\n";
    }
    myfile.close();
    return solver;
}

void runSSPCvarLexicographicVI(
    std::string folder,
    std::shared_ptr<MDP> pMDP,
    State initState,
    std::vector<float> alphas,
    int evalRepeats,
    int cvarInterpPts,
    int costInterpPts,
    bool maximise,
    std::string name,
    std::shared_ptr<CvarValueIteration> cvarSolver,
    bool isSSP
){
    std::string runFileName = folder;
    std::time_t result = std::time(nullptr);
    std::string time = std::asctime(std::localtime(&result));
    time.erase(std::remove(time.begin(), time.end(), ' '), time.end());
    time.erase(std::remove(time.begin(), time.end(), '\n'), time.end());
    runFileName.append("/cvar_vi-expected_value").append("_").append(name).append("_").append(time);
    std::string pathFileName = runFileName;
    pathFileName.append("_paths.txt");
    runFileName.append(".csv");

    std::ofstream myfile;
    myfile.open(runFileName);

    std::ofstream pathfile;
    int pathsToSave = 50;
    pathfile.open(pathFileName);
    pathfile.close();

    // time preprocessing
    auto begin = std::chrono::high_resolution_clock::now();

    std::cout << "Computing worst-case value function..." << std::endl;
    std::unordered_map<State, float, StateHash> worstCaseValue;
    std::unordered_map<State, std::string, StateHash> worstCasePolicy;
    worstCaseVI vi;
    std::tie(worstCaseValue, worstCasePolicy) = vi.valueIteration(*pMDP, maximise);

    std::cout << "Doing CVaR VI to compute agent and adv policy..." << std::endl;
    std::unordered_map<State, float, StateHash> cvarValue;
    if(cvarSolver == NULL){
        cvarSolver = std::make_shared<CvarValueIteration>(cvarInterpPts);
        cvarSolver->sspValueIteration(*pMDP, maximise);
    }else{
        std::cout << "Using CVaR VI passed as argument." << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> preprocessTime = end - begin;
    std::unordered_map<float, float> varEstimates;

    for(auto alpha : alphas){
        myfile << std::to_string(alpha) << ",";

        auto beginAlpha = std::chrono::high_resolution_clock::now();

        // compute empirical var estimate by simulating trajectories through
        // induced MC
        std::cout << "Computing empirical VaR estimate..." << std::endl;
        int samples = 2000;
        std::vector<float> returns;
        for(int i = 0; i < samples; i++){
            CvarHist history = cvarSolver->sspExecuteEpisode(
                pMDP,
                initState,
                alpha,
                worstCasePolicy,
                maximise
            );
            returns.push_back(history.getTotalReturn());
        }
        std::sort(returns.begin(), returns.end());
        int varIndex = (int)std::round(samples * (1.0-alpha));
        float varAtAlpha = returns[varIndex];

        std::cout << "Getting the lexicographic policy..." << std::endl;
        varEstimates[alpha] = varAtAlpha;
        CvarLexicographic lexSolver(costInterpPts, varAtAlpha, worstCaseValue, worstCasePolicy);
        lexSolver.computeLexicographicValue(pMDP, isSSP);

        auto endAlpha = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float> alphaTime = endAlpha - beginAlpha;
        std::chrono::duration<float> totalTime = preprocessTime + alphaTime;
        myfile << totalTime.count() << ",";

        std::vector<std::chrono::duration<float>> epTimes;
        for(int j = 0; j < evalRepeats;  j++){
            std::cout << "Episode: " << j << ", Var Estimate: " << varAtAlpha << std::endl;
            CvarHist history = cvarSolver->sspExecuteEpisodeLexicographic(
                pMDP,
                initState,
                alpha,
                worstCasePolicy,
                lexSolver,
                maximise
            );
            history.printPath();
            if(j < pathsToSave){
                history.printPathToFile(pathFileName);
            }
            myfile << history.getTotalReturn() << ",";
        }
        myfile << "\n";
    }
    myfile.close();

    for(auto pair : varEstimates){
        std::cout << "Alpha: " << pair.first << ", Var estimate: " << pair.second << std::endl;
    }
}
