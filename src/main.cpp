#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <boost/cstdfloat.hpp>
#include <string.h>
#include <sys/stat.h>
#include "mdp.h"
#include "catch.h"
#include "multimodel_mdp.h"
#include "mdp_examples.h"
#include "ssp_mdp_examples.h"
#include "hist.h"
#include "utils.h"
#include "bamdp_solver.h"
#include "bamcp_solver.h"
#include "posterior_sampling_solver.h"
#include "hardcoded_solver.h"
#include "bamcp_maxprob_solver.h"
#include "bamdp_cvar_solver.h"
#include "bamcp_threshold_solver.h"
#include "mcts_cvar_sg.h"
#include "mcts_cvar_sg_offline.h"
#include "finite_mdp_belief.h"
#include "bamdp_cvar_decision_node.h"
#include "mcts_bamdp_cvar_sg.h"
#include "cvar_value_iteration.h"
#include "worst_case_value_iteration.h"
#include "value_iteration.h"
#include "domains.h"
#include "run_algorithms.h"
#include "cvar_expected_mdp_policy.h"
#include "agent_expected_mdp_policy.h"
#include "cvar_lexicographic.h"

#include "known_polytope_generators.h"
#include "random_walks/random_walks.hpp"

#include "volume/volume_sequence_of_balls.hpp"
#include "volume/volume_cooling_gaussians.hpp"
#include "volume/volume_cooling_balls.hpp"
#include "sampling/sampling.hpp"

void checkPathExists(const std::string &s)
{
  struct stat buffer;
  bool exists = (stat(s.c_str(), &buffer) == 0);
  if(!exists){
      std::cout << "Folder provided does not exist." << std::endl;
      std::exit(0);
  }
  return;
}

/* Run the experiments for the Autonomous Navigation domain */
void traffic_ssp_experiment(std::string folder){
    checkPathExists(folder);

    int startID = 717554;
    int goalID = 774827;
    bool includeHighways = true;
    bool addNoise = true;

    bool maximise = false; // cost minimisation
    int cvarInterpPts = 30;
    int costInterpPts = 100;
    int evalRepeats = 20000;
    std::vector<float> alphas{0.02, 0.2, 1.0};
    std::string dataset = "datasets/real_traffic_data/std_variable/am_district7_std5_2020_01-07_10blue.csv";

    std::string name;
    State initState;
    std::shared_ptr<MDP> pMDP;

    std::tie(initState, pMDP) = sspRealTrafficDomain(dataset, startID, goalID, includeHighways, addNoise);
    name = std::to_string(startID).append("_").append(std::to_string(goalID));

    std::shared_ptr<CvarValueIteration> cvarVI = runSSPCvarVI(
        folder,
        pMDP,
        initState,
        alphas,
        evalRepeats,
        cvarInterpPts,
        maximise,
        name
    );

    runSSPCvarLexicographicVI(
        folder,
        pMDP,
        initState,
        alphas,
        evalRepeats,
        cvarInterpPts,
        costInterpPts,
        maximise,
        name,
        cvarVI
    );
}


/* Run the experiments for the Betting Game domain */
void betting_game_experiment(std::string folder){
    checkPathExists(folder);

    bool maximise = false; // cost minimisation
    int cvarInterpPts = 30;
    int costInterpPts = 100;
    int evalRepeats = 20000;
    std::vector<float> alphas{0.02, 0.2, 1.0};

    State initState;
    std::shared_ptr<MDP> pMDP;
    std::tie(initState, pMDP) = bettingWithJackpotDomain();
    std::string name = "";

    std::shared_ptr<CvarValueIteration> cvarVI = runSSPCvarVI(
        folder,
        pMDP,
        initState,
        alphas,
        evalRepeats,
        cvarInterpPts,
        maximise,
        name
    );

    runSSPCvarLexicographicVI(
        folder,
        pMDP,
        initState,
        alphas,
        evalRepeats,
        cvarInterpPts,
        costInterpPts,
        maximise,
        name,
        cvarVI
    );
}


/* Run the experiments for the Deep Sea Treasure domain */
void deep_sea_treasure_experiment(std::string folder){
    checkPathExists(folder);

    bool maximise = false; // cost minimisation
    int cvarInterpPts = 80;
    int costInterpPts = 100;
    int evalRepeats = 20000;
    bool isSSP = false;
    std::vector<float> alphas{0.02, 0.2, 1.0};

    State initState;
    std::shared_ptr<MDP> pMDP;
    std::tie(initState, pMDP) = deepSeaTreasureDomain();
    std::string name = "";

    auto begin = std::chrono::high_resolution_clock::now();
    std::unordered_map<State, float, StateHash> value;
    std::unordered_map<State, std::string, StateHash> policy;

    VI vi;
    std::tie(value, policy) = vi.valueIteration(*pMDP, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> time = end - begin;
    std::cout << time.count();

    std::shared_ptr<CvarValueIteration> cvarVI = runSSPCvarVI(
        folder,
        pMDP,
        initState,
        alphas,
        evalRepeats,
        cvarInterpPts,
        maximise,
        name,
        isSSP
    );

    runSSPCvarLexicographicVI(
        folder,
        pMDP,
        initState,
        alphas,
        evalRepeats,
        cvarInterpPts,
        costInterpPts,
        maximise,
        name,
        cvarVI,
        isSSP
    );
}


/* Run the experiments for the Inventory Control domain */
void inventory_management_experiment(std::string folder){
    checkPathExists(folder);

    bool maximise = false; // cost minimisation
    int cvarInterpPts = 30;
    int costInterpPts = 100;
    int evalRepeats = 20000;
    bool isSSP = false;
    std::vector<float> alphas{0.02, 0.2, 1.0};

    State initState;
    std::shared_ptr<MDP> pMDP;
    std::tie(initState, pMDP) = inventoryDomain();
    std::string name = "";

    auto begin = std::chrono::high_resolution_clock::now();
    std::unordered_map<State, float, StateHash> value;
    std::unordered_map<State, std::string, StateHash> policy;

    VI vi;
    std::tie(value, policy) = vi.valueIteration(*pMDP, false);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> time = end - begin;
    std::cout << time.count();

    std::shared_ptr<CvarValueIteration> cvarVI = runSSPCvarVI(
        folder,
        pMDP,
        initState,
        alphas,
        evalRepeats,
        cvarInterpPts,
        maximise,
        name,
        isSSP
    );

    runSSPCvarLexicographicVI(
        folder,
        pMDP,
        initState,
        alphas,
        evalRepeats,
        cvarInterpPts,
        costInterpPts,
        maximise,
        name,
        cvarVI,
        isSSP
    );
}

/* Runs all experiments and saves results in specified folder */
int main()
{
    std::string folder = "/home";
    betting_game_experiment(folder);
    deep_sea_treasure_experiment(folder);
    traffic_ssp_experiment(folder);
    inventory_management_experiment(folder);
    return 0;
}
