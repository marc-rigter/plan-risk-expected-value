#ifndef ssp_mdp_examples
#define ssp_mdp_examples
#include "mdp.h"

std::shared_ptr<MDP> trafficSSP(int size);
std::shared_ptr<MDP> testExample();
std::shared_ptr<MDP> realTrafficSSP(
    std::string adjacencyMatrixFile,
    int startID,
    int goalID,
    bool includeHighways=true,
    bool addNoise = false
);
std::shared_ptr<MDP> makeBettingWithJackpotMDP(int stages);
std::shared_ptr<MDP> deepSeaTreasureMDP(int horizon);

#endif
