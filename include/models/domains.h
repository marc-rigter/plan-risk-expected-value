#ifndef domains
#define domains
#include "mdp.h"
#include "fully_tied_dirichlet_belief.h"

typedef std::tuple<int, std::shared_ptr<Belief>, std::shared_ptr<MDP>, State> domain;
domain bettingGame(int horizon);
domain bettingGameSmall();
domain bettingGameLarge();
domain medicalDomain(int horizon);
domain medicalDomainSmall();
domain medicalDomainLarge();
domain marsRover(int horizon);
domain marsRoverSmall();
domain marsRoverLarge();
domain trafficDomain();
domain trafficDomainRefactored();

std::tuple<State, std::shared_ptr<MDP>> sspTestDomain();
std::tuple<State, std::shared_ptr<MDP>> sspSyntheticTrafficDomain(int size);
std::tuple<State, std::shared_ptr<MDP>> sspRealTrafficDomain(
    std::string adjacencyMatrixFile,
    int startID,
    int goalID,
    bool includeHighways=true,
    bool addNoise=false
);
std::tuple<State, std::shared_ptr<MDP>> bettingWithJackpotDomain(
);
std::tuple<State, std::shared_ptr<MDP>> deepSeaTreasureDomain(
);
std::tuple<State, std::shared_ptr<MDP>> inventoryDomain();


#endif
