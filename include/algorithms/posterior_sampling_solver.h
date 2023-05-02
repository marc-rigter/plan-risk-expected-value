#ifndef posterior_sampling
#define  posterior_sampling
#include "mdp.h"
#include "multimodel_mdp.h"
#include "state.h"
#include "bamdp_solver.h"

/* Class to implement the posterior sampling approach to solving Bayes
Adaptive MDPs.

Attributes:
    resampleInterval: the number of steps between which to sample a new MDP
        from the posterior and compute a corresponding policy.
    step: the current step within the resample interval.
    currentPolicy: the current policy to be executed corresponding to the
        posterior sample.
*/
class PosteriorSamplingSolver :  public BAMDPSolver {
private:
    int resampleInterval;
    int step;
    std::unordered_map<State, std::string, StateHash> currentPolicy;

public:
    PosteriorSamplingSolver(int resampleInterval_);
    virtual std::string getNextAction(MultiModelMDP& mmdp, State initState, int horizon);
};

#endif
