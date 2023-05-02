#ifndef bamdp_cvar_solver
#define bamdp_cvar_solver
#include <string>
#include "state.h"
#include "multimodel_mdp.h"
#include "hist.h"


/* Base class for Bayes adaptive MDP solvers. */
class BAMDPCvarSolver{
private:

public:
    BAMDPCvarSolver() {};
    float getVaR(
            MultiModelMDP& mmdp,
            State initState, int horizon,
            float alpha,
            std::pair<float, float> range,
            int numPoints,
            int numTrials,
            float biasFactor);
};

#endif
