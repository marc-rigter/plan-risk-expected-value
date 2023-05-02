#include "state.h"
#include "multimodel_mdp.h"
#include "hist.h"
#include "bamdp_cvar_solver.h"
#include "bamcp_threshold_solver.h"
#include "mcts_decision_node.h"

/* Attempts to find the VaR corresponding to the optimal CVaR at confidence
level given by the parameter alpha. The approach is fairly brute force - in an
outer loop we try multiple candidate values for the VaR, and in the inner
loop we evaluate the corresponding CVaR.

Arguments:
    mmdp: the multimodel MDP that we are doing Bayes-adaptive planning over.
    initState: the root state of the problem.
    horizon: the finite horizon of the planning problem.
    alpha: the confidence level.
    range: a pair of floats specifying the minimum and maximum values to search
        to find the appropriate VaR value.
    numPoints: the number of points to attempt to evaluate in the outer loop
        optimisation.
    numTrials: the number of monte carlo trials to use for each candidate
        value
*/
float BAMDPCvarSolver::getVaR(
        MultiModelMDP& mmdp,
        State initState,
        int horizon,
        float alpha,
        std::pair<float, float> range,
        int numPoints,
        int numTrials,
        float biasFactor)
{
    float candidateVal;
    std::shared_ptr<MCTSDecisionNode> pNode;
    float belowThresholdValue;
    float bestOuterValue = -std::numeric_limits<float>::max();
    float varEstimate;

    for(int i = 0; i < numPoints; i++){
        candidateVal = range.first + (range.second - range.first) * ((float)i / (numPoints - 1));
        BAMCPThresholdSolver solver(candidateVal, biasFactor);
        float outerValue;

        pNode = solver.runMCTS(mmdp, initState, horizon, numTrials);
        belowThresholdValue = solver.estimateValue(mmdp, horizon, 500, pNode);

        outerValue = candidateVal + 1/(1-alpha)*belowThresholdValue;

        std::cout << "Estimated value: " << belowThresholdValue << " Node value: " << pNode->getCurrentValue() << std::endl;
        std::cout << "Candidate value for var: " << candidateVal << " Value for outer optimisation: " << outerValue << std::endl;
        if(outerValue > bestOuterValue){
            bestOuterValue = outerValue;
            varEstimate = candidateVal;
        }
    }

    return varEstimate;
}
