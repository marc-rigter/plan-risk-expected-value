#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <random>
#include <memory>
#include <values.h>
#include <gmp.h>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "cvar_adv_decision_node.h"
#include "cvar_chance_node.h"
#include "bamdp_rollout_policy.h"


#include <shogun/lib/config.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/ConstMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/base/init.h>
#include <shogun/evaluation/GradientCriterion.h>
#include <shogun/evaluation/GradientEvaluation.h>
#include <shogun/modelselection/GradientModelSelection.h>

#define PPL_NO_AUTOMATIC_INITIALIZATION
#include <ppl.hh>
namespace ppl = Parma_Polyhedra_Library;
using namespace Parma_Polyhedra_Library::IO_Operators;

#include "random_walks/random_walks.hpp"
#include "known_polytope_generators.h"
#include "sampling/sampling.hpp"
#include "volume/volume_sequence_of_balls.hpp"

float MIN_BIAS_ADV = 5.0;


CvarAdvDecisionNode::CvarAdvDecisionNode(
        std::shared_ptr<MDP> pMDP_,
        const State state_,
        const float alpha_,
        const std::string action_)
        : CvarDecisionNode(pMDP_, state_, alpha_), action(action_)
{
    updatePerturbationBudget();
}


/* Get the unperturbed successor probabilites for executing the state action
pair associated with this adversary decision node. This is simply the
successor probabilities from the underlying MDP.

Args:
    None

Returns:
    the unperturbed successor probabilities for executing this state action
    pair from the underlying MDP.
*/
std::unordered_map<State, float, StateHash> CvarAdvDecisionNode::getSuccessorProbs(){
    return pMDP->getTransitionProbs(s, action);
}

/* select the most promising action according to the lowest lower confidence
bound (as the adversary is trying to minimise to reward) */
int CvarAdvDecisionNode::selectPerturbationActionUCB(float biasFactor){

    // select the action which has the lowest lower confidence bound.
    float bias = std::max(MIN_BIAS_ADV, std::fabs(getCurrentValue()*biasFactor));
    float minUcb = std::numeric_limits<float>::max();
    std::shared_ptr<CvarChanceNode> pChildNode;
    int minActionInd;
    int actionInd;
    int childObs;
    float ucb;
    for(auto pair : chanceNodeChildren){
        actionInd = pair.first;
        pChildNode = chanceNodeChildren[actionInd];
        childObs = pChildNode->getObservations();

        // subtract the exploration bonus for the minimising adversary
        ucb = pChildNode->getCurrentValue();
        ucb -= bias * sqrt(log(observations)/childObs);

        if(ucb < minUcb){
            minUcb = ucb;
            minActionInd = actionInd;
        }
    }
    return minActionInd;
}

void CvarAdvDecisionNode::updatePerturbationBudget(){

    // set the perturbation budget for this node based on alpha
    if(cmpf(alpha, 0.0, 1e-6)){
        perturbationBudget = 1e6;
    }else{
        perturbationBudget = 1.0/this->alpha;
    }
}

/* Returns the action with the lowest estimated Q value which corresponds to
the best action for the adversary.

Returns:
    string: action with best Q value
*/
int CvarAdvDecisionNode::getBestAdvAction(){
    int bestAction;
    float lowestValue = std::numeric_limits<float>::max();
    float value;

    for(auto pair : chanceNodeChildren){
        value = pair.second->getCurrentValue();
        if(value < lowestValue){
            lowestValue = value;
            bestAction = pair.first;
        }
    }
    return bestAction;
}

/* Enumerate all of the perturbation actions which lie on vertices of the
polytope induced by the probability simplex and the constraints on the
perturbation budget. */
void CvarAdvDecisionNode::enumerateVertexPerturbations(){
    Parma_Polyhedra_Library::initialize();
    std::unordered_map<State, float, StateHash> successorProbs = getSuccessorProbs();

    // generate polyhedron constraints
    ppl::Constraint_System cs;
    long constant = 1e9; // constant to scale integers
    int i = 0;
    ppl::Linear_Expression sum = ppl::Linear_Expression(0);

    for(auto pair : successorProbs){
        ppl::Variable x(i);
        cs.insert(x >= 1e6);
        cs.insert(x <= constant);
        cs.insert(x <= round(pair.second*perturbationBudget*constant));
        sum += x;
        i++;
    }

    // enforces probability sums to 1
    cs.insert(sum == constant);

    // loop through vertices
    ppl::C_Polyhedron ph(cs);
    std::unordered_map<State, float, StateHash> perturbationMapping;
    ppl::Generator_System gs = ph.generators(); // Use ph.minimized_generators() to minimal set of points for the polytope
    for(ppl::Generator_System::const_iterator it = gs.begin(); it != gs.end(); it++) {
      const ppl::Generator& g = *it;
      if(!g.is_point()){
          continue;
      }

      // for each vertex convert back to perturbation mapping
      int i = 0;
      for(auto pair : successorProbs){
          float newProb = ppl::raw_value(g.coefficient(ppl::Variable(i))).get_d()/(float)constant;
          perturbationMapping[pair.first] = newProb/pair.second;
          i++;
      }
      vertexPerturbations.push_back(perturbationMapping);
    }

    // also add no perturbation as a vertex
    i = 0;
    for(auto pair : successorProbs){
        perturbationMapping[pair.first] = 1.0;
        i++;
    }
    vertexPerturbations.push_back(perturbationMapping);

    Parma_Polyhedra_Library::finalize();
}

/* Expand an action for the adversary in the Cvar SG. Note that a move for
the adversary in this game corresponds to choosing perturbation factors
for the successor states that obey the perturbation budget and keep the
distribution over successor nodes as a valid distribution.

To perform sampling, a possible perturbed probability distribution is sampled
uniformly from the simplex of the appropriate dimension. Then this possible
perturbed distribution is checked to see if constraints are met.

Args:
    None

Returns:
    a mapping from successor states to probability perturbation factor.*/
int CvarAdvDecisionNode::expandRandomPerturbation(){
    std::unordered_map<State, float, StateHash> perturbationMapping;
    int actionInd;

    // if there is little perturbation budget left or only one successor then
    // just return null perturbation
    if(doNotExpand()){
        actionInd = getNoPerturbationAction();

    // otherwise we can expand a new random action
    }else{
        std::tuple<bool, std::vector<std::unordered_map<State, float, StateHash>>> tup = sampleRandomPerturbations(1);
        perturbationMapping = std::get<1>(tup).front();
        actionInd = addChildChanceNode(perturbationMapping);
    }
    return actionInd;
}

/* Returns a boolean defining whether another action should not be expanded.
In the case that the adversary has only a small budget we will not sample
another adv action or if there is only one successor in which case there
are no probabilities to perturb */
bool CvarAdvDecisionNode::doNotExpand(){
    float eps = 1e-2;
    int numSuccessors = getSuccessorProbs().size();
    if(1.0/alpha < 1.0 + eps || numSuccessors == 1){
        return true;
    }else{
        return false;
    }
}


/* Expand an action for the adversary in the Cvar SG using bayesian optimisation
based on the current estimates for the child nodes.

Args:
    None

Returns:
    a mapping from successor states to probability perturbation factor.*/
int CvarAdvDecisionNode::expandPerturbationBayesOpt(){
    std::unordered_map<State, float, StateHash> perturbationMapping;
    int actionInd;

    // if there are no actions expanded expand randomly
    if(chanceNodeChildren.size() == 0){
        actionInd = expandRandomPerturbation();

    // check if we shouldn't expand new action
    }else if(doNotExpand()){
        actionInd = getNoPerturbationAction();

    // otherwise expand using bayes opt
    }else{
        std::unordered_map<State, float, StateHash> perturbationMapping = sampleMaxExpectedImprovementPerturbation();
        actionInd = addChildChanceNode(perturbationMapping);
    }

    return actionInd;
}

/* Returns a perturbation action which does not modify transiton probs */
int CvarAdvDecisionNode::getNoPerturbationAction(){
    int actionInd;

    // if no children actions add no perturbation action
    if(chanceNodeChildren.size() == 0){
        std::unordered_map<State, float, StateHash> perturbationMapping = getNoPerturbationMapping();
        actionInd = addChildChanceNode(perturbationMapping);

    // if already exists don't expand new null perturbaion
    }else{
        actionInd = actionMapping.size() - 1;
    }
    return actionInd;
}

/* Returns perturbation mapping which does not modify transition probs */
std::unordered_map<State, float, StateHash> CvarAdvDecisionNode::getNoPerturbationMapping(){
    std::unordered_map<State, float, StateHash> perturbationMapping;
    std::unordered_map<State, float, StateHash> successorProbs;
    successorProbs = getSuccessorProbs();
    for(auto pair : successorProbs){
        perturbationMapping[pair.first] = 1.0;
    }
    return perturbationMapping;
}

/* Perform sampling of perturbed probability distribution by sampling
uniformly from the simplex of the appropriate dimension. Then this possible
perturbed distribution is checked to see if constraints are met.*/
std::tuple<bool, std::vector<std::unordered_map<State, float, StateHash>>> CvarAdvDecisionNode::sampleRandomPerturbations(int numSamples){
    std::vector<std::unordered_map<State, float, StateHash>> perturbationMappings;

    // if this is not a valid decision node to expand additional actions
    if(doNotExpand()){
        perturbationMappings.push_back(getNoPerturbationMapping());
        return std::make_tuple(true, perturbationMappings);
    }

    // get the successor states from the state action pair at this node
    std::unordered_map<State, float, StateHash> successorProbs;
    successorProbs = getSuccessorProbs();
    int numSuccessors = successorProbs.size();

    // calculate the maximum allowable value for each probability
    std::vector<State> stateList;
    std::vector<float> maxProbs;
    std::vector<float> initPoint;
    for(auto pair : successorProbs){
        maxProbs.push_back(std::min(1.0f, pair.second*perturbationBudget));
        stateList.push_back(pair.first);
        initPoint.push_back(pair.second);
    }

    // we do sampling on the simplex 1 dimension lower. the last probability is
    // constrained by needing to sum to 1
    initPoint.pop_back();
    int simplexDimension = numSuccessors - 1;

    typedef float NT;
    typedef Cartesian<NT>    Kernel;
    typedef typename Kernel::Point    Point;
    typedef HPolytope<Point> Hpolytope;

    typedef typename Hpolytope::MT    MT;
    typedef typename Hpolytope::VT    VT;
    typedef BoostRandomNumberGenerator<boost::mt19937, NT> RNGType;

    // matrices for defining linear constraints
    MT A;
    VT b;
    A.resize((simplexDimension)*2+2, simplexDimension);
    b.resize((simplexDimension)*2+2);

    // define linear equality constraints to sample on constrained simplex Ax <= b
    for(int i = 0; i < simplexDimension; i++){
        b(i) = maxProbs[i];
        b(i + simplexDimension) = 0.0;

        for(int j = 0; j < simplexDimension; j++){
            if(i == j){
                A(i, j) = 1.0;
                A(i + simplexDimension, j) = -1.0;
            }else{
                A(i, j) = 0.0;
                A(i + simplexDimension, j) = 0.0;
            }
        }
    }

    for(int j = 0; j < simplexDimension; j++){
        A((simplexDimension)*2, j) = 1.0;
        A((simplexDimension)*2+1, j) = -1.0;
    }
    b((simplexDimension)*2) = 1.0;
    b((simplexDimension)*2 + 1) = -1.0 + maxProbs[simplexDimension];

    Hpolytope P;
    P.init(simplexDimension, A, b);

    RNGType rng(simplexDimension);
    Point startingPoint(simplexDimension, initPoint);

    unsigned int walkLen = 10;
    unsigned int nBurns = 5;

    std::list <Point> randomPoints;
    uniform_sampling<CDHRWalk>(randomPoints, P, rng, walkLen, numSamples,
                                startingPoint, nBurns);

    // package points into perturbation mappings.
    for(auto pt : randomPoints){
        std::unordered_map<State, float, StateHash> perturbationMapping;
        float sum = 0.0;
        bool validSample = true;
        float total = 0.0;
        for(auto i = 0; i < numSuccessors; i++){
            State s = stateList.at(i);
            float prob;
            if(i < simplexDimension){
                prob = pt[i];
                perturbationMapping[s] = prob/successorProbs.at(s);
                sum += prob;
            }else{
                prob = 1.0 - sum;
                perturbationMapping[s] = prob/successorProbs.at(s);
            }
            total += prob;

            if(prob > 1.0 || std::isnan(prob)){
                validSample = false;
                break;
            }
        }

        if(!cmpf(total, 1.0, 1e-5)){
            validSample = false;
        }

        if(validSample){
            perturbationMappings.push_back(perturbationMapping);
        }else{
            std::cout << "***Warning***: Invalid perturbation mapping sampled.";
            perturbationMappings.push_back(getNoPerturbationMapping());
            return std::make_tuple(false, perturbationMappings);
        }
    }

    return std::make_tuple(true, perturbationMappings);
}


/* Perform sampling of perturbed probability distribution by sampling
uniformly from the simplex of the appropriate dimension. Then this possible
perturbed distribution is checked to see if constraints are met.*/
std::unordered_map<State, float, StateHash> CvarAdvDecisionNode::sampleMaxExpectedImprovementPerturbation(){
    shogun::init_shogun_with_defaults();
    std::vector<std::unordered_map<State, float, StateHash>> candidatePerturbations;
    int randomSamples;
    if(perturbationBudget < 1.1){
        randomSamples = 20;
    }else{
        randomSamples = 200;
    }

    // get random points to evaluate the expected improvement
    if(!perturbationsSampled){
        perturbationsSampled = true;
        std::tuple<bool, std::vector<std::unordered_map<State, float, StateHash>>> tup = sampleRandomPerturbations(randomSamples);
        bool res = std::get<0>(tup);
        if(res){
            sampledPerturbations = std::get<1>(tup);
        }else{
            sampledPerturbations.push_back(getNoPerturbationMapping());
        }
    }
    candidatePerturbations = sampledPerturbations;

    // if the perturbation sampling failed and there is only 1 return that
    if(candidatePerturbations.size() == 1){
        return candidatePerturbations[0];
    }

    // also get the perturbations corresponding to vertices of the polytope induced
    // by the constraints
    // if(!verticesEnumerated){
    //     verticesEnumerated = true;
    //     enumerateVertexPerturbations();
    // }
    // candidatePerturbations.insert(candidatePerturbations.end(), vertexPerturbations.begin(), vertexPerturbations.end());
    int numSamples = candidatePerturbations.size();
    int bestIndex;

    try{
        // put the data from the existing children nodes into the training set
        std::unordered_map<State, float, StateHash> successorProbs;
        successorProbs = getSuccessorProbs();

        int dims = successorProbs.size();
        index_t n = chanceNodeChildren.size();

        shogun::SGMatrix<float64_t> X(dims, n);
        shogun::SGMatrix<float64_t> X_test(dims, numSamples);
        shogun::SGMatrix<float64_t> Y(1, n);

        // put training data into appropriate matrices
        int dataNum = 0;
        for(auto pair : chanceNodeChildren){
            int actionInd = pair.first;
            std::unordered_map<State, float, StateHash> perturbation = actionMapping[actionInd];
            std::shared_ptr<CvarChanceNode> pChildNode = pair.second;

            int stateNum = 0;
            for(auto stateprob : successorProbs){
                State currentState = stateprob.first;
                X(stateNum, dataNum) = perturbation[currentState];
                stateNum++;
            }
            Y(dataNum, 0) = pChildNode->getCurrentValue();
            dataNum++;
        }

        // put test data into matrices
        dataNum = 0;
        for(auto pert : candidatePerturbations){
            int stateNum = 0;
            for(auto stateprob : successorProbs){
                State currentState = stateprob.first;
                X_test(stateNum, dataNum) = pert[currentState];
                stateNum++;
            }
            dataNum++;
        }

        shogun::CDenseFeatures<float64_t>* feat_train = new shogun::CDenseFeatures<float64_t>(X);
        SG_REF(feat_train);
        shogun::CDenseFeatures<float64_t>* feat_test = new shogun::CDenseFeatures<float64_t>(X_test);
        SG_REF(feat_test);
        shogun::CRegressionLabels* label_train = new shogun::CRegressionLabels(Y);

        // set default values for the parameters to initialise the gradient optimisation
        float64_t defaultNormalisedWidth = 0.2;
        float64_t defaultWidth  = perturbationBudget*defaultNormalisedWidth;
        float64_t defaultSigma = 1.0;

        shogun::CGaussianKernel* kernel = new shogun::CGaussianKernel(10, defaultWidth);

        // set prior mean lower to be optimisatic
        shogun::CConstMean* mean = new shogun::CConstMean(0.0);
        shogun::CGaussianLikelihood* lik = new shogun::CGaussianLikelihood();
        lik->set_sigma(defaultSigma);
        shogun::CExactInferenceMethod* inf = new shogun::CExactInferenceMethod(kernel, feat_train,
                                                                            mean, label_train, lik);
        shogun::CGaussianProcessRegression* gp = new shogun::CGaussianProcessRegression(inf);
        SG_REF(gp);

        shogun:: SGVector<float64_t> mean_vector;
        shogun:: SGVector<float64_t> variance_vector;
        mean_vector=gp->get_mean_vector(feat_test);
        variance_vector=gp->get_variance_vector(feat_test);


        float bestLCB = std::numeric_limits<float>::max();
        float factorUCB = 2.0;
        for(auto i = 0; i < numSamples; i++){
            float lcb = mean_vector[i] - std::sqrt(variance_vector[i])*factorUCB;
            if(lcb < bestLCB){
                bestLCB = lcb;
                bestIndex = i;
            }
        }

        // clean up
        SG_UNREF(gp);
        SG_UNREF(feat_test);
        SG_UNREF(feat_train);

    }catch(...){
        std::cout << "Warning: shogun threw an exception" << std::endl;
        bestIndex = 0;
    }

    // delete from samples so not to expand same point again
    if((unsigned int)bestIndex < sampledPerturbations.size()){
        sampledPerturbations.erase(sampledPerturbations.begin()+bestIndex);
    }else{
        vertexPerturbations.erase(vertexPerturbations.begin()+bestIndex-sampledPerturbations.size());
    }
    return candidatePerturbations[bestIndex];
}

/* add a child chance node corresponding to the action chosen by the adversary */
int CvarAdvDecisionNode::addChildChanceNode(std::unordered_map<State, float, StateHash> perturbationMapping){
    int actionInd = actionMapping.size();
    actionMapping[actionInd] = perturbationMapping;
    chanceNodeChildren[actionInd] = createChild(perturbationMapping);
    return actionInd;
}

std::shared_ptr<CvarChanceNode> CvarAdvDecisionNode::createChild(std::unordered_map<State, float, StateHash> perturbationMapping){
    return std::make_shared<CvarChanceNode>(pMDP, s, alpha, action, perturbationMapping);
}

/* If the perturbation has already been used up just expand a perturbation
action which does not change the probabilities */
int CvarAdvDecisionNode::expandNoPerturbation(){
    std::unordered_map<State, float, StateHash> perturbationMapping;

    // get the successor states from the state action pair at this node
    std::unordered_map<State, float, StateHash> successorProbs;
    successorProbs = getSuccessorProbs();

    for(auto pair : successorProbs){
        perturbationMapping[pair.first] = 1.0;
    }

    int actionInd = actionMapping.size();
    actionMapping[actionInd] = perturbationMapping;
    chanceNodeChildren[actionInd] = std::make_shared<CvarChanceNode>(pMDP, s, alpha, action, perturbationMapping);
    return actionInd;
}

std::unordered_map<int, std::shared_ptr<CvarChanceNode>>  CvarAdvDecisionNode::getChanceChildren(){
    return chanceNodeChildren;
}

std::unordered_map<State, float, StateHash>  CvarAdvDecisionNode::getPerturbationMapping(int index){
    return actionMapping[index];
}

std::string CvarAdvDecisionNode::getAction(){
    return action;
}
