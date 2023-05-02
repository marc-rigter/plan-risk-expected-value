#include <iostream>
#include <cmath>
#include <string>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <queue>
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "gurobi_c++.h"
#include "cvar_value_iteration.h"
#include "cvar_lexicographic.h"
#include "cvar_hist.h"


CvarValueIteration::CvarValueIteration(int numPts_){
    alphaVals = getAlphaValues(numPts_);
    valueComputed = false;
    maxT = 0;
}

std::vector<float> CvarValueIteration::getAlphaValues(int numPts){
    std::vector<float> alphas = logspace(0.0, 1.0, numPts);
    for(unsigned int i = 0; i < alphas.size(); i++){
        alphas[i] -= 1.0;
        alphas[i] /= 10.0;
    }
    alphas.push_back(1.0);
    return alphas;
}

std::tuple<std::string, std::unordered_map<State, float, StateHash>> CvarValueIteration::getOptimalAction(
    std::shared_ptr<MDP> pMDP,
    State currentState,
    float currentAlpha,
    GRBEnv env,
    bool maximise
){

    std::unordered_map<std::string, std::string> stateMap = currentState.getStateMapping();
    stateMap["alpha"] = std::to_string(currentAlpha);
    State currentAugState(stateMap);

    std::unordered_map<State, float, StateHash> bestPerturbation;
    float bestValue;
    if(maximise){
        bestValue = -std::numeric_limits<float>::max();
    }else{
        bestValue = std::numeric_limits<float>::max();
    }

    std::string bestAction;
    for(std::string act : pMDP->getEnabledActions(currentState)){
        GRBModel model = getCvarLP(currentState, currentAlpha, act, cvarValueFunction, env, *pMDP, maximise);
        model.optimize();

        float Q = model.get(GRB_DoubleAttr_ObjVal);
        std::unordered_map<State, float, StateHash> perturbation;
        for(auto kv : pMDP->getTransitionProbs(currentState, act)){
            GRBVar pert = model.getVarByName(kv.first.toString());
            perturbation[kv.first] = pert.get(GRB_DoubleAttr_X);
        }
        if(maximise){
            if(Q > bestValue){
                bestValue = Q;
                bestAction = act;
                bestPerturbation = perturbation;
            }
        }else{
            if(Q < bestValue){
                bestValue = Q;
                bestAction = act;
                bestPerturbation = perturbation;
            }
        }
    }

    return std::make_tuple(bestAction, bestPerturbation);
}

/* Execute an episode using the cvar value computed on the expected MDP.
Note that value iteration must be run on the expected MDP before executing this
function.

Args:
    pExpectedMDP: a pointer to the expected MDP used to compute optimal actions.
    pTrueMDP: a pointer to the true MDP, used to sample transitions during the
        episode.
    initialState: the initial state in the MDP.
    initialAlpha: the initial alpha value.
*/
CvarHist CvarValueIteration::executeEpisode(
        std::shared_ptr<MDP> pExpectedMDP,
        std::shared_ptr<MDP> pTrueMDP,
        State initialState,
        float initialAlpha
){
    if(!valueComputed){
        std::cout << "Cannot execute episode. Must run value iteration first." << std::endl;
        CvarHist emptyHist;
        return emptyHist;
    }

    CvarHist history;
    State currentState = initialState;
    float currentAlpha = initialAlpha;

    GRBEnv env = GRBEnv();
    env.start();
    env.set(GRB_IntParam_OutputFlag, 0);

    for(int t = 0; t<= maxT; t++){
        std::string action;
        std::unordered_map<State, float, StateHash> perturbation;

        // optimise worst-case on zero prob paths
        if(currentAlpha < 1e-6){
            std::tie(action, perturbation) = getOptimalAction(
                                                    pExpectedMDP,
                                                    currentState,
                                                    1e-6,
                                                    env
                                            );

        // otherwise optimise action using value function
        }else{
            std::tie(action, perturbation) = getOptimalAction(
                                                    pExpectedMDP,
                                                    currentState,
                                                    currentAlpha,
                                                    env
                                            );
        }


        // append to the history
        history.addTransition(
                currentState,
                action,
                currentAlpha,
                pExpectedMDP->getTransitionProbs(currentState, action),
                perturbation,
                pExpectedMDP->getReward(currentState, action));

        // Sample the next state according to the best action
        State nextState = pTrueMDP->sampleSuccessor(currentState, action);

        // Update the alpha value according to the best perturbation.
        currentAlpha *= perturbation[nextState];
        if(currentAlpha > 1.0){
            currentAlpha = 1.0;
        }

        // set the current state the the next state.
        currentState = nextState;
    }

    return history;
}

/* execute an episode in an SSP until a goal state is reached

perturbedProbs defines whether the real MDP probabilities or the perturbed
MDP probabilities are used to simulate the transitions during the episode.
*/
CvarHist CvarValueIteration::sspExecuteEpisode(
        std::shared_ptr<MDP> pMDP,
        State initialState,
        float initialAlpha,
        std::unordered_map<State, std::string, StateHash> worstCasePolicy,
        bool maximise,
        bool perturbedProbs
){
    if(!valueComputed){
        std::cout << "Cannot execute episode. Must run value iteration first." << std::endl;
        CvarHist emptyHist;
        return emptyHist;
    }

    CvarHist history;
    State currentState = initialState;
    float currentAlpha = initialAlpha;

    GRBEnv env = GRBEnv();
    env.start();
    env.set(GRB_IntParam_OutputFlag, 0);

    while(!(pMDP->isGoal(currentState))){
        std::string action;
        std::unordered_map<State, float, StateHash> perturbation;

        // optimise worst-case on zero prob paths
        if(currentAlpha < 1e-6){
            action = worstCasePolicy[currentState];
            for(auto pair : pMDP->getTransitionProbs(currentState, action)){
                perturbation[pair.first] = 1.0;
            }

        // otherwise optimise action using value function
        }else{
            std::tie(action, perturbation) = getOptimalAction(
                                                    pMDP,
                                                    currentState,
                                                    currentAlpha,
                                                    env,
                                                    maximise
                                            );
        }


        // append to the history
        history.addTransition(
                currentState,
                action,
                currentAlpha,
                pMDP->getTransitionProbs(currentState, action),
                perturbation,
                pMDP->getReward(currentState, action));

        // Sample the next state according to the best action
        State nextState;
        if(!perturbedProbs){
            nextState = pMDP->sampleSuccessor(currentState, action);
        }else{
            nextState = pMDP->samplePerturbedSuccessor(currentState, action, perturbation);
        }

        // Update the alpha value according to the best perturbation.
        currentAlpha *= perturbation[nextState];
        if(currentAlpha > 1.0){
            currentAlpha = 1.0;
        }

        if(currentAlpha < 0.0){
            currentAlpha = 0.0;
        }

        // set the current state the the next state.
        currentState = nextState;
    }

    return history;
}

/* execute an episode in an SSP until a goal state is reached */
CvarHist CvarValueIteration::sspExecuteEpisodeLexicographic(
        std::shared_ptr<MDP> pMDP,
        State initialState,
        float initialAlpha,
        std::unordered_map<State, std::string, StateHash> worstCasePolicy,
        CvarLexicographic lexSolver,
        bool maximise
){
    if(!valueComputed){
        std::cout << "Cannot execute episode. Must run value iteration first." << std::endl;
        CvarHist emptyHist;
        return emptyHist;
    }

    CvarHist history;
    State currentState = initialState;
    float currentAlpha = initialAlpha;
    float costSoFar = 0.0;

    GRBEnv env = GRBEnv();
    env.start();
    env.set(GRB_IntParam_OutputFlag, 0);

    bool lexico = false;
    while(!(pMDP->isGoal(currentState))){
        std::string action;
        std::string worstCaseAction;
        std::unordered_map<State, float, StateHash> dot;
        std::unordered_map<State, float, StateHash> perturbation;

        // optimise worst-case on zero prob paths
        if(currentAlpha < 1e-4){
            if(!lexico){
                std::cout << "switching to expected value optimisation..." << std::endl;
                lexico = true;
            }

            action = lexSolver.getOptimalAction(
                                                pMDP,
                                                currentState,
                                                costSoFar
                                            );

            worstCaseAction = worstCasePolicy[currentState];
            for(auto pair : pMDP->getTransitionProbs(currentState, action)){
                perturbation[pair.first] = 1.0;
            }

            if(worstCaseAction == action && action != "cost"){
                std::cout << "Same action as worst-case policy." << std::endl;
            }else if(worstCaseAction != action && action != "cost"){
                std::cout << "Different action to worst-case policy." << std::endl;
            }

        // otherwise optimise action using value function
        }else{
            std::tie(action, perturbation) = getOptimalAction(
                                                    pMDP,
                                                    currentState,
                                                    currentAlpha,
                                                    env,
                                                    maximise
                                            );
        }


        // append to the history
        float cost = pMDP->getReward(currentState, action);
        costSoFar += cost;
        history.addTransition(
                currentState,
                action,
                currentAlpha,
                pMDP->getTransitionProbs(currentState, action),
                perturbation,
                cost);

        // Sample the next state according to the best action
        State nextState = pMDP->sampleSuccessor(currentState, action);

        // Update the alpha value according to the best perturbation.
        currentAlpha *= perturbation[nextState];
        if(currentAlpha > 1.0){
            currentAlpha = 1.0;
        }

        if(currentAlpha < 0.0){
            currentAlpha = 0.0;
        }

        // set the current state the the next state.
        currentState = nextState;
    }

    return history;
}

/* Execute an episode using the cvar value computed on the BAMDP.
Note that value iteration must be run on the BAMDP before executing this
function.

Args:
    pBamdp: a pointer to the BAMDP used to compute optimal actions.
    pTrueMDP: a pointer to the true MDP, used to sample transitions during the
        episode.
    initialState: the initial state in the MDP - this state does not include the
        history.
    initialAlpha: the initial alpha value.
*/
CvarHist CvarValueIteration::executeBamdpEpisode(
        std::shared_ptr<MDP> pBamdp,
        std::shared_ptr<MDP> pTrueMDP,
        State initialState,
        float initialAlpha
){
    if(!valueComputed){
        std::cout << "Cannot execute episode. Must run value iteration first." << std::endl;
        CvarHist emptyHist;
        return emptyHist;
    }

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["history"] = initialState.toString();
    stateMap["t"] = "0";
    State initialHistoryState(stateMap);

    CvarHist history;
    State currentHistoryState = initialHistoryState;
    State currentState = initialState;
    float currentAlpha = initialAlpha;

    GRBEnv env = GRBEnv();
    env.start();
    env.set(GRB_IntParam_OutputFlag, 0);

    for(int t = 0; t<= maxT; t++){
        std::string action;
        std::unordered_map<State, float, StateHash> perturbation;

        // if alpha is at or near zero switch to optimising worst-case
        if(currentAlpha < 1e-6){
            std::tie(action, perturbation) = getOptimalAction(
                                                    pBamdp,
                                                    currentHistoryState,
                                                    1e-6,
                                                    env
                                            );

        // otherwise optimise action using value function
        }else{
            std::tie(action, perturbation) = getOptimalAction(
                                                    pBamdp,
                                                    currentHistoryState,
                                                    currentAlpha,
                                                    env
                                            );
        }


        // append to the history
        history.addTransition(
                currentHistoryState,
                action,
                currentAlpha,
                pBamdp->getTransitionProbs(currentHistoryState, action),
                perturbation,
                pBamdp->getReward(currentHistoryState, action));

        // Sample the next state according to the best action
        State nextState = pTrueMDP->sampleSuccessor(currentState, action);

        // Update the alpha value according to the best perturbation.
        stateMap["history"] = getNewHistory(currentHistoryState, action, nextState);
        stateMap["t"] = std::to_string(std::stoi(currentHistoryState.getValue("t"))+1);
        State nextHistoryState(stateMap);

        currentAlpha *= perturbation[nextHistoryState];
        if(currentAlpha > 1.0){
            currentAlpha = 1.0;
        }

        // set the current state the the next state.
        currentState = nextState;
        currentHistoryState = nextHistoryState;
    }

    return history;
}


GRBModel CvarValueIteration::getCvarLP(
    State currentState,
    float currentAlpha,
    std::string act,
    std::unordered_map<State, float, StateHash>& cvarValue,
    GRBEnv& env,
    MDP& m,
    bool maximise
){
    GRBModel model = GRBModel(env);
    float pertMax = 1.0/currentAlpha;
    GRBLinExpr probSum;

    // initialise the Q-value to equal the reward for this state action pair
    GRBLinExpr qVal = m.getReward(currentState, act);

    int i = 0;
    for(auto pair : m.getTransitionProbs(currentState, act)){
        State nextState = pair.first;
        float nextProb = pair.second;

        // separate variable for the perturbation of each successor state
        GRBVar perturbation = model.addVar(0.0, pertMax, 0.0, GRB_CONTINUOUS, nextState.toString());
        probSum += perturbation * nextProb;

        // declare lambdas which are coefficients for the linear piecewise approx.
        GRBLinExpr sumLambdaAlpha;
        GRBLinExpr sumLambdaValue;
        GRBLinExpr sumLambda;
        std::vector<GRBVar> lambdas;
        std::vector<double> wts;
        for(auto alpha : alphaVals){
            GRBVar lambda = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
            lambdas.push_back(lambda);
            wts.push_back(1.0);
            sumLambda += lambda;

            // interpolation for the value of perturbation applied
            sumLambdaAlpha += lambda * alpha / currentAlpha;

            // interpolation for the value of the successor state
            std::unordered_map<std::string, std::string> stateMap = nextState.getStateMapping();
            stateMap["alpha"] = std::to_string(alpha);
            State nextAugState = State(stateMap);
            sumLambdaValue += lambda * cvarValue[nextAugState];
        }

        // lambdas must sum to 1 and sos2 constr for linear piecewise
        model.addConstr(sumLambda == 1.0);
        model.addConstr(sumLambdaAlpha == perturbation);
        model.addSOS(&lambdas[0], &wts[0], lambdas.size(), GRB_SOS_TYPE2);

        // Equation 10. Risk-Sensitive and Robust Decision-Making: a
        // CVaR Optimization Approach, 2015
        qVal += sumLambdaValue * nextProb / currentAlpha;
        i++;
    }
    model.addConstr(probSum == 1.0, "Probability sum");
    if(maximise){
        model.setObjective(qVal, GRB_MINIMIZE);
    }else{
        model.setObjective(qVal, GRB_MAXIMIZE);
    }

    return model;
}

State CvarValueIteration::augToNormalState(State augState){
    std::unordered_map<std::string, std::string> stateMap = augState.getStateMapping();
    stateMap.erase("alpha");
    return State(stateMap);
}

void CvarValueIteration::cvarBackup(
    State currentAugState,
    std::unordered_map<State, float, StateHash>& cvarValue,
    GRBEnv env,
    MDP& m,
    bool maximise
){
    State currentState = augToNormalState(currentAugState);
    float currentAlpha = std::stof(currentAugState.getValue("alpha"));

    // if alpha is zero treat as if very near zero to avoid div by zero
    if(currentAlpha <= 1e-6){
        currentAlpha = 1e-6;
    }

    // separately compute the Q-value for each action using an LP to find worst
    // adversary perturbation
    float bestValue = std::numeric_limits<float>::max();
    if(maximise){
        bestValue *= -1.0;
    }

    std::string bestAction;
    std::unordered_map<State, float, StateHash> bestPerturbation;
    for(auto act : m.getEnabledActions(currentState)){
        GRBModel model = getCvarLP(currentState, currentAlpha, act, cvarValue, env, m, maximise);
        model.optimize();

        // record the Q-value for the best action for the agent
        float Q = model.get(GRB_DoubleAttr_ObjVal);
        std::unordered_map<State, float, StateHash> perturbation;
        for(auto kv : m.getTransitionProbs(currentState, act)){
            GRBVar pert = model.getVarByName(kv.first.toString());
            perturbation[kv.first] = pert.get(GRB_DoubleAttr_X);
        }


        if(maximise){
            if(Q > bestValue){
                bestValue = Q;
                bestAction = act;
                bestPerturbation = perturbation;
            }
        }else{
            if(Q < bestValue){
                bestValue = Q;
                bestAction = act;
                bestPerturbation = perturbation;
            }
        }
    }

    // the values we backup for the linear interpolation are alpha * value
    // see Page 6. Risk-Sensitive and Robust Decision-Making: a
    // CVaR Optimization Approach, 2015
    cvarValue[currentAugState] = bestValue*currentAlpha;

    if(cvarPolicy.find(currentState) == cvarPolicy.end()){
        std::map<std::string, std::string> pol;
        pol[currentAugState.getValue("alpha")] = bestAction;
        cvarPolicy[currentState] = pol;

        std::map<std::string, std::unordered_map<State, float, StateHash>> pertPol;
        pertPol[currentAugState.getValue("alpha")] = bestPerturbation;
        cvarPerturbationPolicy[currentState] = pertPol;
    }else{
        cvarPolicy[currentState][currentAugState.getValue("alpha")] = bestAction;
        cvarPerturbationPolicy[currentState][currentAugState.getValue("alpha")] = bestPerturbation;
    }
}

/* performs cvar value iteration on an ssp which does not have a t state factor */
std::unordered_map<State, float, StateHash> CvarValueIteration::sspValueIteration(MDP& m, bool maximise){
    std::vector<State> states = m.enumerateStates();
    std::unordered_map<State, float, StateHash> cvarValue;

    std::vector<State> augmentedStates;
    for(State s : states){
        std::unordered_map<std::string, std::string> stateMap = s.getStateMapping();
        for(float alpha : alphaVals){
            stateMap["alpha"] = std::to_string(alpha);
            State augState(stateMap);
            augmentedStates.push_back(augState);
        }
    }

    // initialise value to zero at all augmented states
    for(State s : augmentedStates){
        cvarValue[s] = 0.0;
    }

    auto begin = std::chrono::high_resolution_clock::now();

    GRBEnv env = GRBEnv();
    env.start();
    env.set(GRB_IntParam_OutputFlag, 0);
    float maxError = std::numeric_limits<float>::max();
    int iter = 0;
    while(maxError > 1e-3){
        maxError = 0.0;
        for(State s : augmentedStates){
            float cvarOld = cvarValue[s];
            cvarBackup(s, cvarValue, env, m, maximise);
            float error = std::fabs(cvarValue[s] - cvarOld);
            if(error > maxError){
                maxError = error;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> time = end - begin;
        std::cout << "iteration: " << iter << ", error: " << maxError << ", time: " <<  time.count() << std::endl;
        iter++;
    }

    valueComputed = true;
    cvarValueFunction = cvarValue;
    return cvarValue;
}

/* computes the best possible return in the markov chain induced by the optimal
agent and adversary policy. Because the markov chain is defined over a continuous
state space we again use approximate dynamic programming with linear interpolation.

Note that the best possible return under the agent policy and adversarial perturbations
corresponds to the VaR, as this is the worst return which contributes to CVaR.
*/
std::unordered_map<State, float, StateHash> CvarValueIteration::sspGetVaR(
    State initState,
    std::shared_ptr<MDP> m,
    bool maximise,
    int numVarInterpPts
){

    // this will be a "value function" corresponding to the worst possible return
    // from each state-alpha pair.
    std::unordered_map<State, float, StateHash> bestReturnValue;
    varAlphaVals = getAlphaValues(numVarInterpPts);

    // only need to compute var for the states reachable under the policy
    std::vector<State> reachableStates = getReachableStates(initState, m, maximise);

    // REMOVE
    std::cout << "Original number of states: " << m->enumerateStates().size() << ", reachable states for VaR: " << reachableStates.size() << std::endl;

    for(State s : reachableStates){
        std::unordered_map<std::string, std::string> stateMap = s.getStateMapping();
        for(float alph : varAlphaVals){
            stateMap["alpha"] = std::to_string(alph);
            State augState(stateMap);
            bestReturnValue[augState] = 0.0;
        }
    }

    auto begin = std::chrono::high_resolution_clock::now();

    GRBEnv env = GRBEnv();
    env.start();
    env.set(GRB_IntParam_OutputFlag, 0);
    float maxError = std::numeric_limits<float>::max();
    while(maxError > 1e-3){
        maxError = 0.0;
        for(auto pair : bestReturnValue){
            State s = pair.first;
            float bestReturnOld = bestReturnValue[s];
            varBackup(s, bestReturnValue, env, m, maximise);
            float error = std::fabs(bestReturnValue[s] - bestReturnOld);
            if(error > maxError){
                maxError = error;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> time = end - begin;
        std::cout << "error: " << maxError << ", time: " <<  time.count() << std::endl;
    }

    return bestReturnValue;
}

/* Get the states in the base MDP which are reachable under the policy of
the agent against the adversary */
std::vector<State> CvarValueIteration::getReachableStates(
        State initState,
        std::shared_ptr<MDP> pMDP,
        bool maximise
){
    std::vector<State> stateList;
    std::queue<State> queue;
    GRBEnv env = GRBEnv();
    env.start();
    env.set(GRB_IntParam_OutputFlag, 0);

    queue.push(initState);
    while(queue.size() > 0){

          // pop from queue
          State currentState = queue.front();
          queue.pop();
          stateList.push_back(currentState);

          // get all of the actions which may be taken by the agent across
          // range of alpha values
          std::vector<std::string> actionList;
          std::string action;
          std::unordered_map<State, float, StateHash> perturbation;
          for(float alpha : alphaVals){
              std::tie(action, perturbation) = getOptimalAction(
                                                      pMDP,
                                                      currentState,
                                                      alpha,
                                                      env,
                                                      maximise
                                              );

              if(std::find(actionList.begin(), actionList.end(), action) == actionList.end()){
                  actionList.push_back(action);
              }
          }

          // get all possible successor states for the actions that the agent may take
          std::vector<State> successorStates;
          for(std::string act : actionList){
              std::unordered_map<State, float, StateHash> transProbs = pMDP->getTransitionProbs(currentState, act);
              for(auto pair : transProbs){
                  if(std::find(successorStates.begin(), successorStates.end(), pair.first) == successorStates.end()){
                      successorStates.push_back(pair.first);
                  }
              }
          }

          // add successor states to the queue if they are not already in the list
          for(State s : successorStates){
              if(std::find(stateList.begin(), stateList.end(), s) == stateList.end()){
                  queue.push(s);
              }
          }
    }

    return stateList;
}

/* performs a backup to compute the value at risk. This backup assumes that
the agent transitions to the best possible successor under the policy of the
agent and adversary. This therefore computes the best possible return under the
agent and adversary policy, which corresponds to VaR. */
void CvarValueIteration::varBackup(
    State currentAugState,
    std::unordered_map<State, float, StateHash>& bestReturnValue,
    GRBEnv env,
    std::shared_ptr<MDP> m,
    bool maximise
){
    State currentState = augToNormalState(currentAugState);
    float currentAlpha = std::stof(currentAugState.getValue("alpha"));

    // if alpha is zero treat as if very near zero to avoid div by zero
    if(currentAlpha <= 1e-6){
        currentAlpha = 1e-6;
    }

    // get the actions chosen by the agent and adversary
    std::string action;
    std::unordered_map<State, float, StateHash> perturbation;
    std::tie(action, perturbation) = getOptimalAction(
                                            m,
                                            currentState,
                                            currentAlpha,
                                            env,
                                            maximise
                                    );

    std::unordered_map<State, float, StateHash> nominalProbs = m->getTransitionProbs(currentState, action);
    float bestReturn;
    if(maximise){
        bestReturn = -std::numeric_limits<float>::max();
    }else{
        bestReturn = std::numeric_limits<float>::max();
    }

    // check each successor in the induced markov chain
    for(auto pair : nominalProbs){
        State successor = pair.first;
        std::unordered_map<std::string, std::string> stateMap = successor.getStateMapping();
        float pert = perturbation[successor];

        // only include paths which are possible under the adversary perturbations
        if(pert*pair.second > 1e-3){
            float newAlpha = currentAlpha * pert;

            // compute the best possible return by assuming we definitely transition
            // to best successor (no probabilitistc)
            float totalReturn = m->getReward(currentState, action) + interpVarAlphaValue(bestReturnValue, successor, newAlpha);

            if(maximise){
                if(totalReturn > bestReturn){
                    bestReturn = totalReturn;
                }
            }else{
                if(totalReturn < bestReturn){
                    bestReturn = totalReturn;
                }
            }
        }
    }
    bestReturnValue[currentAugState] = bestReturn;
}

/* perform linear interpolation for the value between the two nearest alpha
values */
float CvarValueIteration::interpVarAlphaValue(
        std::unordered_map<State, float, StateHash>& valueFunction,
        State s,
        float alpha
){
    float currentAlpha = alpha;
    float alphaAbove;
    float alphaBelow;
    bool aboveFound = false;

    // find the nearest alpha values to interpolate between
    for(auto alpha : varAlphaVals){
        if(alpha >= currentAlpha && !aboveFound){
            alphaAbove = alpha;
            aboveFound = true;
        }

        if(alpha < currentAlpha){
            alphaBelow = alpha;
        }
    }

    std::unordered_map<std::string, std::string> stateMap = s.getStateMapping();
    stateMap["alpha"] = std::to_string(alphaBelow);
    State stateAlphaBelow(stateMap);
    stateMap["alpha"] = std::to_string(alphaAbove);
    State stateAlphaAbove(stateMap);

    // avoid division by zero if exactly equal to value
    if(cmpf(currentAlpha, alphaAbove, 1e-6)){
        return valueFunction[stateAlphaAbove];
    }

    if(cmpf(currentAlpha, alphaBelow, 1e-6)){
        return valueFunction[stateAlphaBelow];
    }

    // linear interpolation
    float interp = valueFunction[stateAlphaBelow]*(alphaAbove - currentAlpha);
    interp += valueFunction[stateAlphaAbove]*(currentAlpha - alphaBelow);
    interp /= (alphaAbove - alphaBelow);

    return interp;
}



/* Performs cvar value iteration on the MDP provided. This assumes that the
problem is finite horizon. This function assumes that the goal of the agent
is to maximise the reward.

Args:
    m: the MDP to perform value iteration upon.

Returns:
    tuple containing the value as a map from states with timestep and alpha
    value to values.
*/
std::unordered_map<State, float, StateHash> CvarValueIteration::valueIteration(MDP& m, bool maximise){
    std::vector<State> states = m.enumerateStates();
    std::unordered_map<State, float, StateHash> cvarValue;

    for(State s : states){
        std::vector<std::string> sf = s.getStateFactors();
        if(std::find(sf.begin(), sf.end(), "t") == sf.end()){
            std::cerr << "Error: Need MDP states to have a t state factor for finite horizon CVaR value iteration" << std::endl;
            std::exit(-1);
        }
    }

    // create new set of states which includes the alpha value
    std::vector<State> augmentedStates;
    for(State s : states){
        std::unordered_map<std::string, std::string> stateMap = s.getStateMapping();

        if(std::stoi(stateMap["t"]) > maxT){
            maxT = std::stoi(stateMap["t"]);
        }

        for(float alpha : alphaVals){
            stateMap["alpha"] = std::to_string(alpha);
            State augState(stateMap);
            augmentedStates.push_back(augState);
        }
    }

    // set the cvar value to zero at the final time step
    for(State s : augmentedStates){
        if(std::stoi(s.getValue("t")) == maxT){
            std::unordered_map<std::string, std::string> stateMap = s.getStateMapping();
            stateMap["t"] = std::to_string(maxT);
            cvarValue[State(stateMap)] = 0.0;
        }
    }

    // loop backwards through the horizon
    GRBEnv env = GRBEnv();
    env.start();
    env.set(GRB_IntParam_OutputFlag, 0);
    for(int t = maxT - 1; t >= 0; t--){
        std::cout << "t: " << t << std::endl;
        for(State s : augmentedStates){
            if(std::stoi(s.getValue("t")) == t){
                std::unordered_map<std::string, std::string> stateMap = s.getStateMapping();
                stateMap["t"] = std::to_string(t);
                State currentAugState(stateMap);
                cvarBackup(currentAugState, cvarValue, env, m, maximise);
            }
        }
    }

    valueComputed = true;
    cvarValueFunction = cvarValue;
    return cvarValue;
}
