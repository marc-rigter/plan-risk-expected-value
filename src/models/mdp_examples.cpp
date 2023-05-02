#include <unordered_map>
#include <iostream>
#include <random>
#include "catch.h"
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "value_iteration.h"
#include "fully_tied_dirichlet_belief.h"
#include "tied_dirichlet_belief.h"
#include "mdp_examples.h"

/*  Generates and returns an MDP corresponding to navigation over a small grid.
*/
std::shared_ptr<MDP> makeSimpleMDP(int xGoal, int yGoal, float penalty){
    int xMax = 2;
    int yMax = 2;
    std::unordered_map<std::string, std::string> stateMap;
    std::vector<State> stateList;
    std::unordered_map<State, float, StateHash> initProbs;
    std::unordered_map<State, int, StateHash> mdpStateMap;
    std::unordered_map<std::string, int> mdpActionMap;
    std::vector<std::string> actionList{"up", "right", "down", "left"};
    int nStates = 10;
    int nActions = actionList.size();

    // generate state mapping
    int stateIndex = 0;
    for(int i = 0; i <= xMax; i++){
        for(int j = 0; j <= yMax; j++){

            // note that state feature values must be strings
            stateMap["x"] = std::to_string(i);
            stateMap["y"] = std::to_string(j);
            stateList.push_back(State(stateMap));

            // set initial state to (0, 0)
            if(i == 0 && j == 0){
                initProbs[stateList.back()] = 1.0;
            }

            mdpStateMap[stateList.back()] = stateIndex;
            stateIndex++;
        }
    }

    // add terminal state where we will self loop
    stateMap["x"] = "-1";
    stateMap["y"] = "-1";
    State terminalState(stateMap);
    stateList.push_back(terminalState);
    mdpStateMap[stateList.back()] = stateIndex;

    // make action mapping
    for(unsigned i=0; i < actionList.size(); i++){
        mdpActionMap[actionList[i]] = i;
    }

    State s;
    std::string a;
    State sNext;
    int xDelta, yDelta;
    int xNew, yNew;
    int xOld, yOld;
    float successProb = 0.8;
    float totalProb;
    float actionCost = -1.0;
    std::vector<std::vector<std::vector<float>>> T(nStates, std::vector<std::vector<float>>(nActions, std::vector<float>(nStates, 0.0)));
    std::vector<std::vector<float>> R(nStates, std::vector<float>(nActions));

    // assign transition probabilities to 3D vector
    for(unsigned sInd = 0; sInd < stateList.size(); sInd++){
        s = stateList.at(sInd);
        xOld = std::stoi(s.getValue("x"));
        yOld = std::stoi(s.getValue("y"));

        for(unsigned aInd=0; aInd < actionList.size(); aInd++){
            a = actionList[aInd];
            totalProb = 0.0;

            if(a == "up"){
                xDelta = 0;
                yDelta = 1;
            }else if(a == "right"){
                xDelta = 1;
                yDelta = 0;
            }else if(a == "down"){
                xDelta = 0;
                yDelta = -1;
            }else if(a == "left"){
                xDelta = -1;
                yDelta = 0;
            }

            xNew = xOld + xDelta;
            yNew = yOld + yDelta;

            // at the terminal state just self loop
            if(xOld == -1 && yOld == -1){
                T[mdpStateMap[s]][mdpActionMap[a]][mdpStateMap[s]] = 1.0;
                totalProb += 1.0;

            // if this action transitions out of range leave probabilities zero
            // so that it is disabled.
            }else if(xNew > xMax || yNew > yMax || xNew < 0 || yNew < 0){
                continue;

            // at the goal state transition to the terminal state
            }else if(xOld == xGoal && yOld == yGoal){
                T[mdpStateMap[s]][mdpActionMap[a]][mdpStateMap[terminalState]] = 1.0;
                totalProb += 1.0;



            // valid actions transition to corresponding state with success prob
            }else if(xNew <= xMax && yNew <=yMax && xNew >= 0 && yNew >= 0){
                stateMap["x"] = std::to_string(xNew);
                stateMap["y"] = std::to_string(yNew);
                sNext = State(stateMap);

                T[mdpStateMap[s]][mdpActionMap[a]][mdpStateMap[sNext]] = successProb;
                totalProb += successProb;
            }

            // with remaining probability the agent stays in the same state
            T[mdpStateMap[s]][mdpActionMap[a]][mdpStateMap[s]] += 1.0 - totalProb;

            // assign rewards/costs
            if(xOld == -1 && yOld == -1){
                R[mdpStateMap[s]][mdpActionMap[a]] = 0.0;
            }else{

                // add an additional cost penalty of 2 at some states
                if((xOld == 1 && yOld == 0) || (xOld == 1 && yOld == 2)){
                    R[mdpStateMap[s]][mdpActionMap[a]] = actionCost - penalty;
                }else{
                    R[mdpStateMap[s]][mdpActionMap[a]] = actionCost;
                }
            }
        }
    }

    return std::make_shared<MDP>(initProbs, T, R, mdpStateMap, mdpActionMap);
}


/* Makes a single MDP corresponding to navigation with uncertain success
probabilities for the navigation actions. This is mean to loosely correspond
to shared autonomy with unknown teleoperator competence.

There are two ways of navigating to the goal location: either by navigating
through the narrow corridor, or navigating through the wide corridor.
The arguments provided give the probabilities of successfully navigating
through each. The number of states that are needed to pass through
to get to the goal in each of the narrow and wide corridors are provided.

Every action to try to move to the next state incurs a reward of -1. A
failed action attempt results in a self-loop.
*/
std::shared_ptr<MDP> makeIndoorNavMDP(float wideSuccess, float narrowSuccess){
    std::vector<std::string> actionList{"narrow", "wide", "loop"};
    std::unordered_map<std::string, std::string> stateMap;
    std::unordered_map<State, float, StateHash> initProbs;
    std::vector<State> stateList;
    float actionCost = 1.0;

    // generate states for initial and final states
    stateMap["name"] = "start";
    stateList.push_back(State(stateMap));
    initProbs[State(stateMap)] = 1.0;
    stateMap["name"] = "end";
    stateList.push_back(State(stateMap));

    // generate states that pass through the narrow corridor
    int narrowStates = 1;
    for(int i = 0; i < narrowStates; i++){
        stateMap["name"] = "narrow" + std::to_string(i);
        stateList.push_back(State(stateMap));
    }

    // generate states that pass through the wide corridor
    int wideStates = 3;
    for(int i = 0; i < wideStates; i++){
        stateMap["name"] = "wide" + std::to_string(i);
        stateList.push_back(State(stateMap));
    }

    // generate state mapping
    int stateIndex = 0;
    std::unordered_map<State, int, StateHash> mdpStateMap;
    for(auto s : stateList){
        mdpStateMap[s] = stateIndex;
        stateIndex++;
    }

    // generate action mapping
    int actionIndex = 0;
    std::unordered_map<std::string, int> mdpActionMap;
    for(auto act : actionList){
        mdpActionMap[act] = actionIndex;
        actionIndex++;
    }

    // assign transition probabilities to 3D vector
    int nActions = actionList.size();
    int nStates = stateList.size();
    std::vector<std::vector<std::vector<float>>> T(nStates, std::vector<std::vector<float>>(nActions, std::vector<float>(nStates, 0.0)));
    std::vector<std::vector<float>> R(nStates, std::vector<float>(nActions));

    // terminal state only has loop action active
    stateMap["name"] = "end";
    State finalState(stateMap);
    T[mdpStateMap[finalState]][mdpActionMap["loop"]][mdpStateMap[finalState]] = 1.0;

    // add actions along the narrow corridor.
    State initState;
    State nextState;
    float success;
    int n;
    for(auto act : actionList){
        if(act == "loop"){
            continue;
        }else if(act == "wide"){
            success = wideSuccess;
            n = wideStates;
        }else if(act == "narrow"){
            success = narrowSuccess;
            n = narrowStates;
        }

        // add the transition at the initial state
        stateMap["name"] = "start";
        initState = State(stateMap);
        stateMap["name"] = act + std::to_string(0);
        nextState = State(stateMap);
        T[mdpStateMap[initState]][mdpActionMap[act]][mdpStateMap[nextState]] = success;
        T[mdpStateMap[initState]][mdpActionMap[act]][mdpStateMap[initState]] = 1.0 - success;
        R[mdpStateMap[initState]][mdpActionMap[act]] = -actionCost;

        // add the transitions through the corridors
        for(int i = 0; i < n; i++){
            stateMap["name"] = act + std::to_string(i);
            initState = State(stateMap);

            if(i == n - 1){
                stateMap["name"] = "end";
                nextState = State(stateMap);
            }else{
                stateMap["name"] = act + std::to_string(i + 1);
                nextState = State(stateMap);
            }

            T[mdpStateMap[initState]][mdpActionMap[act]][mdpStateMap[nextState]] = success;
            T[mdpStateMap[initState]][mdpActionMap[act]][mdpStateMap[initState]] = 1.0 - success;
            R[mdpStateMap[initState]][mdpActionMap[act]] = -actionCost;
        }
    }

    return std::make_shared<MDP>(initProbs, T, R, mdpStateMap, mdpActionMap);
}

/* This function generates the medical decision making domain from the paper
Robust and Adaptive Planning Under Model Uncertainty, Sharma et al.

Note that in this version the number of successors has been reduced as problems
with a large number of successors are especially difficult.

Args:
    days: the number of days in the horizon over which to apply treatments.
*/
std::shared_ptr<MDP> makeMedicalMDP(int days, int seed){

    // seed random number generator so that same problem produced deterministically.
    std::default_random_engine generator;
    generator.seed(seed);
    std::mt19937 rng;
    rng.seed(seed);
    std::srand(seed);

    std::unordered_map<State, float, StateHash> initProbs;

    // there are three differenct actions
    std::vector<std::string> actionList{"A", "B", "C"};
    actionList.push_back("end");

    // make action mapping
    std::unordered_map<std::string, int> mdpActionMap;
    for(unsigned i=0; i < actionList.size(); i++){
        mdpActionMap[actionList[i]] = i;
    }

    // generate state mapping
    std::unordered_map<std::string, std::string> stateMap;
    std::vector<State> stateList;
    std::unordered_map<State, int, StateHash> mdpStateMap;
    int stateIndex = 0;
    int initHealth = 5;
    int maxHealth = 19;

    for(int day = 0; day < days; day++){
        for(int health = 0; health <= maxHealth; health++){

            // note that state feature values must be strings
            stateMap["t"] = std::to_string(day);
            stateMap["health"] = std::to_string(health);
            stateList.push_back(State(stateMap));

            // set initial state
            if(day == 0 && health == initHealth){
                initProbs[stateList.back()] = 1.0;
            }

            mdpStateMap[stateList.back()] = stateIndex;
            stateIndex++;
        }
    }

    // add terminal state where we will self loop
    stateMap["t"] = std::to_string(days);
    stateMap["health"] = "-1";
    State terminalState(stateMap);
    stateList.push_back(terminalState);
    mdpStateMap[stateList.back()] = stateIndex;

    int nStates = stateList.size();
    int nActions = actionList.size();
    std::vector<std::vector<std::vector<float>>> T(nStates, std::vector<std::vector<float>>(nActions, std::vector<float>(nStates, 0.0)));

    // generate the transition probability distribution for each action for this sample
    std::vector<int> deltas{-5, -1, 0, 1, 2};
    std::unordered_map<std::string, std::vector<float>> dists;

    // In this version the actions are fixed: A being most stable and C being least
    // float mu = 0.0;
    // float sig = 0.02;
    // std::vector<float> A_dist{0.0, 0.0, 1.0, 0.0, 0.0};
    // std::vector<float> B_dist{0.0, 0.0, 0.0, 1.0, 0.0};
    // std::vector<float> C_dist{0.0, 0.0, 0.0, 0.0, 1.0};
    // dists["A"] = A_dist;
    // dists["B"] = B_dist;
    // dists["C"] = C_dist;

    // add noise to initial dist for this sample
    // float mu = 0.0;
    // float sig = 0.02;
    // float A_factor = 1.0;
    // float B_factor = 10.0;
    // float C_factor = 20.0;
    // std::normal_distribution<double> distribution(mu, sig);
    // for(unsigned int i = 0; i < A_dist.size(); i++){
    //     float number = fabs(distribution(generator));
    //     for(std::string act : actionList){
    //         std::vector<float> dist;
    //         if(act == "A"){
    //             dists[act][i] += A_factor*number;
    //         }else if(act == "B"){
    //             dists[act][i] += B_factor*number;
    //         }else if(act == "C"){
    //             dists[act][i] += C_factor*number;
    //         }
    //     }
    // }

    // unlike above here i've changed it so that the order of which action is
    // the high variance one changes with sample.
    float mu = 0.0;
    float sig = 0.02;
    float factor_weak = 1.0;
    float factor_moderate = 15.0;
    float factor_strong = 25.0;

    std::vector<float> weak{0.0, 0.0, 1.0, 0.0, 0.0};
    std::vector<float> moderate{0.0, 0.0, 0.0, 1.0, 0.0};
    std::vector<float> strong{0.0, 0.0, 0.0, 0.0, 1.0};

    // shuffle the order of which actions correspond to which response profile
    std::vector<std::string> responses{"weak", "moderate", "strong"};
    std::random_shuffle(responses.begin(), responses.end());
    std::string resp;

    int j = 0;
    for(auto act : actionList){
        if(act == "end"){
            continue;
        }

        resp = responses.at(j);
        if(resp == "weak"){
            dists[act] = weak;
        }else if(resp == "moderate"){
            dists[act] = moderate;
        }else if(resp == "strong"){
            dists[act] = strong;
        }
        j++;
    }

    std::normal_distribution<double> distribution(mu, sig);
    for(unsigned int i = 0; i < deltas.size(); i++){
        float number = fabs(distribution(generator));

        int j = 0;
        for(auto act : actionList){
            if(act == "end"){
                continue;
            }

            std::string resp = responses.at(j);
            if(resp == "weak"){
                dists[act][i] += factor_weak*number;
            }else if(resp == "moderate"){
                dists[act][i] += factor_moderate*number;
            }else if(resp == "strong"){
                dists[act][i] += factor_strong*number;
            }
            j++;
        }
    }

    for(auto pair : dists){
        float sum = std::accumulate(pair.second.begin(), pair.second.end(), 0.0);
        for(unsigned int i = 0; i < pair.second.size(); i++){
            dists[pair.first][i] /= sum;
        }
    }

    // for(unsigned int j = 0; j < actionList.size(); j++){
    //     std::string act = actionList[j];
    //     std::vector<float> dist = dists[act];
    //     std::cout << "Action: " << act << ", : ";
    //     for(auto num : dist){
    //         std::cout << num << ", ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // std::cout << std::endl;

    for(int health = 0; health <= maxHealth; health++){

        // add transition for final day to terminal state
        stateMap["t"] = std::to_string(days - 1);
        stateMap["health"] = std::to_string(health);
        State finalDayState(stateMap);
        T[mdpStateMap.at(finalDayState)][mdpActionMap["end"]][mdpStateMap.at(terminalState)] = 1.0;

        // if the health is zero the health remains zero
        if(health == 0){
            for(int day = 0; day < days-1; day++){
                stateMap["t"] = std::to_string(day);
                stateMap["health"] = std::to_string(health);
                State currentState(stateMap);

                stateMap["t"] = std::to_string(day + 1);
                stateMap["health"] = std::to_string(health);
                State nextState(stateMap);
                T[mdpStateMap.at(currentState)][mdpActionMap["end"]][mdpStateMap.at(nextState)] = 1.0;
            }
            continue;
        }

        // get valid successor health values from this state
        std::vector<int> successorHealth;
        std::vector<int> successorDeltas;
        for(int d : deltas){
            int healthVal = health+d;
            if(healthVal >= 0 && healthVal <= maxHealth){
                successorHealth.push_back(healthVal);
            }else if(healthVal < 0){
                successorHealth.push_back(0);
            }else if(healthVal > maxHealth){
                successorHealth.push_back(maxHealth);
            }
        }

        // for this health level generate random transition values for each treatment
        for(std::string act : actionList){
            if(act == "end"){
                continue;
            }

            int numSuccessors = successorHealth.size();
            for(int day = 0; day < days-1; day++){
                stateMap["t"] = std::to_string(day);
                stateMap["health"] = std::to_string(health);
                State currentState(stateMap);

                for(int i = 0; i < numSuccessors; i++){
                    int nextHealth = successorHealth[i];
                    stateMap["t"] = std::to_string(day + 1);
                    stateMap["health"] = std::to_string(nextHealth);
                    State nextState(stateMap);
                    T[mdpStateMap.at(currentState)][mdpActionMap[act]][mdpStateMap.at(nextState)] += dists[act][i];
                }
            }
        }
    }

    // add self loop at the terminal state
    T[mdpStateMap.at(terminalState)][mdpActionMap["end"]][mdpStateMap.at(terminalState)] = 1.0;

    // add the reward here
    std::vector<std::vector<float>> R(nStates, std::vector<float>(nActions));
    for(int health = 0; health <= maxHealth; health++){
        State state;

        stateMap["t"] = std::to_string(days - 1);
        stateMap["health"] = std::to_string(health);
        state = State(stateMap);
        R[mdpStateMap[state]][mdpActionMap["end"]] = (float)health;
    }

    return std::make_shared<MDP>(initProbs, T, R, mdpStateMap, mdpActionMap);
}

std::shared_ptr<Belief> getMarsRoverBelief(std::shared_ptr<MDP> marsRoverMDP, int horizon, float initCount, bool fullyTied){
    std::unordered_map<std::string, float> priorPseudoStateCounts;
    priorPseudoStateCounts["straight"] = initCount;
    priorPseudoStateCounts["veer_left"] = initCount;
    priorPseudoStateCounts["veer_right"] = initCount;

    std::vector<std::string> actionList{"u", "d", "r", "l", "end"};
    std::vector<std::string> pseudoStates{"straight", "veer_left", "veer_right"};
    std::unordered_map<std::string, std::shared_ptr<TiedDirichletDistribution>> dirichletDistributions;

    for(std::string act : actionList){
        std::shared_ptr<TiedDirichletDistribution> dist = std::make_shared<TiedDirichletDistribution>(priorPseudoStateCounts);
        dirichletDistributions[act] = dist;
    }

    typedef std::unordered_map<std::string, State> successorMapping;
    typedef std::unordered_map<State, std::unordered_map<std::string, successorMapping>, StateHash> pseudoMap;
    std::shared_ptr<pseudoMap> pseudoStateMapping = std::make_shared<pseudoMap>();

    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = std::to_string(horizon);
    stateMap["x"] = "-1";
    stateMap["y"] = "-1";
    State terminalState(stateMap);
    std::vector<std::vector<std::string>> mat = getRoverMatrix();

    for(State s : marsRoverMDP->enumerateStates()){
        std::unordered_map<std::string, successorMapping> actionMap;
        int x = std::stoi(s.getValue("x"));
        int y = std::stoi(s.getValue("y"));
        int t = std::stoi(s.getValue("t"));

        for(std::string action : marsRoverMDP->getEnabledActions(s)){
            successorMapping map;

            int dx;
            int dy;
            int dx_left;
            int dy_left;
            int dx_right;
            int dy_right;
            if(action == "u"){
                dx = 0;
                dy = -1;
                dx_left = -1;
                dy_left = -1;
                dx_right = 1;
                dy_right = -1;
            }else if(action == "r"){
                dx = 1;
                dy = 0;
                dx_left = 1;
                dy_left = -1;
                dx_right = 1;
                dy_right = 1;
            }else if(action == "d"){
                dx = 0;
                dy = 1;
                dx_left = 1;
                dy_left = 1;
                dx_right = -1;
                dy_right = 1;
            }else if(action == "l"){
                dx = -1;
                dy = 0;
                dx_left = -1;
                dy_left = 1;
                dx_right = -1;
                dy_right = -1;
            }

            int nextX_straight = x + dx;
            int nextY_straight = y + dy;

            int nextX_left = x + dx_left;
            int nextY_left = y + dy_left;

            int nextX_right = x + dx_right;
            int nextY_right = y + dy_right;


            for(std::string outcome : pseudoStates){

                // at the terminal state it self loops
                if(t == horizon && x == -1 && y == -1){
                    map[outcome] = terminalState;
                    continue;
                }

                // at final state or at goal
                if(t == horizon-1 || mat[y][x] == "G" || mat[y][x] == "X"){
                    map[outcome] = terminalState;
                    continue;
                }

                int newX;
                int newY;
                if(outcome == "straight"){
                    newX = nextX_straight;
                    newY = nextY_straight;
                }else if(outcome == "veer_left"){
                    newX = nextX_left;
                    newY = nextY_left;
                }else if(outcome == "veer_right"){
                    newX = nextX_right;
                    newY = nextY_right;
                }

                // enforce states stay within grid
                stateMap["t"] = std::to_string(t + 1);
                if(newX >= (int)mat[0].size()){
                    newX = mat[0].size() - 1;
                }
                if(newX < 0){
                    newX = 0;
                }
                if(newY >= (int)mat.size()){
                    newY = mat.size() - 1;
                }
                if(newY < 0){
                    newY = 0;
                }

                stateMap["x"] = std::to_string(newX);
                stateMap["y"] = std::to_string(newY);
                State sNext(stateMap);
                map[outcome] = sNext;
            }

            actionMap[action] = map;
        }
        (*pseudoStateMapping)[s] = actionMap;
    }

    if(fullyTied){
        TiedDirichletDistribution dist(priorPseudoStateCounts);
        return std::make_shared<FullyTiedDirichletBelief>(marsRoverMDP, dist, pseudoStateMapping);
    }else{
        return std::make_shared<TiedDirichletBelief>(marsRoverMDP, dirichletDistributions, pseudoStateMapping);
    }
}

std::vector<std::vector<std::string>> getRoverMatrix(){
    // std::vector<std::vector<std::string>> mat {
    //             {".",".",".",".",".",".",".",".",".",".",".","."},
    //             {".","X",".",".",".",".","G",".",".",".",".","."},
    //             {".",".",".",".",".",".",".",".",".",".",".","."},
    //             {"X",".",".",".",".",".",".","X",".","X",".","."},
    //             {"X",".",".",".","X",".",".",".",".",".",".","."},
    //             {".",".",".",".",".","X",".","X",".",".",".","."},
    //             {".","X",".",".",".",".",".","X",".",".","X","."},
    //             {".",".",".",".",".",".",".",".",".",".",".","X"},
    //             {".","X",".",".",".","X",".",".",".",".","X","."},
    //             {".",".",".",".",".",".",".","X",".",".",".","."},
    //             {"X",".","X",".",".",".",".",".",".",".",".","."},
    //             {".",".","X",".",".",".",".",".",".",".",".","X"},
    //             {"X",".",".",".",".",".",".",".",".",".","X","."},
    //             {".",".",".","X",".",".","S",".",".","X",".","."}
    // };

    std::vector<std::vector<std::string>> mat {
                {".",".",".",".",".","X","X","X"},
                {"G","G",".",".",".",".","X","X"},
                {"G","G",".",".",".",".",".","X"},
                {".",".","X",".",".",".",".","."},
                {".",".",".",".",".",".",".","."},
                {"X",".",".",".",".",".",".","."},
                {"X",".",".",".","X",".",".","."},
                {"X",".",".","X",".",".",".","."},
                {".",".",".",".",".",".",".","X"},
                {".","X",".","S",".",".",".","X"}
    };
    return mat;
};

std::shared_ptr<MDP> marsRoverMDP(int horizon){
    std::vector<std::vector<std::string>> mat = getRoverMatrix();

    // there are four different actions
    std::vector<std::string> actionList{"u", "d", "r", "l", "end"};
    std::unordered_map<State, float, StateHash> initProbs;

    // make action mapping
    std::unordered_map<std::string, int> mdpActionMap;
    for(unsigned i=0; i < actionList.size(); i++){
        mdpActionMap[actionList[i]] = i;
    }

    // generate state mapping
    std::unordered_map<std::string, std::string> stateMap;
    std::vector<State> stateList;
    std::unordered_map<State, int, StateHash> mdpStateMap;
    int stateIndex = 0;

    float probStraight = 0.8;
    float probLeft = 0.1;
    float probRight = 0.1;
    int goalX = 0;
    int goalY = 0;

    for(int t = 0; t < horizon; t++){
        for(unsigned int x = 0; x < mat[0].size(); x++){
            for(unsigned int y = 0; y < mat.size(); y++){
                stateMap["t"] = std::to_string(t);
                stateMap["x"] = std::to_string(x);
                stateMap["y"] = std::to_string(y);
                stateList.push_back(State(stateMap));

                // set initial state
                if(t == 0 && mat[y][x] == "S"){
                    initProbs[stateList.back()] = 1.0;
                }

                if(mat[y][x] == "G" && (x >= goalX || y >= goalY)){
                    goalX = x;
                    goalY = y;
                }

                mdpStateMap[stateList.back()] = stateIndex;
                stateIndex++;
            }
        }
    }

    // add terminal state where we will self loop
    stateMap["t"] = std::to_string(horizon);
    stateMap["x"] = "-1";
    stateMap["y"] = "-1";
    State terminalState(stateMap);
    stateList.push_back(terminalState);
    mdpStateMap[stateList.back()] = stateIndex;


    int nStates = stateList.size();
    int nActions = actionList.size();
    std::unordered_map<State, std::unordered_map<std::string, std::unordered_map<State, float, StateHash>>, StateHash> transitionProbs;
    std::vector<std::vector<std::vector<float>>> T(nStates, std::vector<std::vector<float>>(nActions, std::vector<float>(nStates, 0.0)));

    for(auto state : stateList){
        int t;
        int x;
        int y;
        t = std::stoi(state.getValue("t"));
        x = std::stoi(state.getValue("x"));
        y = std::stoi(state.getValue("y"));

        // if we are in the terminal state we simply add a self loop
        std::unordered_map<State, float, StateHash> successorProbs;
        if(t == horizon && x == -1 && y == -1){
            T[mdpStateMap.at(state)][mdpActionMap["end"]][mdpStateMap.at(state)] = 1.0;

            successorProbs[state] = 1.0;
            std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
            stateTransitionProbs["end"] = successorProbs;
            transitionProbs[state] = stateTransitionProbs;
            continue;
        }

        // if we are at the final stage, the goal state  or crater go to terminal state
        if(t == horizon-1 || mat[y][x] == "G" || mat[y][x] == "X"){
            T[mdpStateMap.at(state)][mdpActionMap["end"]][mdpStateMap.at(terminalState)] = 1.0;

            successorProbs[terminalState] = 1.0;
            std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
            stateTransitionProbs["end"] = successorProbs;
            transitionProbs[state] = stateTransitionProbs;
            continue;
        }


        for(auto action : actionList){
            if(action == "end"){
                continue;
            }

            int dx;
            int dy;
            int dx_left;
            int dy_left;
            int dx_right;
            int dy_right;
            if(action == "u"){
                dx = 0;
                dy = -1;
                dx_left = -1;
                dy_left = -1;
                dx_right = 1;
                dy_right = -1;
            }else if(action == "r"){
                dx = 1;
                dy = 0;
                dx_left = 1;
                dy_left = -1;
                dx_right = 1;
                dy_right = 1;
            }else if(action == "d"){
                dx = 0;
                dy = 1;
                dx_left = 1;
                dy_left = 1;
                dx_right = -1;
                dy_right = 1;
            }else if(action == "l"){
                dx = -1;
                dy = 0;
                dx_left = -1;
                dy_left = 1;
                dx_right = -1;
                dy_right = -1;
            }

            stateMap["t"] = std::to_string(t + 1);

            int nextX_straight = x + dx;
            int nextY_straight = y + dy;

            int nextX_left = x + dx_left;
            int nextY_left = y + dy_left;

            int nextX_right = x + dx_right;
            int nextY_right = y + dy_right;

            std::vector<std::string> outcomes{"straight", "veer_left", "veer_right"};
            float totalProb = 0.0;
            std::unordered_map<State, float, StateHash> successorProbs;
            for(auto outcome : outcomes){
                float prob;
                int newX;
                int newY;
                if(outcome == "straight"){
                    prob = probStraight;
                    newX = nextX_straight;
                    newY = nextY_straight;
                }else if(outcome == "veer_left"){
                    prob = probLeft;
                    newX = nextX_left;
                    newY = nextY_left;
                }else if(outcome == "veer_right"){
                    prob = probRight;
                    newX = nextX_right;
                    newY = nextY_right;
                }

                if(newX >= 0 && newX < (int)mat[0].size()
                    && newY >= 0 && newY < (int)mat.size()){
                        stateMap["x"] = std::to_string(newX);
                        stateMap["y"] = std::to_string(newY);
                        successorProbs[State(stateMap)] = prob;
                        totalProb += prob;

                        T[mdpStateMap.at(state)][mdpActionMap[action]][mdpStateMap.at(State(stateMap))] = prob;
                }
            }

            for(auto kv : successorProbs){
                successorProbs[kv.first] = kv.second/totalProb;
                T[mdpStateMap.at(state)][mdpActionMap[action]][mdpStateMap.at(kv.first)] = kv.second/totalProb;
            }

            // if the state is not in hash table
            if (transitionProbs.find(state) != transitionProbs.end()){

                // if state exists in hash table but not action action
                if(transitionProbs.at(state).find(action) == transitionProbs.at(state).end()){
                    transitionProbs.at(state)[action] = successorProbs;
                }

            // if neither state nor action exist in hash table
            }else{
                std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
                stateTransitionProbs[action] = successorProbs;
                transitionProbs[state] = stateTransitionProbs;
            }
        }
    }

    // add the reward here
    std::vector<std::vector<float>> R(nStates, std::vector<float>(nActions, 0.0));
    float rewardPerStep = -2.0;
    float goalReward = 150.0;
    float nonGoalReward = 100.0;
    float distancePenalty = -5.0;
    float craterPenalty = -50.0;
    for(auto state : stateList){
        int t;
        int x;
        int y;
        t = std::stoi(state.getValue("t"));
        x = std::stoi(state.getValue("x"));
        y = std::stoi(state.getValue("y"));

        // skip if terminal state
        if(t == horizon && x == -1 && y == -1){
            continue;
        }

        // reaching the goal incurrs reward goal
        if(mat[y][x] == "G"){
            R[mdpStateMap.at(state)][mdpActionMap["end"]] = goalReward;
            continue;
        }

        // if we are at the final stage or the crater get reward based on dist.
        if(t == horizon-1 || mat[y][x] == "X"){
            float pen =  distancePenalty * std::sqrt(std::pow((goalX - x), 2.0) + std::pow((goalY - y), 2.0));
            float adjustedReward = nonGoalReward + pen;
            if(mat[y][x] == "X"){
                adjustedReward += craterPenalty;
            }
            R[mdpStateMap.at(state)][mdpActionMap["end"]] = adjustedReward;
            continue;
        }

        for(auto action : actionList){
            if(action == "end"){
                continue;
            }
            R[mdpStateMap[state]][mdpActionMap[action]] = rewardPerStep;
        }
    }


    bool check = true;
    return std::make_shared<MDP>(initProbs, T, R, mdpStateMap, mdpActionMap, check, transitionProbs);
}


/* This MDP is to simulate an inventory control problem. At each stage,
the agent decides how much inventory to purchase. At each stage the agent
incurs a cost according to how much demand is sold and the storage.

Arguments:
    stages: the time horizon.
    maxInventory: the maximum inventory that can be held by the agent.
*/
std::shared_ptr<MDP> makeInventoryMDP(int stages, int maxInventory){
    std::unordered_map<State, float, StateHash> initProbs;
    std::vector<std::string> actionList;

    for(int i = 0; i < maxInventory + 1; i++){
        actionList.push_back(std::to_string(i));
    }
    actionList.push_back("end");

    // generate state mapping
    std::unordered_map<std::string, std::string> stateMap;
    std::vector<State> stateList;
    std::unordered_map<State, int, StateHash> mdpStateMap;
    int stateIndex = 0;
    int initInventory = 0;

    // with some probability there is no demand. Otherwise demand is uniform
    // between 1 and maxInventory
    float probNoDemand = 0.3;

    // the full time horizon is going to be double the number of decision stages.
    // this is because the MDP class only implements state-action costs
    // so another state has to be added to handle the random demand.
    for(int t = 0; t < 2*stages + 1; t++){
        for(int inventory = 0; inventory < maxInventory + 1; inventory++){
            for(int demand = 0; demand < inventory + 1; demand++){

                // note that state feature values must be strings
                stateMap["t"] = std::to_string(t);
                stateMap["inventory"] = std::to_string(inventory);
                stateMap["demand"] = std::to_string(demand);
                stateList.push_back(State(stateMap));

                // set initial state to (0, 0)
                if(t == 0 && demand == 0 && (inventory == initInventory)){
                    initProbs[stateList.back()] = 1.0;
                }

                mdpStateMap[stateList.back()] = stateIndex;
                stateIndex++;
            }
        }
    }

    // add terminal state where we will self loop
    stateMap["t"] = std::to_string(2*stages + 1);
    stateMap["inventory"] = "-1";
    stateMap["demand"] = "-1";
    State terminalState(stateMap);
    stateList.push_back(terminalState);
    mdpStateMap[stateList.back()] = stateIndex;

    // make action mapping
    std::unordered_map<std::string, int> mdpActionMap;
    for(unsigned i=0; i < actionList.size(); i++){
        mdpActionMap[actionList[i]] = i;
    }

    // generate the transition probability matrix
    int nStates = stateList.size();
    int nActions = actionList.size();

    std::unordered_map<State, std::unordered_map<std::string, std::unordered_map<State, float, StateHash>>, StateHash> transitionProbs;
    std::vector<std::vector<std::vector<float>>> T;

    for(auto state : stateList){
        int inventory;
        int t;
        int demand;
        inventory = std::stoi(state.getValue("inventory"));
        t = std::stoi(state.getValue("t"));
        demand = std::stoi(state.getValue("demand"));

        std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;

        // if we are at the final stage the only valid action is the end
        // episode which sends us to the terminal state
        if(t == stages*2){
            std::unordered_map<State, float, StateHash> successorProbs;
            successorProbs[terminalState] = 1.0;
            stateTransitionProbs["end"] = successorProbs;
            transitionProbs[state] = stateTransitionProbs;
            continue;
        }

        // if we are in the terminal state we simply add a self loop
        if(inventory == -1 && t == 2*stages + 1){
            std::unordered_map<State, float, StateHash> successorProbs;
            successorProbs[terminalState] = 1.0;
            stateTransitionProbs["end"] = successorProbs;
            transitionProbs[state] = stateTransitionProbs;
            continue;
        }

        // at even time steps the agent places a purchase request
        if(t % 2 == 0){
            for(auto action : actionList){
                std::unordered_map<State, float, StateHash> successorProbs;


                if(action == "end"){
                    continue;
                }

                // the maximum action would max out the inventory
                if(std::stoi(action) > maxInventory - inventory){
                    continue;
                }

                int nextInventory = inventory + std::stoi(action);
                float prob_sum = 0.0;

                for(int nextDemand = 0; nextDemand < nextInventory + 1; nextDemand++){
                    stateMap["t"] = std::to_string(t + 1);
                    stateMap["inventory"] = std::to_string(nextInventory);
                    stateMap["demand"] = std::to_string(nextDemand);
                    successorProbs[State(stateMap)] = 0.0;
                }

                for(int nextDemand = 0; nextDemand < nextInventory + 1; nextDemand++){
                    stateMap["t"] = std::to_string(t + 1);
                    stateMap["inventory"] = std::to_string(nextInventory);
                    stateMap["demand"] = std::to_string(nextDemand);

                    float prob;
                    if(nextInventory == 0){
                        prob = 1.0;
                        successorProbs[State(stateMap)] += prob;
                        prob_sum += prob;
                    }else{
                        if(nextDemand == 0){
                            prob = probNoDemand;
                        }else if(nextDemand < nextInventory){
                            prob = (1 - probNoDemand)/maxInventory;
                        }
                        if(nextDemand == nextInventory){
                            prob = (maxInventory - nextInventory + 1)*(1 - probNoDemand)/maxInventory;
                        }
                        successorProbs[State(stateMap)] += prob;
                        prob_sum += prob;
                    }
                }

                stateTransitionProbs[action] = successorProbs;
            }
            transitionProbs[state] = stateTransitionProbs;
        }

        // at odd time steps the only action is "0" and the demand is satisfied
        if(t % 2 != 0){
            int nextInventory = inventory - demand;

            stateMap["t"] = std::to_string(t + 1);
            stateMap["inventory"] = std::to_string(nextInventory);
            stateMap["demand"] = "0";

            std::unordered_map<State, float, StateHash> successorProbs;
            successorProbs[State(stateMap)] = 1.0;
            stateTransitionProbs["0"] = successorProbs;
            transitionProbs[state] = stateTransitionProbs;
        }
    }

    // params for cost function
    float purchaseCost = 2.0;
    float holdingCost =  1.0;
    float revenue = 10.0;

    // add the cost here
    std::vector<std::vector<float>> R(nStates, std::vector<float>(nActions));
    for(auto state : stateList){
        int t;
        int inventory;
        int demand;
        t = std::stoi(state.getValue("t"));
        inventory = std::stoi(state.getValue("inventory"));
        demand = std::stoi(state.getValue("demand"));

        for(auto action : actionList){
            if(action == "end"){
                continue;
            }

            // cost
            if(t % 2 == 0 && t < stages*2){
                R[mdpStateMap[state]][mdpActionMap[action]] = inventory * holdingCost + std::stoi(action) * purchaseCost;
            }

            if(t % 2 != 0){
                R[mdpStateMap[state]][mdpActionMap["0"]] = - revenue * demand + (revenue - purchaseCost) * maxInventory;
            }
        }

        if(t == stages * 2){
            R[mdpStateMap[state]][mdpActionMap["end"]] = inventory * holdingCost;
        }
    }

    bool check = false;
    return std::make_shared<MDP>(initProbs, T, R, mdpStateMap, mdpActionMap, check, transitionProbs);
}

/* This MDP is to simulate an inventory control problem. At each stage,
the agent decides how much inventory to purchase. At each stage the agent
incurs a cost according to how much demand is sold and the storage.

Arguments:
    stages: the time horizon.
    maxInventory: the maximum inventory that can be held by the agent.
*/
std::shared_ptr<MDP> makeInventoryRandomWalkMDP(int stages, int maxInventory){
    std::unordered_map<State, float, StateHash> initProbs;
    std::vector<std::string> actionList;

    for(int i = 0; i < maxInventory + 1; i++){
        actionList.push_back(std::to_string(i));
    }
    actionList.push_back("end");

    // generate state mapping
    std::unordered_map<std::string, std::string> stateMap;
    std::vector<State> stateList;
    std::unordered_map<State, int, StateHash> mdpStateMap;
    int stateIndex = 0;
    int initInventory = 0;
    int initDemand = 10;

    // the full time horizon is going to be double the number of decision stages.
    // this is because the MDP class only implements state-action costs
    // so another state has to be added to handle the random demand.
    for(int t = 0; t < 2*stages + 1; t++){
        for(int inventory = 0; inventory < maxInventory + 1; inventory++){
            for(int demand = 0; demand < maxInventory + 1; demand++){

                // note that state feature values must be strings
                stateMap["t"] = std::to_string(t);
                stateMap["inventory"] = std::to_string(inventory);
                stateMap["demand"] = std::to_string(demand);
                stateList.push_back(State(stateMap));

                // set initial state to (0, 0)
                if(t == 0 && demand == initDemand && (inventory == initInventory)){
                    initProbs[stateList.back()] = 1.0;
                }

                mdpStateMap[stateList.back()] = stateIndex;
                stateIndex++;
            }
        }
    }

    // add terminal state where we will self loop
    stateMap["t"] = std::to_string(2*stages + 1);
    stateMap["inventory"] = "-1";
    stateMap["demand"] = "-1";
    State terminalState(stateMap);
    stateList.push_back(terminalState);
    mdpStateMap[stateList.back()] = stateIndex;

    // make action mapping
    std::unordered_map<std::string, int> mdpActionMap;
    for(unsigned i=0; i < actionList.size(); i++){
        mdpActionMap[actionList[i]] = i;
    }

    // generate the transition probability matrix
    int nStates = stateList.size();
    int nActions = actionList.size();

    std::unordered_map<State, std::unordered_map<std::string, std::unordered_map<State, float, StateHash>>, StateHash> transitionProbs;
    std::vector<std::vector<std::vector<float>>> T;

    for(auto state : stateList){
        int inventory;
        int t;
        int demand;
        inventory = std::stoi(state.getValue("inventory"));
        t = std::stoi(state.getValue("t"));
        demand = std::stoi(state.getValue("demand"));

        std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;

        // if we are at the final stage the only valid action is the end
        // episode which sends us to the terminal state
        if(t == stages*2){
            std::unordered_map<State, float, StateHash> successorProbs;
            successorProbs[terminalState] = 1.0;
            stateTransitionProbs["end"] = successorProbs;
            transitionProbs[state] = stateTransitionProbs;
            continue;
        }

        // if we are in the terminal state we simply add a self loop
        if(inventory == -1 && t == 2*stages + 1){
            std::unordered_map<State, float, StateHash> successorProbs;
            successorProbs[terminalState] = 1.0;
            stateTransitionProbs["end"] = successorProbs;
            transitionProbs[state] = stateTransitionProbs;
            continue;
        }

        // at even time steps the agent places a purchase request
        if(t % 2 == 0){
            for(auto action : actionList){
                std::unordered_map<State, float, StateHash> successorProbs;

                if(action == "end"){
                    continue;
                }

                // the maximum action would max out the inventory
                if(std::stoi(action) > maxInventory - inventory){
                    continue;
                }

                int nextInventory = inventory + std::stoi(action);
                int maxDemandChange = 5;

                for(int nextDemand = demand - maxDemandChange; nextDemand < demand + maxDemandChange + 1; nextDemand++){
                    if(nextDemand >= 0 && nextDemand <= maxInventory){
                        stateMap["t"] = std::to_string(t + 1);
                        stateMap["inventory"] = std::to_string(nextInventory);
                        stateMap["demand"] = std::to_string(nextDemand);
                        successorProbs[State(stateMap)] = 0.0;
                    }
                }

                for(int nextDemand = demand - maxDemandChange; nextDemand < demand + maxDemandChange + 1; nextDemand++){
                    int nextDemandClipped;
                    if(nextDemand < 0){
                        nextDemandClipped = 0;
                    }else if(nextDemand > maxInventory){
                        nextDemandClipped = maxInventory;
                    }else{
                        nextDemandClipped = nextDemand;
                    }

                    stateMap["t"] = std::to_string(t + 1);
                    stateMap["inventory"] = std::to_string(nextInventory);
                    stateMap["demand"] = std::to_string(nextDemandClipped);
                    successorProbs[State(stateMap)] += 1.0/(maxDemandChange * 2.0 + 1.0);
                }

                stateTransitionProbs[action] = successorProbs;
            }
            transitionProbs[state] = stateTransitionProbs;
        }

        // at odd time steps the only action is "0" and the demand is satisfied
        if(t % 2 != 0){
            int nextInventory = inventory - demand;
            if(nextInventory < 0){
                nextInventory = 0;
            }

            stateMap["t"] = std::to_string(t + 1);
            stateMap["inventory"] = std::to_string(nextInventory);
            stateMap["demand"] = std::to_string(demand);

            std::unordered_map<State, float, StateHash> successorProbs;
            successorProbs[State(stateMap)] = 1.0;
            stateTransitionProbs["0"] = successorProbs;
            transitionProbs[state] = stateTransitionProbs;
        }
    }

    // params for cost function
    float purchaseCost = 1.0;
    float holdingCost =  1.0;
    float revenue = 3.0;

    // add the cost here
    std::vector<std::vector<float>> R(nStates, std::vector<float>(nActions));
    for(auto state : stateList){
        int t;
        int inventory;
        int demand;
        t = std::stoi(state.getValue("t"));
        inventory = std::stoi(state.getValue("inventory"));
        demand = std::stoi(state.getValue("demand"));

        for(auto action : actionList){
            if(action == "end"){
                continue;
            }

            // cost
            if(t % 2 == 0 && t < stages*2){
                R[mdpStateMap[state]][mdpActionMap[action]] = inventory * holdingCost + std::stoi(action) * purchaseCost;
            }

            if(t % 2 != 0){
                int numSold;
                if(inventory >= demand){
                    numSold = demand;
                }else{
                    numSold = inventory;
                }
                float constant =  (revenue - purchaseCost) * maxInventory; // add constant to ensure that cost is positive
                R[mdpStateMap[state]][mdpActionMap["0"]] = - revenue * numSold + constant;
            }
        }

        if(t == stages * 2){
            R[mdpStateMap[state]][mdpActionMap["end"]] = inventory * holdingCost;
        }
    }

    bool check = false;
    return std::make_shared<MDP>(initProbs, T, R, mdpStateMap, mdpActionMap, check, transitionProbs);
}

/* This MDP is to simulate a multistage betting game. The agent starts out
with a fixed amount of money. The agent is able to bet an amount of money
on the next game, up to a maximum of 5, and starts out with 10 money. After
betting an amount of money, if the agent wins, they earn that much money. If
they lose, they lose the amount money that was bet.

Arguments:
    stages: the number of times that the game will be repeated.
    successProb: the probability that the agent wins each game.
*/
std::shared_ptr<MDP> makeBettingMDP(float successProb, int stages, const std::vector<int>& bets){
    std::unordered_map<State, float, StateHash> initProbs;
    std::vector<std::string> actionList;

    int maxBet = 0;
    if(bets.size() > 0){
        for(int i : bets){
            actionList.push_back(std::to_string(i));
            if(i > maxBet){
                maxBet = i;
            }
        }
    }else{
        maxBet = 5;
        for(int i = 0; i < maxBet + 1; i++){
            actionList.push_back(std::to_string(i));
        }
    }
    actionList.push_back("end");

    // generate state mapping
    std::unordered_map<std::string, std::string> stateMap;
    std::vector<State> stateList;
    std::unordered_map<State, int, StateHash> mdpStateMap;
    int stateIndex = 0;
    int initMoney = 10;
    int maxMoney = initMoney + maxBet * stages;
    for(int stage = 0; stage < stages + 1; stage++){
        for(int money = 0; money < maxMoney + 1; money++){

            // note that state feature values must be strings
            stateMap["t"] = std::to_string(stage);
            stateMap["money"] = std::to_string(money);
            stateList.push_back(State(stateMap));

            // set initial state to (0, 0)
            if(stage == 0 && money == initMoney){
                initProbs[stateList.back()] = 1.0;
            }

            mdpStateMap[stateList.back()] = stateIndex;
            stateIndex++;
        }
    }

    // add terminal state where we will self loop
    stateMap["t"] = std::to_string(stages + 1);
    stateMap["money"] = "-1";
    State terminalState(stateMap);
    stateList.push_back(terminalState);
    mdpStateMap[stateList.back()] = stateIndex;

    // make action mapping
    std::unordered_map<std::string, int> mdpActionMap;
    for(unsigned i=0; i < actionList.size(); i++){
        mdpActionMap[actionList[i]] = i;
    }

    // generate the transition probability matrix
    int nStates = stateList.size();
    int nActions = actionList.size();

    std::vector<std::vector<std::vector<float>>> T(nStates, std::vector<std::vector<float>>(nActions, std::vector<float>(nStates, 0.0)));
    for(auto state : stateList){
        int money;
        int stage;
        money = std::stoi(state.getValue("money"));
        stage = std::stoi(state.getValue("t"));

        // if we are at the final stage the only valid action is the end
        // episode which sends us to the terminal state
        if(stage == stages){
            T[mdpStateMap.at(state)][mdpActionMap["end"]][mdpStateMap.at(terminalState)] = 1.0;
            continue;
        }

        // if we are in the terminal state we simply add a self loop
        if(money == -1 && stage == stages + 1){
            T[mdpStateMap.at(state)][mdpActionMap["end"]][mdpStateMap.at(state)] = 1.0;
            continue;
        }

        for(auto action : actionList){
            if(action == "end"){
                continue;
            }

            // if the bet amount is greater than amount of money do not activate
            // action
            if(std::stoi(action) > money){
                continue;
            }

            int loseAmount;
            int winAmount;
            stateMap["t"] = std::to_string(stage + 1);

            // success transitions to state with gained money at next stage
            winAmount = money + std::stoi(action);
            if(winAmount > maxMoney){
                winAmount = maxMoney;
            }
            stateMap["money"] = std::to_string(winAmount);
            T[mdpStateMap.at(state)][mdpActionMap[action]][mdpStateMap.at(State(stateMap))] = successProb;

            // failure transitions to state with gained money at next stage
            loseAmount = money - std::stoi(action);
            if(loseAmount < 0){
                loseAmount = 0;
            }
            stateMap["money"] = std::to_string(loseAmount);
            T[mdpStateMap.at(state)][mdpActionMap[action]][mdpStateMap.at(State(stateMap))] += 1.0 - successProb;
        }
    }

    // add the reward here
    std::vector<std::vector<float>> R(nStates, std::vector<float>(nActions));
    for(int money = 0; money < maxMoney + 1; money++){
        State state;

        stateMap["t"] = std::to_string(stages);
        stateMap["money"] = std::to_string(money);
        state = State(stateMap);
        R[mdpStateMap[state]][mdpActionMap["end"]] = (float)money;
    }

    return std::make_shared<MDP>(initProbs, T, R, mdpStateMap, mdpActionMap);
}

std::tuple<std::unordered_map<std::string, int>, std::unordered_map<std::string, int>, std::unordered_map<std::string, int>, std::unordered_map<std::string, int>> getTrafficOutcomes(){
    std::unordered_map<std::string, int> outcomesVeryQuiet;
     outcomesVeryQuiet["medium"] = 7;
     outcomesVeryQuiet["slow"] = 8;

     std::unordered_map<std::string, int> outcomesQuiet;
     outcomesQuiet["fast"] = 4;
     outcomesQuiet["medium"] = 5;
     outcomesQuiet["slow"] = 11;

     std::unordered_map<std::string, int> outcomesAverage;
     outcomesAverage["fast"] = 2;
     outcomesAverage["medium"] = 4;
     outcomesAverage["slow"] = 13;

     std::unordered_map<std::string, int> outcomesBusy;
     outcomesBusy["fast"] = 1;
     outcomesBusy["medium"] = 2;
     outcomesBusy["slow"] = 18;
 return std::make_tuple(outcomesVeryQuiet, outcomesQuiet, outcomesAverage, outcomesBusy);
}

// MDP to simulate traffic navigation where the cost of of the time required to
// reach a goal state. In the non-refactored version, an additional state factor
// accumulates the total time elapsed over the episode and the cost at the end
// of the episode is based on that additional state factor.
// for the refactored version, the agent transitions to a state representing
// how much time has just been incurred and must execute the "cost" action
// which incurrs the cost. This lowers the state space, but doubles the horizon
// that must be looked over.

std::shared_ptr<MDP> trafficMDP(int horizon){

    // actions for movements along roads of varying busyness.
    std::vector<std::string> actionList{"quiet_up", "quiet_down", "quiet_left", "quiet_right", "very_quiet_up", "very_quiet_down", "very_quiet_left", "very_quiet_right", "busy_up", "busy_down", "busy_left", "busy_right", "average_left", "average_down", "average_right", "average_up", "end"};
    std::unordered_map<State, float, StateHash> initProbs;

    // make action mapping
    std::unordered_map<std::string, int> mdpActionMap;
    for(unsigned i=0; i < actionList.size(); i++){
        mdpActionMap[actionList[i]] = i;
    }

    // outcomes for how fast transition is done
    std::tuple<std::unordered_map<std::string, int>, std::unordered_map<std::string, int>, std::unordered_map<std::string, int>, std::unordered_map<std::string, int>> outcomes = getTrafficOutcomes();
    std::unordered_map<std::string, int> outcomesVeryQuiet = std::get<0>(outcomes);
    std::unordered_map<std::string, int> outcomesQuiet = std::get<1>(outcomes);
    std::unordered_map<std::string, int> outcomesAverage = std::get<2>(outcomes);
    std::unordered_map<std::string, int> outcomesBusy = std::get<3>(outcomes);

    // find the maximum elapsed time in a transition.
    int max_mins_per_step = 0;
    for(auto kv : outcomesBusy){
        if(kv.second > max_mins_per_step){
            max_mins_per_step = kv.second;
        }
    }

    int maxMins = max_mins_per_step * horizon;

    // generate state mapping
    std::unordered_map<std::string, std::string> stateMap;
    std::unordered_map<State, int, StateHash> mdpStateMap;
    std::vector<State> stateList;
    int stateIndex = 0;
    int goalX = 3;
    int goalY = 4;
    int maxX = 3;
    int maxY = 4;

    for(int t = 0; t < horizon; t++){
        int maxMinsThisStep = max_mins_per_step*t;
        for(int min = 0; min <= maxMinsThisStep; min++)
            for(int x = 0; x <= maxX; x++){
                for(int y = 0; y <= maxY; y++){
                    stateMap["t"] = std::to_string(t); // steps through the MDP, one per step
                    stateMap["min"] = std::to_string(min); // minutes elapsed used for final cost.
                    stateMap["x"] = std::to_string(x);
                    stateMap["y"] = std::to_string(y);
                    stateList.push_back(State(stateMap));

                    // set initial state
                    if(t == 0 && min == 0 && x == 1 && y == 0){
                        initProbs[stateList.back()] = 1.0;
                    }

                    mdpStateMap[stateList.back()] = stateIndex;
                    stateIndex++;
                }
            }
    }

    // add terminal state where we will self loop
    stateMap["t"] = std::to_string(horizon);
    stateMap["min"] = "-1";
    stateMap["x"] = "-1";
    stateMap["y"] = "-1";
    State terminalState(stateMap);
    stateList.push_back(terminalState);
    mdpStateMap[stateList.back()] = stateIndex;

    int nStates = stateList.size();
    int nActions = actionList.size();

    std::unordered_map<State, std::unordered_map<std::string, std::unordered_map<State, float, StateHash>>, StateHash> transitionProbs;
    std::vector<std::vector<std::vector<float>>> T;

    for(auto state : stateList){
        int t;
        int min;
        int x;
        int y;
        t = std::stoi(state.getValue("t"));
        min = std::stoi(state.getValue("min"));
        x = std::stoi(state.getValue("x"));
        y = std::stoi(state.getValue("y"));

        // self loop at the terminal state
        std::unordered_map<State, float, StateHash> successorProbs;
        if(t == horizon && x == -1 && y == -1){
            successorProbs[state] = 1.0;
            std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
            stateTransitionProbs["end"] = successorProbs;
            transitionProbs[state] = stateTransitionProbs;
            continue;
        }

        // if we are at the final stage or the goal state go to terminal state
        if(t == horizon-1 || (x == goalX && y == goalY)){
            successorProbs[terminalState] = 1.0;
            std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
            stateTransitionProbs["end"] = successorProbs;
            transitionProbs[state] = stateTransitionProbs;
            continue;
        }

        std::vector<std::string> enabledActions;
        if(x == 0){
            if(y < maxY){
                enabledActions.push_back("busy_up");
            }
        }

        if(x == 1){
            if(y < maxY){
                enabledActions.push_back("average_up");
            }
        }

        if(x == 2){
            if(y < maxY){
                enabledActions.push_back("quiet_up");
            }
        }

        if(x == 3){
            if(y < maxY){
                enabledActions.push_back("very_quiet_up");
            }
        }

        std::string leftAction;
        std::string rightAction;
        if(y == 0){
            leftAction = "very_quiet_left";
            rightAction = "very_quiet_right";
        }else if(y == 1 || y == 2 || y == 3){
            if(x == 0){
                rightAction = "busy_right";
            }else if(x == 1 || x == 2){
                rightAction = "quiet_right";
            }

            if(x == 1){
                leftAction = "busy_left";
            }else if(x == 2 || x == 3){
                leftAction = "quiet_left";
            }


        }else if(y == 4 || y == 5){
            if(x == 0){
                rightAction = "busy_right";
            }else if(x == 1){
                rightAction = "average_right";
            }else if(x == 2){
                rightAction = "quiet_right";
            }

            if(x == 1){
                leftAction = "busy_left";
            }else if(x == 2){
                leftAction = "average_left";
            }else if(x == 3){
                leftAction = "quiet_left";
            }
        }

        if(x == 0){
            enabledActions.push_back(rightAction);
        }

        if(x == 1 || x == 2){
            enabledActions.push_back(leftAction);
            enabledActions.push_back(rightAction);
        }

        if(x == 3){
            enabledActions.push_back(leftAction);
        }


        for(auto action : enabledActions){

            int dx;
            int dy;
            if(action == "quiet_up" || action == "average_up" || action == "busy_up" || action == "very_quiet_up"){
                dx = 0;
                dy = 1;
            }else if(action == "quiet_right" || action == "average_right" || action == "busy_right" || action == "very_quiet_right"){
                dx = 1;
                dy = 0;
            }
            else if(action == "quiet_left" || action == "average_left" || action == "busy_left" || action == "very_quiet_left"){
                dx = -1;
                dy = 0;
            }else if(action == "quiet_down" || action == "average_down" || action == "busy_down" || action == "very_quiet_down"){
                dx = 0;
                dy = -1;
            }

            std::unordered_map<std::string, int> outcomes;
            if(action == "very_quiet_up" || action == "very_quiet_right" || action == "very_quiet_left" || action == "very_quiet_down"){
                outcomes = outcomesVeryQuiet;
            }

            if(action == "quiet_up" || action == "quiet_right" || action == "quiet_left" || action == "quiet_down"){
                outcomes = outcomesQuiet;
            }

            if(action == "average_up" || action == "average_left" || action == "average_right" || action == "average_down"){
                outcomes = outcomesAverage;
            }

            if(action == "busy_up" || action == "busy_left" || action == "busy_right" || action == "busy_down"){
                outcomes = outcomesBusy;
            }

            std::unordered_map<State, float, StateHash> successorProbs;
            for(auto outcome : outcomes){
                stateMap["t"] = std::to_string(t + 1);
                stateMap["x"] = std::to_string(x + dx);
                stateMap["y"] = std::to_string(y + dy);

                int minNext = min + outcome.second;
                if(minNext > maxMins){
                    minNext = maxMins;
                }
                stateMap["min"] = std::to_string(minNext);
                successorProbs[State(stateMap)] = 1.0/outcomes.size();
            }

            // if the state is in hash table
            if (transitionProbs.find(state) != transitionProbs.end()){

                // if state exists in hash table but not action action
                if(transitionProbs.at(state).find(action) == transitionProbs.at(state).end()){
                    transitionProbs.at(state)[action] = successorProbs;
                }

            // if neither state nor action exist in hash table
            }else{
                std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
                stateTransitionProbs[action] = successorProbs;
                transitionProbs[state] = stateTransitionProbs;
            }
        }

    }

    // add the reward here
    std::vector<std::vector<float>> R(nStates, std::vector<float>(nActions, 0.0));
    float goalBonus = 80.0;
    for(auto state : stateList){
        int t;
        int x;
        int y;
        int min;
        t = std::stoi(state.getValue("t"));
        min = std::stoi(state.getValue("min"));
        x = std::stoi(state.getValue("x"));
        y = std::stoi(state.getValue("y"));

        // if we are at the final stage or the goal state the reward is max min minus the final time
        if((x == goalX && y == goalY)){
            R[mdpStateMap[state]][mdpActionMap["end"]] = goalBonus - float(min);
        }else if(t == horizon-1){
            R[mdpStateMap[state]][mdpActionMap["end"]] = -float(min);
        }
    }

    bool check = false;
    return std::make_shared<MDP>(initProbs, T, R, mdpStateMap, mdpActionMap, check, transitionProbs);
}

std::shared_ptr<MDP> trafficMDPRefactored(int horizon){

    // actions for movements along roads of varying busyness.
    std::vector<std::string> actionList{"quiet_up", "quiet_down", "quiet_left", "quiet_right", "very_quiet_up", "very_quiet_down", "very_quiet_left", "very_quiet_right", "busy_up", "busy_down", "busy_left", "busy_right", "average_left", "average_down", "average_right", "average_up", "cost", "end"};
    std::unordered_map<State, float, StateHash> initProbs;

    // make action mapping
    std::unordered_map<std::string, int> mdpActionMap;
    for(unsigned i=0; i < actionList.size(); i++){
        mdpActionMap[actionList[i]] = i;
    }

    // outcomes for how fast transition is done
    std::tuple<std::unordered_map<std::string, int>, std::unordered_map<std::string, int>, std::unordered_map<std::string, int>, std::unordered_map<std::string, int>> outcomes = getTrafficOutcomes();
    std::unordered_map<std::string, int> outcomesVeryQuiet = std::get<0>(outcomes);
    std::unordered_map<std::string, int> outcomesQuiet = std::get<1>(outcomes);
    std::unordered_map<std::string, int> outcomesAverage = std::get<2>(outcomes);
    std::unordered_map<std::string, int> outcomesBusy = std::get<3>(outcomes);

    // find the maximum elapsed time in a transition.
    int max_mins_per_step = 0;
    for(auto kv : outcomesBusy){
        if(kv.second > max_mins_per_step){
            max_mins_per_step = kv.second;
        }
    }

    int maxMins = max_mins_per_step * horizon;

    // generate state mapping
    std::unordered_map<std::string, std::string> stateMap;
    std::unordered_map<State, int, StateHash> mdpStateMap;
    std::vector<State> stateList;
    int stateIndex = 0;
    int goalX = 3;
    int goalY = 4;
    int maxX = 3;
    int maxY = 4;

    // the horizon is double the original requested horizon because at each
    // step we have a step to incur cost
    for(int t = 0; t < horizon * 2; t++){
        for(int x = 0; x <= maxX; x++){
            for(int y = 0; y <= maxY; y++){
                  std::vector<int> possibleMins{0};
                  if(t % 2 != 0){
                      possibleMins.push_back(outcomesVeryQuiet["medium"]);
                      possibleMins.push_back(outcomesVeryQuiet["slow"]);
                      possibleMins.push_back(outcomesQuiet["fast"]);
                      possibleMins.push_back(outcomesQuiet["medium"]);
                      possibleMins.push_back(outcomesQuiet["slow"]);
                      possibleMins.push_back(outcomesAverage["fast"]);
                      possibleMins.push_back(outcomesAverage["medium"]);
                      possibleMins.push_back(outcomesAverage["slow"]);
                      possibleMins.push_back(outcomesBusy["fast"]);
                      possibleMins.push_back(outcomesBusy["medium"]);
                      possibleMins.push_back(outcomesBusy["slow"]);
                  }

                  for(unsigned int i = 0; i < possibleMins.size(); i++){
                    stateMap["t"] = std::to_string(t); // steps through the MDP, one per step
                    stateMap["min"] = std::to_string(possibleMins[i]); // minutes elapsed used for final cost.
                    stateMap["x"] = std::to_string(x);
                    stateMap["y"] = std::to_string(y);
                    stateList.push_back(State(stateMap));

                    // set initial state
                    if(t == 0 && possibleMins[i] == 0 && x == 1 && y == 0){
                        initProbs[stateList.back()] = 1.0;
                    }

                    mdpStateMap[stateList.back()] = stateIndex;
                    stateIndex++;
                  }
            }
        }
    }

    // add terminal state where we will self loop
    stateMap["t"] = std::to_string(horizon);
    stateMap["min"] = "-1";
    stateMap["x"] = "-1";
    stateMap["y"] = "-1";
    State terminalState(stateMap);
    stateList.push_back(terminalState);
    mdpStateMap[stateList.back()] = stateIndex;

    int nStates = stateList.size();
    int nActions = actionList.size();

    std::unordered_map<State, std::unordered_map<std::string, std::unordered_map<State, float, StateHash>>, StateHash> transitionProbs;
    std::vector<std::vector<std::vector<float>>> T;

    for(auto state : stateList){
        int t;
        int min;
        int x;
        int y;
        t = std::stoi(state.getValue("t"));
        min = std::stoi(state.getValue("min"));
        x = std::stoi(state.getValue("x"));
        y = std::stoi(state.getValue("y"));

        // in refactored one executing the "cost" action incurs cost and returns to state
        std::unordered_map<State, float, StateHash> successorProbs;
        if(min != 0){
            stateMap["t"] = std::to_string(t + 1);
            stateMap["x"] = std::to_string(x);
            stateMap["y"] = std::to_string(y);
            stateMap["min"] = "0";
            State sNext(stateMap);
            successorProbs[sNext] = 1.0;
            std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
            stateTransitionProbs["cost"] = successorProbs;
            transitionProbs[state] = stateTransitionProbs;
            continue;
        }

        // self loop at the terminal state
        if(t == horizon && x == -1 && y == -1){
            successorProbs[state] = 1.0;
            std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
            stateTransitionProbs["end"] = successorProbs;
            transitionProbs[state] = stateTransitionProbs;
            continue;
        }

        // if we are at the final stage or the goal state go to terminal state
        if(t == horizon-1 || (x == goalX && y == goalY)){
            successorProbs[terminalState] = 1.0;
            std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
            stateTransitionProbs["end"] = successorProbs;
            transitionProbs[state] = stateTransitionProbs;
            continue;
        }

        std::vector<std::string> enabledActions;
        if(x == 0){
            if(y < maxY){
                enabledActions.push_back("busy_up");
            }
        }

        if(x == 1){
            if(y < maxY){
                enabledActions.push_back("average_up");
            }
        }

        if(x == 2){
            if(y < maxY){
                enabledActions.push_back("quiet_up");
            }
        }

        if(x == 3){
            if(y < maxY){
                enabledActions.push_back("very_quiet_up");
            }
        }

        std::string leftAction;
        std::string rightAction;
        if(y == 0){
            leftAction = "very_quiet_left";
            rightAction = "very_quiet_right";
        }else if(y == 1 || y == 2 || y == 3){
            if(x == 0){
                rightAction = "busy_right";
            }else if(x == 1 || x == 2){
                rightAction = "quiet_right";
            }

            if(x == 1){
                leftAction = "busy_left";
            }else if(x == 2 || x == 3){
                leftAction = "quiet_left";
            }


        }else if(y == 4 || y == 5){
            if(x == 0){
                rightAction = "busy_right";
            }else if(x == 1){
                rightAction = "average_right";
            }else if(x == 2){
                rightAction = "quiet_right";
            }

            if(x == 1){
                leftAction = "busy_left";
            }else if(x == 2){
                leftAction = "average_left";
            }else if(x == 3){
                leftAction = "quiet_left";
            }
        }

        if(x == 0){
            enabledActions.push_back(rightAction);
        }

        if(x == 1 || x == 2){
            enabledActions.push_back(leftAction);
            enabledActions.push_back(rightAction);
        }

        if(x == 3){
            enabledActions.push_back(leftAction);
        }


        for(auto action : enabledActions){

            int dx;
            int dy;
            if(action == "quiet_up" || action == "average_up" || action == "busy_up" || action == "very_quiet_up"){
                dx = 0;
                dy = 1;
            }else if(action == "quiet_right" || action == "average_right" || action == "busy_right" || action == "very_quiet_right"){
                dx = 1;
                dy = 0;
            }
            else if(action == "quiet_left" || action == "average_left" || action == "busy_left" || action == "very_quiet_left"){
                dx = -1;
                dy = 0;
            }else if(action == "quiet_down" || action == "average_down" || action == "busy_down" || action == "very_quiet_down"){
                dx = 0;
                dy = -1;
            }

            std::unordered_map<std::string, int> outcomes;
            if(action == "very_quiet_up" || action == "very_quiet_right" || action == "very_quiet_left" || action == "very_quiet_down"){
                outcomes = outcomesVeryQuiet;
            }

            if(action == "quiet_up" || action == "quiet_right" || action == "quiet_left" || action == "quiet_down"){
                outcomes = outcomesQuiet;
            }

            if(action == "average_up" || action == "average_left" || action == "average_right" || action == "average_down"){
                outcomes = outcomesAverage;
            }

            if(action == "busy_up" || action == "busy_left" || action == "busy_right" || action == "busy_down"){
                outcomes = outcomesBusy;
            }

            std::unordered_map<State, float, StateHash> successorProbs;
            for(auto outcome : outcomes){
                stateMap["t"] = std::to_string(t + 1);
                stateMap["x"] = std::to_string(x + dx);
                stateMap["y"] = std::to_string(y + dy);

                int minNext = min + outcome.second;
                if(minNext > maxMins){
                    minNext = maxMins;
                }
                stateMap["min"] = std::to_string(minNext);
                successorProbs[State(stateMap)] = 1.0/outcomes.size();
            }

            // if the state is in hash table
            if (transitionProbs.find(state) != transitionProbs.end()){

                // if state exists in hash table but not action action
                if(transitionProbs.at(state).find(action) == transitionProbs.at(state).end()){
                    transitionProbs.at(state)[action] = successorProbs;
                }

            // if neither state nor action exist in hash table
            }else{
                std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
                stateTransitionProbs[action] = successorProbs;
                transitionProbs[state] = stateTransitionProbs;
            }
        }

    }

    // add the reward here
    std::vector<std::vector<float>> R(nStates, std::vector<float>(nActions, 0.0));
    float goalBonus = 80.0;
    for(auto state : stateList){
        int t;
        int x;
        int y;
        int min;
        t = std::stoi(state.getValue("t"));
        min = std::stoi(state.getValue("min"));
        x = std::stoi(state.getValue("x"));
        y = std::stoi(state.getValue("y"));

        // if we are at the final stage or the goal state the reward is max min minus the final time
        if((x == goalX && y == goalY && min == 0)){
            R[mdpStateMap[state]][mdpActionMap["end"]] = goalBonus;
        }

        if(min > 0){
            R[mdpStateMap[state]][mdpActionMap["cost"]] = -float(min);
        }
    }

    bool check = false;
    return std::make_shared<MDP>(initProbs, T, R, mdpStateMap, mdpActionMap, check, transitionProbs);
}

std::shared_ptr<TiedDirichletBelief> getTrafficBelief(
    std::shared_ptr<MDP> trafficMDP,
    int horizon,
    std::vector<float> initCounts,
    bool refactored
){
    std::unordered_map<std::string, float> priorPseudoStateCounts;
    priorPseudoStateCounts["fast"] = initCounts[0];
    priorPseudoStateCounts["medium"] = initCounts[1];
    priorPseudoStateCounts["slow"] = initCounts[2];

    std::unordered_map<std::string, float>  veryQuietCounts;
    veryQuietCounts["medium"] = initCounts[1];
    veryQuietCounts["slow"] = initCounts[2];

    std::vector<std::string> actionList{"quiet_up", "quiet_down", "quiet_left", "quiet_right", "very_quiet_up", "very_quiet_down", "very_quiet_left", "very_quiet_right", "busy_up", "busy_down", "busy_left", "busy_right", "average_left", "average_down", "average_right", "average_up", "cost", "end"};

    std::unordered_map<std::string, std::shared_ptr<TiedDirichletDistribution>> dirichletDistributions;
    std::shared_ptr<TiedDirichletDistribution> averageDist = std::make_shared<TiedDirichletDistribution>(priorPseudoStateCounts);
    std::shared_ptr<TiedDirichletDistribution> quietDist = std::make_shared<TiedDirichletDistribution>(priorPseudoStateCounts);
    std::shared_ptr<TiedDirichletDistribution> veryQuietDist = std::make_shared<TiedDirichletDistribution>(veryQuietCounts);
    std::shared_ptr<TiedDirichletDistribution> busyDist = std::make_shared<TiedDirichletDistribution>(priorPseudoStateCounts);
    std::shared_ptr<TiedDirichletDistribution> endDist = std::make_shared<TiedDirichletDistribution>(priorPseudoStateCounts);

    std::tuple<std::unordered_map<std::string, int>, std::unordered_map<std::string, int>, std::unordered_map<std::string, int>, std::unordered_map<std::string, int>> outcomes = getTrafficOutcomes();
    std::unordered_map<std::string, int> outcomesVeryQuiet = std::get<0>(outcomes);
    std::unordered_map<std::string, int> outcomesQuiet = std::get<1>(outcomes);
    std::unordered_map<std::string, int> outcomesAverage = std::get<2>(outcomes);
    std::unordered_map<std::string, int> outcomesBusy = std::get<3>(outcomes);

    int max_mins_per_step = 0;
    for(auto kv : outcomesBusy){
        if(kv.second > max_mins_per_step){
            max_mins_per_step = kv.second;
        }
    }
    int maxMins = max_mins_per_step * horizon;

    for(std::string act : actionList){
        if(act == "quiet_up" || act == "quiet_right" || act == "quiet_left" || act == "quiet_down"){
            dirichletDistributions[act] = quietDist;
        }

        if(act == "very_quiet_up" || act == "very_quiet_right" || act == "very_quiet_left" || act == "very_quiet_down"){
            dirichletDistributions[act] = veryQuietDist;
        }

        if(act == "busy_up" || act == "busy_left" || act == "busy_right" || act == "busy_down"){
            dirichletDistributions[act] = busyDist;
        }

        if(act == "average_up" || act == "average_right" || act == "average_left" || act == "average_down"){
            dirichletDistributions[act] = averageDist;
        }

        if(act == "end" || act == "cost"){
            dirichletDistributions[act] = endDist;
        }
    }

    typedef std::unordered_map<std::string, State> successorMapping;
    typedef std::unordered_map<State, std::unordered_map<std::string, successorMapping>, StateHash> pseudoMap;
    std::shared_ptr<pseudoMap> pseudoStateMapping = std::make_shared<pseudoMap>();

    // add terminal state where we will self loop
    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = std::to_string(horizon);
    stateMap["min"] = "-1";
    stateMap["x"] = "-1";
    stateMap["y"] = "-1";
    State terminalState(stateMap);

    for(State s : trafficMDP->enumerateStates()){
        std::unordered_map<std::string, successorMapping> actionMap;
        int t;
        int min;
        int x;
        int y;
        t = std::stoi(s.getValue("t"));
        min = std::stoi(s.getValue("min"));
        x = std::stoi(s.getValue("x"));
        y = std::stoi(s.getValue("y"));

        for(std::string action : trafficMDP->getEnabledActions(s)){
            int dx;
            int dy;
            if(action == "quiet_up" || action == "average_up" || action == "busy_up" || action == "very_quiet_up"){
                dx = 0;
                dy = 1;
            }else if(action == "quiet_right" || action == "average_right" || action == "busy_right" || action == "very_quiet_right"){
                dx = 1;
                dy = 0;
            }else if(action == "quiet_left" || action == "average_left" || action == "busy_left" || action == "very_quiet_left"){
                dx = -1;
                dy = 0;
            }else if(action == "quiet_down" || action == "average_down" || action == "busy_down" || action == "very_quiet_down"){
                dx = 0;
                dy = -1;
            }

            successorMapping map;
            std::unordered_map<State, float, StateHash> successors = trafficMDP->getTransitionProbs(s, action);

            if(s == terminalState || successors.begin()->first == terminalState){
                map["slow"] = terminalState;
                map["medium"] = terminalState;
                map["fast"] = terminalState;
                actionMap[action] = map;
                continue;
            }

            if(action == "cost"){
                stateMap["t"] = std::to_string(t + 1);
                stateMap["x"] = std::to_string(x);
                stateMap["y"] = std::to_string(y);
                stateMap["min"] = "0";
                State sNext(stateMap);
                map["slow"] = sNext;
                map["medium"] = sNext;
                map["fast"] = sNext;
                actionMap[action] = map;
                continue;
            }

            std::unordered_map<std::string, int> outcomes;
            if(action == "very_quiet_up" || action == "very_quiet_right" || action == "very_quiet_left" || action == "very_quiet_down"){
                outcomes = outcomesVeryQuiet;
            }

            if(action == "quiet_up" || action == "quiet_right" || action == "quiet_left" || action == "quiet_down"){
                outcomes = outcomesQuiet;
            }

            if(action == "average_up" || action == "average_left" || action == "average_right" || action == "average_down"){
                outcomes = outcomesAverage;
            }

            if(action == "busy_up" || action == "busy_left" || action == "busy_right" || action == "busy_down"){
                outcomes = outcomesBusy;
            }

            std::unordered_map<State, float, StateHash> successorProbs;
            for(auto outcome : outcomes){
                stateMap["t"] = std::to_string(t + 1);
                stateMap["x"] = std::to_string(x + dx);
                stateMap["y"] = std::to_string(y + dy);

                int minNext = min + outcome.second;
                if(minNext > maxMins){
                    minNext = maxMins;
                }
                stateMap["min"] = std::to_string(minNext);
                map[outcome.first] = State(stateMap);
            }
            actionMap[action] = map;
        }
        (*pseudoStateMapping)[s] = actionMap;
    }
    return std::make_shared<TiedDirichletBelief>(trafficMDP, dirichletDistributions, pseudoStateMapping);
}

/* Returns tied dirichlet belief for the CVaR betting game */
std::shared_ptr<FullyTiedDirichletBelief> getBettingGameBelief(
    float winCounts,
    float loseCounts,
    int stages,
    std::shared_ptr<MDP> pMDP)
{

    // find the maximum allowed bet
    std::unordered_map<State, float, StateHash> initProbs = pMDP->getInitialState();
    std::vector<std::string> bets = pMDP->getEnabledActions(initProbs.begin()->first);
    int maxBet = 0;
    for(auto st : bets){
        if(std::stoi(st) > maxBet){
            maxBet = std::stoi(st);
        }
    }

    // Note these are set based on the function which builds the betting game MDP
    int initMoney = 10;
    int maxMoney = initMoney + maxBet * stages;

    std::vector<std::string> pseudoStates{"win", "lose"};
    std::unordered_map<std::string, float> priorPseudoStateCounts;
    priorPseudoStateCounts["win"] = winCounts;
    priorPseudoStateCounts["lose"] = loseCounts;

    TiedDirichletDistribution dist(priorPseudoStateCounts);


    std::unordered_map<std::string, std::string> stateMap;
    stateMap["t"] = std::to_string(stages+1);
    stateMap["money"] = "-1";
    State terminalState(stateMap);

    typedef std::unordered_map<std::string, State> successorMapping;
    std::shared_ptr<std::unordered_map<State, std::unordered_map<std::string, successorMapping>, StateHash>> pseudoStateMapping = std::make_shared<std::unordered_map<State, std::unordered_map<std::string, successorMapping>, StateHash>>();
    for(State s : pMDP->enumerateStates()){
        std::unordered_map<std::string, successorMapping> actionMap;
        int stage = std::stoi(s.getValue("t"));
        int money = std::stoi(s.getValue("money"));

        for(std::string act : pMDP->getEnabledActions(s)){
            successorMapping map;

            for(std::string outcome : pseudoStates){

                // if we are at the final stage or the terminals state the
                // successor will be the terminal state
                if(stage == stages || (money == -1 && stage == stages+1)){
                    map[outcome] = terminalState;
                    continue;
                }

                stateMap["t"] = std::to_string(stage + 1);

                if(outcome == "win"){
                    int winAmount = money + std::stoi(act);
                    if(winAmount > maxMoney){
                        winAmount = maxMoney;
                    }
                    stateMap["money"] = std::to_string(winAmount);
                }else{
                    int loseAmount = money - std::stoi(act);
                    if(loseAmount < 0){
                        loseAmount = 0;
                    }
                    stateMap["money"] = std::to_string(loseAmount);
                }

                map[outcome] = State(stateMap);
            }
            actionMap[act] = map;
        }
        (*pseudoStateMapping)[s] = actionMap;
    }
    std::shared_ptr<FullyTiedDirichletBelief> b = std::make_shared<FullyTiedDirichletBelief>(pMDP, dist, pseudoStateMapping);
    return b;
}
