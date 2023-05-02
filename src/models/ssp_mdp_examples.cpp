#include <unordered_map>
#include <iostream>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "catch.h"
#include "state.h"
#include "mdp.h"
#include "utils.h"
#include "value_iteration.h"
#include "fully_tied_dirichlet_belief.h"
#include "tied_dirichlet_belief.h"
#include "mdp_examples.h"

std::mt19937 rng(0);
std::uniform_real_distribution<> uniform(0, 0.1);

std::pair<std::unordered_map<std::string, float>, bool> toTimeProbMap(std::string dictString, bool addNoise){
    dictString.erase(std::remove(dictString.begin(), dictString.end(), '['), dictString.end());
    dictString.erase(std::remove(dictString.begin(), dictString.end(), '{'), dictString.end());
    dictString.erase(std::remove(dictString.begin(), dictString.end(), '}'), dictString.end());
    dictString.erase(std::remove(dictString.begin(), dictString.end(), ']'), dictString.end());
    dictString.erase(std::remove(dictString.begin(), dictString.end(), ' '), dictString.end());

    std::string delimiter = ",";
    size_t pos = 0;
    std::string subString;
    std::vector<std::string> pairs;
    std::unordered_map<std::string, float> timeProbMap;

    // break up each key:pair value separated by comma in dictionary
    while ((pos = dictString.find(delimiter)) != std::string::npos) {
        subString = dictString.substr(0, pos);
        pairs.push_back(subString);
        dictString.erase(0, pos + delimiter.length());
    }
    pairs.push_back(dictString);

    // initialise the probabilities to zero for each time value
    delimiter = ":";
    std::string largestTime = "0.0";
    int ind = 0;

    // initialise the dictionary and find the largest time value
    bool isHighway = false;
    for(std::string pair : pairs){
        pos = pair.find(delimiter);
        std::string key = pair.substr(0, pos);

        // check if this is a highway transition
        if(key == "'ID'"){
            pair.erase(0, pos + delimiter.length());
            if(pair == "'freeway'"){
                isHighway = true;
            }
        }

        if(key == "'ID'" || key == "'start'" || key ==  "'end'"){
            continue;
        }

        // initialise keys in dict
        std::string time = key.substr(0, 7);
        timeProbMap[time] = 0.0;

        // find largest time value
        if(std::stof(time) > std::stof(largestTime)){
            largestTime = time;
        }
        ind++;
    }

    // increment probabilities based on dictionary
    for(std::string pair : pairs){
        pos = pair.find(delimiter);
        std::string key = pair.substr(0, pos);
        if(key == "'ID'" || key == "'start'" || key ==  "'end'"){
            continue;
        }

        std::string time = key.substr(0, 7);
        pair.erase(0, pos + delimiter.length());
        float prob = std::stof(pair);
        if(prob > 0.0){
            timeProbMap[time] += prob;
        }
    }

    // if adding noise add a small random probability to the largest transition time
    if(isHighway && addNoise){
        timeProbMap[largestTime] += uniform(rng);
    }

    // remove any zero probabilities
    float sum = 0.0;
    std::unordered_map<std::string, float> finalTimeProbMap;
    for(auto pair : timeProbMap){
        if(pair.second > 0.0){
            if(!isHighway && pair.second < 0.02){
                continue;
            }

            std::string time = pair.first;
            finalTimeProbMap[time] = pair.second;
            sum += pair.second;
        }
    }

    // renormalise for noise added
    for(auto pair : finalTimeProbMap){
        finalTimeProbMap[pair.first] /= sum;
    }

    return std::make_pair(finalTimeProbMap, isHighway);
}

std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> getStateTransitionProbs(
    State baseState,
    std::string dataset,
    bool includeHighways,
    bool addNoise
){
    std::fstream file;
    file.open(dataset, std::ios::in);
    std::string line, word, temp;
    std::string correctLine;
    std::unordered_map<int, std::string> columnIDMapping;
    std::vector<std::string> ids;
    std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;

    // find the correct line corresponding to the base state
    bool firstLine = true;
    while(std::getline(file, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);
        int ind = 0;

        while (ss >> std::ws) {
            std::string csvElement;

            if (ss.peek() == '"') {
               ss >> std::quoted(csvElement);
               std::string discard;
               std::getline(ss, discard, ',');
            }else{
               std::getline(ss, csvElement, ',');
            }

            if(csvElement == baseState.getValue("id")){
                correctLine = line;
            }

            // also generate a mapping of columns to successor ids
            if(firstLine){
                columnIDMapping[ind] = csvElement;
                ind++;
            }else{
                break;
            }
        }

        firstLine = false;
    }

    // process the line of the adjacency matrix corresponding to the current state
    std::stringstream ss(correctLine);
    int ind = 0;
    while (ss >> std::ws) {
        std::string csvElement;

        if (ss.peek() == '"') {
           ss >> std::quoted(csvElement);
           std::string discard;
           std::getline(ss, discard, ',');
        }else{
           std::getline(ss, csvElement, ',');
        }

        // check if there are successor probabilities for this entry
        if ((csvElement != "-") && (csvElement != baseState.getValue("id")) && csvElement.length() > 0){

            // parse the string to generate a mapping between durations and probabilities
            std::pair<std::unordered_map<std::string, float>, bool> parsed  = toTimeProbMap(csvElement, addNoise);
            std::unordered_map<std::string, float> timeProbMap = parsed.first;
            bool isHighway = parsed.second;

            if(isHighway && !includeHighways){
                continue;
            }

            float probSum = 0.0;
            for(auto pair : timeProbMap){
                probSum += pair.second;
            }
            if(!cmpf(probSum, 1.0)){
                std::cout << "Warning: probability doesn't sum to 1." << std::endl;
                continue;
            }

            std::unordered_map<State, float, StateHash> successorProbs;
            for(auto pair : timeProbMap){
                std::unordered_map<std::string, std::string> stateMap;
                stateMap["id"] = columnIDMapping[ind];
                stateMap["cost"] = pair.first;
                successorProbs[State(stateMap)] = pair.second;
            }
            stateTransitionProbs[columnIDMapping[ind]] = successorProbs;
        }
        ind++;
    }

    return stateTransitionProbs;
}

std::shared_ptr<MDP> realTrafficSSP(
    std::string dataset,
    int startID,
    int goalID,
    bool includeHighways,
    bool addNoise
){
    std::mt19937 rng(0);

    std::fstream file;
    file.open(dataset, std::ios::in);

    std::string line, word, temp;
    std::vector<std::string> row;
    std::vector<std::string> ids;

    while(std::getline(file, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);

        bool first = true;
        while (ss >> std::ws) {
            std::string csvElement;

            if (ss.peek() == '"') {
               ss >> std::quoted(csvElement);
               std::string discard;
               std::getline(ss, discard, ',');
            }else{
               std::getline(ss, csvElement, ',');
            }

            if(first && csvElement.length() > 0){
                ids.push_back(csvElement);
            }
            first = false;
        }
    }

    std::vector<std::string> actionList{"cost", "end"};
    std::vector<State> stateList;
    std::unordered_map<std::string, std::string> stateMap;
    std::unordered_map<State, int, StateHash> mdpStateMap;
    std::unordered_map<std::string, int> mdpActionMap;
    std::unordered_map<State, float, StateHash> initProbs;

    // mapping for base states
    int stateIndex = 0;
    bool startFound = false;
    bool goalFound = false;
    for(std::string id : ids){
        actionList.push_back(id);

        stateMap["id"] = id;
        stateMap["cost"] = "0.0";
        stateList.push_back(State(stateMap));

        if(id == std::to_string(startID)){
            initProbs[stateList.back()] = 1.0;
            startFound = true;
        }

        if(id == std::to_string(goalID)){
            goalFound = true;
        }

        mdpStateMap[stateList.back()] = stateIndex;
        stateIndex++;
    }

    if(!goalFound){
        std::cout << "Could not find goal ID" << std::endl;
        std::exit(0);
    }

    if(!startFound){
        std::cout << "Could not find start ID" << std::endl;
        std::exit(0);
    }

    // mapping for actions
    for(unsigned i=0; i < actionList.size(); i++){
        mdpActionMap[actionList[i]] = i;
    }

    std::unordered_map<State, std::unordered_map<std::string, std::unordered_map<State, float, StateHash>>, StateHash> transitionProbs;
    std::vector<std::vector<std::vector<float>>> T;


    std::vector<State> baseStates = stateList;
    for(auto baseState : baseStates){
        std::string id;
        std::string cost;

        id = baseState.getValue("id");
        cost = baseState.getValue("cost");

        // self loop at the terminal state
        if(id == std::to_string(goalID)){
            std::unordered_map<State, float, StateHash> successorProbs;
            successorProbs[baseState] = 1.0;
            std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
            stateTransitionProbs["end"] = successorProbs;
            transitionProbs[baseState] = stateTransitionProbs;
            continue;
        }

        // get successor states which incurr a time cost
        std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs = getStateTransitionProbs(baseState, dataset, includeHighways, addNoise);

        // for each successor add to list if it isn't already there
        for(auto pair : stateTransitionProbs){
            for(auto pair2 : pair.second){
                State costState = pair2.first;
                if(std::count(stateList.begin(), stateList.end(), costState) == 0){

                    // if cost state has not been processed add to state list
                    stateList.push_back(costState);
                    mdpStateMap[stateList.back()] = stateIndex;
                    stateIndex++;

                    // also add a transition back to base state using cost act
                    std::unordered_map<std::string, std::string> stateMap = costState.getStateMapping();
                    stateMap["cost"] = "0.0";
                    State nextBaseState = State(stateMap);

                    std::unordered_map<State, float, StateHash> afterCostSuccessorProbs;
                    afterCostSuccessorProbs[nextBaseState] = 1.0;
                    std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
                    stateTransitionProbs["cost"] = afterCostSuccessorProbs;
                    transitionProbs[costState] = stateTransitionProbs;
                }
            }
        }

        transitionProbs[baseState] = stateTransitionProbs;
    }

    // add the reward here
    std::vector<std::vector<float>> C(stateList.size(), std::vector<float>(actionList.size(), 0.0));
    for(auto state : stateList){
        float cost;
        cost = std::stof(state.getValue("cost"));
        C[mdpStateMap[state]][mdpActionMap["cost"]] = cost;
    }

    bool check = false;
    return std::make_shared<MDP>(initProbs, T, C, mdpStateMap, mdpActionMap, check, transitionProbs);
}

std::vector<std::vector<float>> getTrafficDurations(){
    std::vector<float> roadType1{0.7, 1.1, 1.3, 1.7, 20.5};
    std::vector<float> roadType2{1.7, 2.6, 2.7, 3.5, 17.1};
    std::vector<float> roadType3{2.9, 3.8, 3.9, 4.6, 15.3};
    std::vector<float> roadType4{4.1, 4.8, 5.3, 6.3, 12.1};
    std::vector<float> roadType5{5.7, 6.0, 6.5, 7.3,  8.8};

    std::vector<std::vector<float>> allDurations{
                                    roadType1,
                                    roadType2,
                                    roadType3,
                                    roadType4,
                                    roadType5
    };
 return allDurations;
}

/* MDP to simulate traffic navigation with uncertain travel durations.

Arguments:
    size: defines the edge length for a size x size grid to navigate.
*/
std::shared_ptr<MDP> trafficSSP(int size){

    // actions for movements along roads of varying busyness.
    std::vector<std::string> actionList{"up", "left", "right", "down", "cost", "end"};
    std::unordered_map<State, float, StateHash> initProbs;

    // make action mapping
    std::unordered_map<std::string, int> mdpActionMap;
    for(unsigned i=0; i < actionList.size(); i++){
        mdpActionMap[actionList[i]] = i;
    }

    // outcomes for how fast transition is done
    std::vector<std::vector<float>> allDurations = getTrafficDurations();
    int numDurationDists = allDurations.size();
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> uni(0, numDurationDists-1);

    // generate state mapping
    std::unordered_map<std::string, std::string> stateMap;
    std::unordered_map<State, int, StateHash> mdpStateMap;
    std::vector<State> stateList;
    int stateIndex = 0;
    int maxX = size - 1;
    int maxY = size - 1;
    int goalX = maxX;
    int goalY = maxY;

    for(int x = 0; x <= maxX; x++){
        for(int y = 0; y <= maxY; y++){
            stateMap["x"] = std::to_string(x);
            stateMap["y"] = std::to_string(y);
            stateMap["cost"] = "0.0";
            stateList.push_back(State(stateMap));

            // set initial state
            if(x == 0 && y == 0){
                initProbs[stateList.back()] = 1.0;
            }

            mdpStateMap[stateList.back()] = stateIndex;
            stateIndex++;
        }
    }

    std::unordered_map<State, std::unordered_map<std::string, std::unordered_map<State, float, StateHash>>, StateHash> transitionProbs;
    std::vector<std::vector<std::vector<float>>> T;

    // base states are those with cost = 0.0 and no cost the be incurred.
    // at other states the agent has to execute the "cost" action to receive
    // the duration cost and then transition to the associated base state
    std::vector<State> baseStates = stateList;
    for(auto baseState : baseStates){
        int x;
        int y;
        std::string cost;
        x = std::stoi(baseState.getValue("x"));
        y = std::stoi(baseState.getValue("y"));
        cost = baseState.getValue("cost");

        // self loop at the terminal state
        if(x == goalX && y == goalY){
            std::unordered_map<State, float, StateHash> successorProbs;
            successorProbs[baseState] = 1.0;
            std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
            stateTransitionProbs["end"] = successorProbs;
            transitionProbs[baseState] = stateTransitionProbs;
            continue;
        }

        std::vector<std::string> enabledActions;
        if(x < maxX){
            enabledActions.push_back("right");
        }

        if(x > 0){
            enabledActions.push_back("left");
        }

        if(y > 0){
            enabledActions.push_back("down");
        }

        if(y < maxY){
            enabledActions.push_back("up");
        }

        for(auto action : enabledActions){
            int dx;
            int dy;
            if(action == "up"){
                dx = 0;
                dy = 1;
            }else if(action == "right"){
                dx = 1;
                dy = 0;
            }else if(action == "left"){
                dx = -1;
                dy = 0;
            }else if(action == "down"){
                dx = 0;
                dy = -1;
            }

            int sampleIndex;
            sampleIndex = uni(rng);
            std::vector<float> durations = allDurations[sampleIndex];

            std::unordered_map<State, float, StateHash> baseStateSuccessorProbs;
            for(auto duration : durations){

                // transition from base state to state representing cost
                stateMap["x"] = std::to_string(x + dx);
                stateMap["y"] = std::to_string(y + dy);
                stateMap["cost"] = std::to_string(duration);
                State sNext(stateMap);
                if(std::count(stateList.begin(), stateList.end(), sNext) == 0){
                   stateList.push_back(sNext);
                   mdpStateMap[stateList.back()] = stateIndex;
                   stateIndex++;
                }
                baseStateSuccessorProbs[sNext] = 1.0/durations.size();

                // add transition from state representing cost back to base state
                stateMap["cost"] = "0.0";
                State sNextAfterCost(stateMap);
                std::unordered_map<State, float, StateHash> afterCostSuccessorProbs;
                afterCostSuccessorProbs[sNextAfterCost] = 1.0;
                std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
                stateTransitionProbs["cost"] = afterCostSuccessorProbs;
                transitionProbs[sNext] = stateTransitionProbs;
            }

            // if the state is in hash table
            if (transitionProbs.find(baseState) != transitionProbs.end()){

                // if state exists in hash table but not action action
                if(transitionProbs.at(baseState).find(action) == transitionProbs.at(baseState).end()){
                    transitionProbs.at(baseState)[action] = baseStateSuccessorProbs;
                }

            // if neither state nor action exist in hash table
            }else{
                std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;
                stateTransitionProbs[action] = baseStateSuccessorProbs;
                transitionProbs[baseState] = stateTransitionProbs;
            }
        }

    }

    // add the reward here
    std::vector<std::vector<float>> C(stateList.size(), std::vector<float>(actionList.size(), 0.0));
    for(auto state : stateList){
        float cost;
        cost = std::stof(state.getValue("cost"));
        C[mdpStateMap[state]][mdpActionMap["cost"]] = cost;
    }

    bool check = false;
    return std::make_shared<MDP>(initProbs, T, C, mdpStateMap, mdpActionMap, check, transitionProbs);
}

std::shared_ptr<MDP> testExample(){
    std::vector<std::string> actionList{"A", "B", "C"};
    std::unordered_map<State, float, StateHash> initProbs;

    // make action mapping
    std::unordered_map<std::string, int> mdpActionMap;
    for(unsigned i=0; i < actionList.size(); i++){
        mdpActionMap[actionList[i]] = i;
    }

    std::unordered_map<std::string, std::string> stateMap;
    std::unordered_map<State, int, StateHash> mdpStateMap;
    std::vector<State> stateList;
    int stateIndex = 0;

    for(int i; i <= 8; i++){
        stateMap["state"] = std::to_string(i);
        stateList.push_back(State(stateMap));

        if(i == 0){
            initProbs[stateList.back()] = 1.0;
        }

        mdpStateMap[stateList.back()] = stateIndex;
        stateIndex++;
    }

    stateMap["state"] = "0";
    State state0(stateMap);
    stateMap["state"] = "1";
    State state1(stateMap);
    stateMap["state"] = "2";
    State state2(stateMap);
    stateMap["state"] = "3";
    State state3(stateMap);
    stateMap["state"] = "4";
    State state4(stateMap);
    stateMap["state"] = "5";
    State state5(stateMap);
    stateMap["state"] = "6";
    State state6(stateMap);
    stateMap["state"] = "7";
    State state7(stateMap);
    stateMap["state"] = "8";
    State state8(stateMap);

    std::unordered_map<State, std::unordered_map<std::string, std::unordered_map<State, float, StateHash>>, StateHash> transitionProbs;
    std::vector<std::vector<std::vector<float>>> T;

    for(auto state : stateList){

        std::unordered_map<std::string, std::unordered_map<State, float, StateHash>> stateTransitionProbs;

        if(state.getValue("state") == "0"){
            std::unordered_map<State, float, StateHash> successorProbsA;
            successorProbsA[state3] = 0.8;
            successorProbsA[state2] = 0.1;
            successorProbsA[state1] = 0.1;
            stateTransitionProbs["A"] = successorProbsA;
        }

        if(state.getValue("state") == "1" || state.getValue("state") == "2"
            || state.getValue("state") == "4" || state.getValue("state") == "5"
            || state.getValue("state") == "6"  || state.getValue("state") == "7"
            || state.getValue("state") == "8"
        )
        {
            std::unordered_map<State, float, StateHash> successorProbsA;
            successorProbsA[state8] = 1.0;
            stateTransitionProbs["A"] = successorProbsA;
        }

        if(state.getValue("state") == "3"){
            std::unordered_map<State, float, StateHash> successorProbsA;
            successorProbsA[state4] = 0.95;
            successorProbsA[state5] = 0.05;

            std::unordered_map<State, float, StateHash> successorProbsB;
            successorProbsB[state6] = 0.3;
            successorProbsB[state7] = 0.7;

            std::unordered_map<State, float, StateHash> successorProbsC;
            successorProbsC[state8] = 1.0;

            stateTransitionProbs["A"] = successorProbsA;
            stateTransitionProbs["B"] = successorProbsB;
            stateTransitionProbs["C"] = successorProbsC;
        }

        transitionProbs[state] = stateTransitionProbs;
    }

    std::vector<std::vector<float>> C(stateList.size(), std::vector<float>(actionList.size(), 0.0));
    for(auto state : stateList){
        for(std::string act : actionList){
            float cost = 0.0;
            if(state.getValue("state") == "1" && act == "A"){
                cost = 10.0;
            }else if(state.getValue("state") == "2" && act == "A"){
                cost = 8.0;
            }else if(state.getValue("state") == "3" && act == "C"){
                cost = 7.0;
            }else if(state.getValue("state") == "5" && act == "A"){
                cost = 20.0;
            }else if(state.getValue("state") == "6" && act == "A"){
                cost = 9.0;
            }else if(state.getValue("state") == "7" && act == "A"){
                cost = 1.0;
            }

            C[mdpStateMap[state]][mdpActionMap[act]] = cost;
        }
    }

    bool check = false;
    return std::make_shared<MDP>(initProbs, T, C, mdpStateMap, mdpActionMap, check, transitionProbs);
}

std::shared_ptr<MDP> makeBettingWithJackpotMDP(int stages){
    std::unordered_map<State, float, StateHash> initProbs;
    std::vector<std::string> actionList;

    int maxMoney = 100;
    int jackpotFactor = 10;
    float winProb = 0.7;
    float jackpotProb = 0.05;
    int maxBet = 5;
    for(int i = 0; i < maxBet + 1; i++){
        actionList.push_back(std::to_string(i));
    }
    actionList.push_back("end");

    // generate state mapping
    std::unordered_map<std::string, std::string> stateMap;
    std::vector<State> stateList;
    std::unordered_map<State, int, StateHash> mdpStateMap;
    int stateIndex = 0;
    int initMoney = 10;

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
            T[mdpStateMap.at(state)][mdpActionMap[action]][mdpStateMap.at(State(stateMap))] = winProb;

            // jackpot
            int jackpotAmount = money + jackpotFactor*std::stoi(action);
            if(jackpotAmount > maxMoney){
                jackpotAmount = maxMoney;
            }
            stateMap["money"] = std::to_string(jackpotAmount);
            T[mdpStateMap.at(state)][mdpActionMap[action]][mdpStateMap.at(State(stateMap))] += jackpotProb;

            // failure transitions to state with gained money at next stage
            loseAmount = money - std::stoi(action);
            if(loseAmount < 0){
                loseAmount = 0;
            }
            stateMap["money"] = std::to_string(loseAmount);
            T[mdpStateMap.at(state)][mdpActionMap[action]][mdpStateMap.at(State(stateMap))] += 1.0 - winProb - jackpotProb;
        }
    }

    // add the cost here
    std::vector<std::vector<float>> R(nStates, std::vector<float>(nActions));
    for(int money = 0; money < maxMoney + 1; money++){
        State state;

        stateMap["t"] = std::to_string(stages);
        stateMap["money"] = std::to_string(money);
        state = State(stateMap);
        R[mdpStateMap[state]][mdpActionMap["end"]] = (float)(maxMoney - money);
    }

    return std::make_shared<MDP>(initProbs, T, R, mdpStateMap, mdpActionMap);
}


// std::vector<std::vector<std::string>> getDSTMatrix(){
//
//     // horizon is 15
//     std::vector<std::vector<std::string>> mat {
//                 {"---","---","---","---","---","---","---","---","---","---","---","---","---","---","---","---","---"},
//                 {"---","---","---","---","---","---","---","---","---","---","---","---","---","---","---","---","---"},
//                 {"---","---","---","---","---","---","---","---","---","---","---","---","---","---","---","---","999"},
//                 {"---","---","---","---","---","---","---","---","---","---","---","---","---","---","---","750","XXX"},
//                 {"SSS","---","---","---","---","---","---","---","---","---","---","---","---","---","---","XXX","XXX"},
//                 {"---","---","---","---","---","---","---","---","---","---","---","---","---","---","---","XXX","XXX"},
//                 {"---","---","---","---","---","---","---","---","---","---","---","---","---","---","590","XXX","XXX"},
//                 {"---","---","---","---","---","---","---","---","---","---","---","---","---","450","XXX","XXX","XXX"},
//                 {"---","---","---","---","---","---","---","---","---","---","---","---","350","XXX","XXX","XXX","XXX"},
//                 {"010","---","---","---","---","---","---","---","---","---","215","270","XXX","XXX","XXX","XXX","XXX"},
//                 {"XXX","---","---","---","---","---","---","---","130","170","XXX","XXX","XXX","XXX","XXX","XXX","XXX"},
//                 {"XXX","015","025","---","---","---","---","100","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX"},
//                 {"XXX","XXX","XXX","030","---","---","075","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX"},
//                 {"XXX","XXX","XXX","XXX","040","055","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX"},
//                 {"XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX"},
//
//     };
//     return mat;
// };

std::vector<std::vector<std::string>> getDSTMatrix(){

    // horizon is 15
    std::vector<std::vector<std::string>> mat {
                {"---","---","---","---","---","---","---","---","---","---","---","---","---","---","---"},
                {"SSS","---","---","---","---","---","---","---","---","---","---","---","---","---","---"},
                {"---","---","---","---","---","---","---","---","---","---","---","---","---","---","---"},
                {"010","---","---","---","---","---","---","---","---","---","---","---","---","---","---"},
                {"XXX","---","---","---","---","---","---","---","---","---","---","---","---","---","---"},
                {"XXX","015","025","---","---","---","---","---","---","---","---","---","---","---","---"},
                {"XXX","XXX","XXX","030","---","---","---","---","---","---","---","---","---","---","---"},
                {"XXX","XXX","XXX","XXX","040","055","---","---","---","---","---","---","---","---","---"},
                {"XXX","XXX","XXX","XXX","XXX","XXX","075","---","---","---","---","---","---","---","---"},
                {"XXX","XXX","XXX","XXX","XXX","XXX","XXX","100","140","---","---","---","---","---","---"},
                {"XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","190","---","---","---","---","---"},
                {"XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","240","300","---","---","---"},
                {"XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","360","---","---"},
                {"XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","430","500"},
                {"XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX","XXX"},

    };
    return mat;
};


std::shared_ptr<MDP> deepSeaTreasureMDP(int horizon){
    std::vector<std::vector<std::string>> mat = getDSTMatrix();

    // there are four different actions
    std::vector<std::string> actionList{"u", "ur", "ul", "d", "dr", "dl", "r", "l", "end"};
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

    float probStraight = 0.6;
    float probLeft = 0.2;
    float probRight = 0.2;

    for(int t = 0; t < horizon; t++){
        for(unsigned int x = 0; x < mat[0].size(); x++){
            for(unsigned int y = 0; y < mat.size(); y++){
                if(mat[y][x] == "XXX"){
                    continue;
                }

                stateMap["t"] = std::to_string(t);
                stateMap["x"] = std::to_string(x);
                stateMap["y"] = std::to_string(y);
                stateList.push_back(State(stateMap));

                // set initial state
                if(t == 0 && mat[y][x] == "SSS"){
                    initProbs[stateList.back()] = 1.0;
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
            continue;
        }

        // if we are at the final stage, the goal state  or crater go to terminal state
        if(t == horizon-1 || (mat[y][x] != "SSS" && mat[y][x] != "---")){
            T[mdpStateMap.at(state)][mdpActionMap["end"]][mdpStateMap.at(terminalState)] = 1.0;
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
            }else if(action == "ur"){
                dx = 1;
                dy = -1;
                dx_left = 0;
                dy_left = -1;
                dx_right = 1;
                dy_right = 0;
            }else if(action == "ul"){
                dx = -1;
                dy = -1;
                dx_left = -1;
                dy_left = 0;
                dx_right = 0;
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
            }else if(action == "dr"){
                dx = 1;
                dy = 1;
                dx_left = 1;
                dy_left = 0;
                dx_right = 0;
                dy_right = 1;
            }else if(action == "dl"){
                dx = -1;
                dy = 1;
                dx_left = 0;
                dy_left = 1;
                dx_right = -1;
                dy_right = 0;
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

                // only transition if within map and not rocks
                if(newX >= 0 && newX < (int)mat[0].size()
                    && newY >= 0 && newY < (int)mat.size() && mat[newY][newX] != "XXX"){
                        stateMap["x"] = std::to_string(newX);
                        stateMap["y"] = std::to_string(newY);
                        successorProbs[State(stateMap)] = prob;
                        totalProb += prob;
                        T[mdpStateMap.at(state)][mdpActionMap[action]][mdpStateMap.at(State(stateMap))] = prob;
                }
            }

            if(totalProb > 0.0){
                for(auto kv : successorProbs){
                    successorProbs[kv.first] = kv.second/totalProb;
                    T[mdpStateMap.at(state)][mdpActionMap[action]][mdpStateMap.at(kv.first)] = kv.second/totalProb;
                }
            }
        }
    }

    // add the reward here
    std::vector<std::vector<float>> R(nStates, std::vector<float>(nActions, 0.0));
    float costPerStep = 5.0;
    float maxTreasure = 500.0;
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

        // reaching a treasure gets maxTreasure - treasure cost.
        if(mat[y][x] != "SSS" && mat[y][x] != "---"){
            R[mdpStateMap.at(state)][mdpActionMap["end"]] = maxTreasure - (float)std::stoi(mat[y][x]);
            continue;
        }

        // reaching the horizon with no treasure incurs the maximum cost
        if((mat[y][x] == "SSS" || mat[y][x] == "---") && t == horizon - 1){
            R[mdpStateMap.at(state)][mdpActionMap["end"]] = maxTreasure;
            continue;
        }

        for(auto action : actionList){
            if(action == "end"){
                continue;
            }
            R[mdpStateMap[state]][mdpActionMap[action]] = costPerStep;
        }
    }

    return std::make_shared<MDP>(initProbs, T, R, mdpStateMap, mdpActionMap);
}
