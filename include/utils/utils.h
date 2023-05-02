#ifndef utils
#define utils
#include <iostream>
#include <cmath>
#include <vector>
#include "state.h"

bool cmpf(float A, float B, float eps = 0.0001);
std::vector<float> logspace(float a, float b, int k);
void printRoverDomain(std::vector<std::tuple<State,  std::string, float, std::unordered_map<State, float, StateHash>, std::unordered_map<State, float, StateHash>, float>> cvarGamePath);
void printTrafficDomain(std::vector<std::tuple<State,  std::string, float, std::unordered_map<State, float, StateHash>, std::unordered_map<State, float, StateHash>, float>> cvarGamePath);

#endif
