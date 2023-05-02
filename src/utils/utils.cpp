#include "utils.h"
#include <math.h>
#include <iostream>
#include <vector>
#include "state.h"
#include "mdp_examples.h"


bool cmpf(float A, float B, float eps)
{
    return (fabs(A - B) < eps);
}

std::vector<float> logspace(float a, float b, int k) {
  std::vector<float> logspace;
  for (int i = 0; i < k; i++) {
    logspace.push_back(pow(10, i * (b - a) / (k - 1)));
  }
  return logspace;
}

void printRoverDomain(std::vector<std::tuple<State,  std::string, float, std::unordered_map<State, float, StateHash>, std::unordered_map<State, float, StateHash>, float>> cvarGamePath){
    std::vector<std::vector<std::string>> mat = getRoverMatrix();
    for(auto tuple : cvarGamePath){
        int x = std::stoi(std::get<0>(tuple).getValue("x"));
        int y = std::stoi(std::get<0>(tuple).getValue("y"));
        if(y < 0 || x < 0){
            continue;
        }
        if(mat[y][x] == "X" || mat[y][x] == "+"){
            mat[y][x] = "+";
        }else{
            mat[y][x] = "o";
        }
    }

    int n = 0;
    std::cout << "  ";
    for(auto c : mat[0]){
        std::cout << n;
        if(n<=9){
            std::cout << " ";
        }
        n++;
    }
    std::cout << std::endl;

    int r = 0;
    for(auto row : mat){
        std::cout << r;
        if(r<=9){
            std::cout << " ";
        }
        r += 1;
        for(auto c : row){
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }
}

void printTrafficDomain(std::vector<std::tuple<State,  std::string, float, std::unordered_map<State, float, StateHash>, std::unordered_map<State, float, StateHash>, float>> cvarGamePath){
    std::vector<std::vector<std::string>> mat {
                {".",".",".","."},
                {".",".",".","."},
                {".",".",".","."},
                {".",".",".","."},
                {".",".",".","."}
    };

    for(auto tuple : cvarGamePath){
        int x = std::stoi(std::get<0>(tuple).getValue("x"));
        int y = std::stoi(std::get<0>(tuple).getValue("y"));
        if(y < 0 || x < 0){
            continue;
        }
        mat[y][x] = "X";
    }

    int n = 0;
    std::cout << "  ";
    std::vector<std::string> speeds{"m", "b", "m", "q"};
    for(auto c : speeds){
        std::cout << n << " ";
        n++;
    }
    std::cout << std::endl;


    for(int r = mat.size() - 1; r >= 0; r--){
        std::cout << r << " ";
        std::vector<std::string> row = mat[r];
        for(auto c : row){
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }
}
