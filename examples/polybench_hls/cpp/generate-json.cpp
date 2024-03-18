#include "../../../nlohmann/json.hpp"

#include "atax.cpp"
#include <iostream>
#include <stdio.h>
#include <string>

using json = nlohmann::json;

bool RUN = false;

json parse_input_json(std::string file_path) {
    // // Open the JSON file
    // std::ifstream file(file_path);

    // // Check if the file is open
    // if (!file.is_open()) {
    //     std::cerr << "Error opening file." << std::endl;
    //     return 1;
    // }

    // Parse the JSON data
    json j;
    return j;
    // try {
    //     file >> j;
    // } catch (json::parse_error& e) {
    //     std::cerr << "Parse error: " << e.what() << std::endl;
    //     return 1;
    // }
}


int main(int argc, char *argv[]) {
  json input_json = parse_input_json("atax-data-init.json");

  int A[116][124];
  for input_json["A"];
  int x[124];
  x = input_json["x"];
  int y[124];
  y = input_json["y"];
  kernel_atax(A, x, y);

  json j;
  json mems;
  mems["mem_3"] = A;
  mems["mem_2"] = x;
  mems["mem_1"] = y;

  j["cycles"] = "N/A";
  j["memories"] = mems;
  return j;

  std::cout << j.dump(4) << std::endl;
  return 0;
}