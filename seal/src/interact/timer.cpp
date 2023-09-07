#include "timer.h"

#include <iostream>

Timer::Timer(std::string name) {
  funcName = name;
  startTime = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
  endTime = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime -
                                                                   startTime);
  std::cout << funcName << " costs " << duration.count() << " microseconds"
            << std::endl;
}