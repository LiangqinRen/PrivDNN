#include <chrono>
#include <string>

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> endTime;
    std::chrono::microseconds duration;

    std::string funcName;

public:
    Timer(std::string name = std::string("Function"));
    ~Timer();
};