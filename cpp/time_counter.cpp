#include <chrono>

void function(){
    auto start = std::chrono::high_resolution_clock::now();
    // do something
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}