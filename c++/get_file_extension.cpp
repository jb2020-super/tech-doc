#include <string>

void get_file_extension(const std::string& src, std::string& path, std::string& ext){
    auto pos = src.find_last_of('.');
    path = std::string(src.begin(), src.begin() + pos);
    ext = std::string(src.begin() + pos, src.end());
}