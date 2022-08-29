// Not so sure where to put these...
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <cassert>

namespace stein
{
    namespace util
    {
        uint64_t count_fields(const std::string &s, char ch = ',')
        {
            return 1 + std::count(s.begin(), s.end(), ch);
        }

        void split_line(const std::string &str, std::vector<std::string> &v, char ch = ',')
        {
            assert(count_fields(str) == v.size());
            std::string l;
            std::stringstream ss(str);
            size_t idx{0};
            while (std::getline(ss, l, ','))
            {
                v[idx] = l;
                ++idx;
            }
        };

        std::vector<std::string> split_line(const std::string &str, char ch = ',')
        {
            std::vector<std::string> tmp(count_fields(str, ch));
            split_line(str, tmp, ch);
            return tmp;
        };
    }
}