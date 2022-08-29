#pragma once

// various utilities for reading/writing files
#include <string>
#include <filesystem>
#include <fstream>
#include <cassert>

#include "types.H"
#include "util.H"

namespace stein
{
    namespace file
    {
        class File
        {
        public:
            File(void) {}
            template <typename T>
            void write(const std::string& file_name, const std::vector<Vector_t<T>>& points)
            {
                // write your points to file.
                // i assume these files are enormous so this will do for now
                std::fstream of;
                of.open(file_name, std::ios_base::out);
                const size_t n = points.size();
                for (size_t p_idx=0; p_idx<n; ++p_idx)
                {
                    const size_t m = points[p_idx].size();
                    for (size_t j=0; j<m; ++j)
                    {
                        of << points[p_idx](j);
                        if (j < m-1) { of << ","; }
                        else { of << "\n"; }
                    }
                }
                of.close();
            }

            template<typename T>
            std::vector<stein::Vector_t<T>> read(const std::string& file_name) 
            {
                assert( std::filesystem::exists(file_name) );
                auto convert = [](const std::string& s)
                {
                    return (std::is_same<T,double>::value) ? std::stod(s) : std::stof(s);
                }; 
                std::fstream of;
                std::vector<stein::Vector_t<T>> out;
                of.open(file_name, std::ios_base::in);
                if (of.is_open())
                {
                    std::string l;
                    while (std::getline(of, l) )
                    {
                        std::vector<std::string> line { stein::util::count_fields(l, ',') };
                        stein::util::split_line(l, line);
                        stein::Vector_t<T> v(line.size());
                        for (size_t i=0; i<v.size(); ++i) { v(i) = std::stod(line[i]); }
                        out.push_back(v);
                    }
                }
                return out;
            }
        private:
            const std::string _file_name;

        };
    }
}