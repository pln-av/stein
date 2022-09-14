#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <cassert>

#include "stein/la/matrix.H"

class Writer
{
public:
  
  Writer(const std::string f) : _file_name(f) {  };

  void initialise(void)
  {
    
    _out_file.open(_file_name, std::ios::out);
    assert( _out_file.is_open() );
  }
  
  ~Writer() { _out_file.close(); }

  template <typename T>
  void write(const T& x, typename T::value_type f)
  {
    const stein::util::int_t n { x.size() };
    for (stein::util::int_t i=0; i<n; ++i)
      {
	_out_file << std::to_string(x(i)) << ",";
      }
    _out_file << f << "\n";
  }
private:
  std::string _file_name;
  std::ofstream _out_file;
};
