// compute stein points of multivariate normal using OptimLib
// optimlib % g++ -std=c++20 -O3 -march=native -ffp-contract=fast mvn1.C 
// -I/Users/patricknoble/Documents/Library/Eigen -I/Users/patricknoble/Documents/Projects/stein/code/c++/lib -I/Users/patricknoble/Documents/Library/OptimLib -o mvn1
#include <iostream>
#include <chrono>

#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include "optim.hpp"

#include <types.H>
#include <target.H>
#include <kernel.H>
#include <stein.H>
#include <file.H>

double target_obj(const stein::dVector_t& x, stein::dVector_t* grad, void* opt_data)
{

    // NB the element type in stein::Vector_t cannot be correctly deduced when passed to optim::de
    // I can probably work out why, but at this point just write in the type manually.  note that
    // OptimLib REQUIRES double precision, so there is no real loss other than syntactic sugar

    // note that OptimLib has strict type requirements on inputs, so we can quite reliably
    // typedef everything closely, since OptimLib is so restrictive
    stein::MVN_t* _mvn_obj_ = reinterpret_cast<stein::MVN_t*> (opt_data);
    return -1.0*_mvn_obj_->operator()(x);
}

struct SRK_OPT_DATA
{
    // not used, but you could use this in the void ptr as a 3rd alternative.
    SRK_OPT_DATA( std::vector<stein::dVector_t> x, stein::SRK_t* srk) : x(x), srk(srk) {};
    std::vector<stein::dVector_t> x;
    stein::SRK_t* srk;
};

double point_obj(const stein::dVector_t& x, stein::dVector_t* grad, void* opt_data)
{
    stein::SRK_t* _srk_obj_ = reinterpret_cast<stein::SRK_t*> (opt_data);
    return 10.0;
}

std::chrono::time_point<std::chrono::high_resolution_clock> time_now(void)
{
    return std::chrono::high_resolution_clock::now();
}

int main(int argc, char* argv[])
{
    std::cout << "Compute MVN Mode with OptimLib."  << std::endl;
    const size_t d { 2 };

    stein::dMatrix_t sigsq(d, d);
    stein::dVector_t mu(d);

    sigsq(0, 0) = 1.0;
    sigsq(0, 1) = sigsq(1, 0) = 0.50;
    sigsq(1, 1) = 2.0;
    
    mu(0) = -0.50; 
    mu(1) = 0.50;
    
    // call obj with optim::de
     Eigen::VectorXd ic = Eigen::VectorXd::Random(d);

    // we can pass a lambda, or the void ptr.  i wonder which is faster?
    // do this one as function, next one as lambda for this test example.
    stein::target::MVN mvn {mu, sigsq};
    const size_t n { 20 };
    std::cout << "Begin Solving for " << n << " Stein Points." << std::endl;
    bool success = optim::de(ic, target_obj, &mvn);
    if (success)
    {
        std::cout << " ** Stein Point (1) Found: " << ic.transpose() << std::endl;
    }
    else
    {
        std::cout << " *** failed to find minimum. " << std::endl;
        std::exit(1);
    }
    
    // now lets try to call the optimisation routines for stein
    std::vector< stein::dVector_t > x;
    x.push_back(ic);

    // to compute points we need kernel and SRK 
    const double alpha { 1.0 };
    const double beta { -0.5 };
    const stein::kernel::IMQ imq{alpha, beta};

    stein::SRK srk { mvn, imq };

    // lets do it this way to avoid the void ptr?
    auto point_obj = [&srk, &x](const stein::dVector_t&z, stein::dVector_t* g, void* optdata)
    {
        double out { 0.50* srk(z, z) };
        for (auto&& xi : x)
        {
            out += srk(xi, z);
        }
        return out;
    };

    // try this in a loop...
    size_t point_idx = x.size();
    auto s0 { time_now() };
    while (point_idx < n)
    {
        ic = x.back();
        auto s { time_now() };
        bool success = optim::de(ic, point_obj, nullptr );
        auto e { time_now() };
        const std::chrono::duration<double> elapsed { e-s };
        if (success)
        {
            ++point_idx;
            std::cout << " ** Stein Point (" << point_idx << ") Found: " << ic.transpose() << " in " << elapsed.count() << " seconds." << std::endl;
            x.push_back(ic);
        }
    }
    auto e0 { time_now() };
    const std::chrono::duration<double> elapsed0 { e0 - s0 };
    std::cout << "Solver Completed in " << elapsed0.count()/60.0 << " minutes. " << std::endl;

    // write points to file
    const std::string file_name { "points.csv" };
    stein::file::File of = stein::file::File{};
    of.write(file_name, x);

    const std::vector<stein::dVector_t> x2 = of.read<double>(file_name);
    stein::KSD<stein::SRK_t> ksd(srk);
    for (size_t i=0; i<x2.size(); ++i)
    {
        std::cout << "KSD(0," << i << ")=" << ksd(x.begin(), x.begin()+i) << std::endl;

    }
}
