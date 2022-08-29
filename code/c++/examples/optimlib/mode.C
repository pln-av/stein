// compute mode of multivariate normal using OptimLib
// (base) optimlib % g++ -std=c++20 mode.C -I/Users/patricknoble/Documents/Library/Eigen -I/Users/patricknoble/Documents/Projects/stein/code/c++/lib -I/Users/patricknoble/Documents/Library/OptimLib -o mode
#include <iostream>

#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include "optim.hpp"

#include <types.H>
#include <target.H>

/*
double obj(const stein::dVector_t& x, stein::dVector_t* grad, void* opt_data)
{

    // NB the element type in stein::Vector_t cannot be correctly deduced when passed to optim::de
    // I can probably work out why, but at this point just write in the type manually.  note that
    // OptimLib REQUIRES double precision, so there is no real loss other than syntactic sugar

    // note that OptimLib has strict type requirements on inputs, so we can quite reliably
    // typedef everything closely, since OptimLib is so restrictive
    using MVN_t = stein::target::MVN<stein::dVector_t,stein::dMatrix_t>;
    MVN_t* _mvn_obj_ = reinterpret_cast<MVN_t*> (opt_data);
    return -1.0*_mvn_obj_->operator()(x);
}
*/

int main()
{
    std::cout << "Compute MVN Mode with OptimLib."  << std::endl;

    std::cout << "Initialise MVN Target." << std::endl;
    const size_t d { 2 };

    stein::dMatrix_t sigsq(d, d);
    stein::dVector_t mu(d);

    sigsq(0, 0) = 1.0;
    sigsq(0, 1) = sigsq(1, 0) = 0.50;
    sigsq(1, 1) = 2.0;
    
    mu(0) = -0.50; 
    mu(1) = 0.50;

    stein::target::MVN mvn {mu, sigsq};
    
    // call obj with optim::de
    std::cout << "Initialize random ic and solve with optim::de." << std::endl;
    Eigen::VectorXd ic = Eigen::VectorXd::Random(d);

    // we can pass a lambda, or the void ptr.  i wonder which is faster?
    auto tobj = [&mvn](const stein::dVector_t&x, stein::dVector_t* g, void* optdata){ return -1.0*mvn(x);};
    bool success = optim::de(ic, tobj, &mvn);
    if (success)
    {
        std::cout << " *** minimum found at x_1: \n" << ic << std::endl;
    }
    else
    {
        std::cout << " *** failed to find minimum. " << std::endl;
    }
    

}
