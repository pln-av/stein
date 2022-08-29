#include <iostream>
#include <Eigen/Dense>

// to compile include search paths:
// Eigen: -I/Users/patricknoble/Documents/Library/Eigen 
// Stein: -I/Users/patricknoble/Documents/Projects/stein/code/c++/lib 

#include "types.H"
#include "target.H"
#include "kernel.H"
#include "stein.H"

int main()
{

    std::cout << "Test Stein 1." << std::endl;
    const size_t d { 2 };

    stein::Matrix_t sigsq(d, d);
    stein::Vector_t mu(d);

    sigsq(0, 0) = 1.0;
    sigsq(0, 1) = sigsq(1, 0) = 0.50;
    sigsq(1, 1) = 2.0;
    
    mu(0) = -0.50; 
    mu(1) = 0.50;

    std::cout << " -- Initialise MVN: " << std::endl;
    const stein::target::MVN mvn {mu, sigsq};

    std::cout << " -- Initialise Kernel: " << std::endl;
    const double alpha { 1.0 };
    const double beta { -0.5 };
    const stein::kernel::IMQ imq{alpha, beta};

    std::cout << " -- Initialise SRK: " << std::endl;
    stein::SRK srk { mvn, imq };

    stein::Vector_t z  = Eigen::VectorXd::Random(d);
    stein::Vector_t z1 = Eigen::VectorXd::Random(d);
    stein::Vector_t z2 = Eigen::VectorXd::Random(d);
    stein::Vector_t z3 = Eigen::VectorXd::Random(d);
    std::cout << z << std::endl;
    std::cout << z1 << std::endl;
    std::cout << "srk(z, z1)=" << srk(z, z1) << std::endl;
    std::cout << "srk(z, z2)=" << srk(z, z2) << std::endl;
    std::cout << "srk(z, z3)=" << srk(z, z3) << std::endl;

}