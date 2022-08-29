#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include "optim.hpp"
        
#define OPTIM_PI 3.14159265358979

double 
ackley_fn(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* opt_data)
{
    const double x = vals_inp(0);
    const double y = vals_inp(1);

    const double obj_val = 20 + std::exp(1) - 20*std::exp( -0.2*std::sqrt(0.5*(x*x + y*y)) ) - std::exp( 0.5*(std::cos(2 * OPTIM_PI * x) + std::cos(2 * OPTIM_PI * y)) );
            
    return obj_val;
}
        
int main()
{
    Eigen::VectorXd x = 2.0 * Eigen::VectorXd::Ones(2); // initial values: (2,2)
        
    bool success = optim::de(x, ackley_fn, nullptr);
        
    if (success) {
        std::cout << "de: Ackley test completed successfully." << std::endl;
    } else {
        std::cout << "de: Ackley test completed unsuccessfully." << std::endl;
    }
        
    //arma::cout << "de: solution to Ackley test:\n" << x << arma::endl;
    std::cout << "de: solution to Ackley test:\n" << x << std::endl; 
    return 0;
}