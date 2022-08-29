#include <iostream>
#include <numbers>

#include <Eigen/Dense>

template <typename T=float>
using Matrix_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T=float>
using Vector_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;


template <typename VectorT, typename MatrixT>
class MVN
{
    /*
     * Invision that MVN is instantiated with mu == VectorXd, sigsq = MatrixXd, which are 
     * both Eigen::Matrix types but note:
     * VectorXd == Eigen::Matrix<double,-1,1>
     * MatrixXd == Eigen::Matrix<double,-1,-1>
     * so we need to different template args.  This is generic enough to handle special
     * matrices, and matrices coming from expression types etc.
     * 
     * Last, I will go with row ordering, as I think this will be the most likely access pattern
     * for us where we are just doing repeated matrix multiplications.
     */

public:
    using Scalar_t = Eigen::MatrixBase<VectorT>::Scalar;
    MVN(const Eigen::MatrixBase<VectorT>& mu, const Eigen::MatrixBase<MatrixT>& sigsq)
        : _mu { mu }, _sigsq { sigsq }, _inv_sigsq { sigsq.inverse() }
    {        
         
        _dimension = static_cast<size_t> (mu.size());
        _norm_const = 1.0/std::sqrt( std::pow( 2.0*std::numbers::pi_v<Scalar_t>, _dimension) * sigsq.determinant() );
    }

    [[nodiscard]]
    inline Scalar_t operator()(const Eigen::MatrixBase<VectorT>& z) const 
    {
        return _norm_const*std::exp(-0.50* (z-_mu).transpose() * _inv_sigsq * (z-_mu));
    }

    [[nodiscard]]
    inline VectorT log_grad(const Eigen::MatrixBase<VectorT>& z) const
    { 
        return -(z - _mu).transpose()*_inv_sigsq; 
    }

private:
    
    // mean and variance.  is the reference ever a problem?
    const Eigen::MatrixBase<VectorT>& _mu;
    const Eigen::MatrixBase<MatrixT>& _sigsq;

    const Matrix_t<Scalar_t> _inv_sigsq;
    Scalar_t _norm_const; 
    size_t _dimension;

};

template <typename T>
class Kernel
{
public:
    Kernel(T alpha, T beta) : _alpha(alpha), _beta(beta) {}

    template <typename VectorT>
    [[nodiscard]]
    inline T operator()(const Eigen::MatrixBase<VectorT>& x, const Eigen::MatrixBase<VectorT>& y) const
    {
        auto&& diff { x - y };
        auto arg = _alpha + diff.squaredNorm();
        return std::pow(arg, _beta);
    }

    template <typename VectorT>
    [[nodiscard]]
    inline VectorT grad_x(const Eigen::MatrixBase<VectorT>& x, const Eigen::MatrixBase<VectorT>& y) const
    {
        auto&& diff { x - y };
        auto arg = _alpha + diff.squaredNorm();
        return 2.0*_beta*diff*std::pow(arg, _beta - 1);
        
    }

     template <typename VectorT>
    [[nodiscard]]
    inline VectorT grad_y(const Eigen::MatrixBase<VectorT>& x, const Eigen::MatrixBase<VectorT>& y) const
    {
        auto&& diff { x - y };
        auto arg = _alpha + diff.squaredNorm();
        return -2.0*_beta*diff*std::pow(arg, _beta - 1);
    }

    
    template <typename VectorT>
    [[nodiscard]]
    VectorT grad_xy(const Eigen::MatrixBase<VectorT>& x, const Eigen::MatrixBase<VectorT>& y) const
    {
        auto&& diff { x - y};
        auto arg { _alpha + diff.squaredNorm() };
        auto R { std::pow(arg, _beta - 2) };
        auto t1 { 2.0*(_beta - 1)*diff.array().abs2()* R };
        auto t2 { arg*R };
        return -2.0*_beta*(t1 + t2);
    }
    

    
    template <typename VectorT>
    [[nodiscard]]
    VectorT grad_yx(const Eigen::MatrixBase<VectorT>& x, const Eigen::MatrixBase<VectorT>& y) const
    {
        auto&& diff { x - y};
        auto arg { _alpha + diff.squaredNorm() };
        auto R  { std::pow(arg, _beta - 2) };
        auto t1 { 2.0*(_beta - 1)*diff.array().abs2()* R };
        auto t2 { arg*R };
        return -2.0*_beta*(t1 + t2);
    }

private:
    const T _alpha;
    const T _beta;
};

template <typename Target_t, typename Kernel_t>
class SRK
{   
public:

    // type is determined by the target, not the kernel.  
    // would be good to get this nice and consistent throughout though
    using Scalar_t = Target_t::Scalar_t;
    SRK(const Target_t& target, const Kernel_t& kernel) : _target(target), _kernel(kernel) {};

    template <typename VectorT>
    [[nodiscard]]
    Scalar_t operator()(const Eigen::MatrixBase<VectorT>& x, const Eigen::MatrixBase<VectorT>& y) const
    {
        // grad vector components
        
        auto dkdx { _kernel.grad_x(x, y) };
        auto dkdy { _kernel.grad_y(x, y) };
        auto d2k { _kernel.grad_xy(x, y) };
        auto dgdx { _target.log_grad(x) };
        auto dgdy { _target.log_grad(y) };
        Scalar_t k { _kernel(x, y) };
        // build solution.  element wise so do as an array
        return (d2k.array() + dkdx.array()*dgdy.array() +
               dkdy.array()*dgdx.array() + k*dgdx.array()*dgdy.array()).sum();

        
    }
private:
    const Target_t& _target;
    const Kernel_t& _kernel;

};

int main()
{

    std::cout << "Multivariate Normal Distribution in C++ " << std::endl;
    std::cout << " -- Initialise MVN Parameters:" << std::endl;
    const size_t d { 2 };

    Matrix_t sigsq(d, d);
    Vector_t mu(d);

    sigsq(0, 0) = 1.0;
    sigsq(0, 1) = sigsq(1, 0) = 0.50;
    sigsq(1, 1) = 2.0;
    std::cout << " --> Sigma: \n" << sigsq << std::endl;
    
    mu(0) = -0.50; 
    mu(1) = 0.50;
    std::cout << " --> mu: \n" << mu << std::endl;
    const MVN mvn {mu, sigsq};

    std::cout << " -- Initialise Kernel: " << std::endl;
    const float alpha { 1.0 };
    const float beta { -0.5 };
    const Kernel imq{alpha, beta};

    std::cout << " -- Initialise SRK: " << std::endl;
    SRK srk { mvn, imq };

    
    Vector_t z(d), z1(d);
    z(0) = 1.0; z(1) =  -1.0;
    z1(0) = 4.0; z1(1) = 0.0;
    std::cout << " --> compute srk(x, y) for two points... = " << srk(z, z1) << std::endl;

}