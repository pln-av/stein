#pragma once
#include <limits>

#include "types.H"
#include "kernel.H"
#include "target.H"
#include <Eigen/Dense>

namespace stein
{
    template <typename Target_t, typename Kernel_t>
    class SRK
    {
    public:
        // type is determined by the target, not the kernel.
        // would be good to get this nice and consistent throughout though
        using Scalar_t = Target_t::Scalar_t;
        SRK(const Target_t &target, const Kernel_t &kernel) : _target(target), _kernel(kernel){};

        template <typename VectorT>
        [[nodiscard]] Scalar_t operator()(const Eigen::MatrixBase<VectorT> &x, const Eigen::MatrixBase<VectorT> &y) const
        {
            // grad vector components

            const auto dkdx{_kernel.grad_x(x, y)};
            const auto dkdy{_kernel.grad_y(x, y)};
            const auto d2k{_kernel.grad_xy(x, y)};
            const auto dgdx{_target.log_grad(x)};
            const auto dgdy{_target.log_grad(y)};
            Scalar_t k{_kernel(x, y)};
      
            return (d2k.array() + dkdx.array() * dgdy.array() +
                    dkdy.array() * dgdx.array() + k * dgdx.array() * dgdy.array())
                .sum();
        }

    private:
        const Target_t &_target;
        const Kernel_t &_kernel;
    };

    
    using SRK_t = stein::SRK<stein::MVN_t, stein::IMQ_t>;

    template <typename T>
    class KSD
    {
        using Scalar_t = T::Scalar_t;
    public:
        KSD(const T& ksd) : _ksd(ksd) {};
        template <typename U>
        Scalar_t operator()(U beg, U end)
        {
            // this should be like a standard stl function
            // so doesnt make sense if beg==end
            if (beg==end) { 0.0; }

            Scalar_t out = 0.0;
            auto it_outer = beg;
            uint64_t n {0};
            while (it_outer != end)
            {
                auto it_inner = beg;
                while (it_inner != end)
                {
                    out += _ksd(*it_outer, *it_inner);
                    ++it_inner;
                    ++n;
                }
                ++it_outer;
            }

            return std::sqrt(out/ (n*n) );
        }

    private:
        const T& _ksd;
    };
}