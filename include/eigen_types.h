#pragma once

#include <Eigen/Dense>

namespace lqr
{
    using VectorXs = Eigen::VectorXd;
    using MatrixXs = Eigen::MatrixXd;
    
    using VectorMap = Eigen::Map<VectorXs>;
    using MatrixMap = Eigen::Map<MatrixXs>;
    using ConstVectorMap = Eigen::Map<const VectorXs>;
    using ConstMatrixMap = Eigen::Map<const MatrixXs>;

    using VectorRef = Eigen::Ref<VectorXs>;
    using MatrixRef = Eigen::Ref<MatrixXs>;
    using ConstVectorRef = Eigen::Ref<const VectorXs>;
    using ConstMatrixRef = Eigen::Ref<const MatrixXs>;
    
}   // namespace lqr

