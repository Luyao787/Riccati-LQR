#pragma once
#include <vector>
#include "knotpoint.h"

#define lqr_float double

namespace lqr
{

struct LQR
{
    int n;
    int m;
    int N;

    std::vector<Knotpoint> knotpoints;
    std::vector<VectorXs> xs;
    std::vector<VectorXs> us;
};

} // namespace tvlqr