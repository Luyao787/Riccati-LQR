# pragma once

# include "eigen_types.h"

namespace lqr
{

struct Knotpoint
{
    MatrixXs E; // E = [B A]
    VectorXs c;
    MatrixXs H; // H = [R S; S^T Q]
    VectorXs h; // h = [r; q]
    
    MatrixXs L;    // L = [Luu Lux; Lxu Lxx]
    MatrixXs V, M; // V = E^T * Lxx, M = H + V * V^T

    VectorXs lp; 
    VectorXs Pb, Pb_tmp;
    
    Knotpoint(int nx, int nu)
    {
        int nxu = nx + nu;
        E.resize(nx, nxu);
        c.resize(nx);
        H.resize(nxu, nxu);
        h.resize(nxu);
        L.resize(nxu, nxu);
        V.resize(nxu, nx);
        M.resize(nxu, nxu);
        lp.resize(nxu);
        Pb.resize(nx);
        Pb_tmp.resize(nx);
    }
};
    
} // namespace lqr

