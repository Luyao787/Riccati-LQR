#pragma once

#include <Eigen/Dense>
#include <sched.h>
#include "eigen_types.h"
#include "utils.h"

namespace lqr {

LQR* LQR_setup(
    int n, int m, int N,
    const std::vector<Knotpoint>& knotpoints)
{
    LQR* lqr = new LQR; 
    lqr->n = n;
    lqr->m = m;
    lqr->N = N; 
    lqr->knotpoints = knotpoints; // Copy knotpoints
    lqr->xs.resize(N + 1);
    lqr->us.resize(N);
    
    // Set thread affinity to CPU core 0
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset); // Set thread affinity to CPU core 0
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) < 0) {
        printf("Error setting thread affinity for thread %d\n", sched_getcpu());
    }

    return lqr;
}

void LQR_free(LQR* lqr)
{
    delete lqr;
}

void backward(LQR& lqr)
{
    int N  = lqr.N;
    int nx = lqr.n;
    int nu = lqr.m;
    
    lqr.knotpoints[N].L.bottomRightCorner(nx, nx) = lqr.knotpoints[N].H.bottomRightCorner(nx, nx).llt().matrixL();
    lqr.knotpoints[N].lp.tail(nx) = lqr.knotpoints[N].h.tail(nx);

    for (int k = N - 1; k >= 0; --k) {
        MatrixRef V_k      = lqr.knotpoints[k].V;
        MatrixRef M_k      = lqr.knotpoints[k].M;
        MatrixRef L_k      = lqr.knotpoints[k].L;

        ConstMatrixRef E_k = lqr.knotpoints[k].E;
        ConstVectorRef c_k = lqr.knotpoints[k].c;
        ConstMatrixRef H_k = lqr.knotpoints[k].H;
        ConstVectorRef h_k = lqr.knotpoints[k].h;    

        ConstMatrixRef Lxx_next = lqr.knotpoints[k + 1].L.bottomRightCorner(nx, nx);   
        // auto Lxx_tri = Lxx_next.triangularView<Eigen::Lower>(); 
        // V_k.noalias() = E_k.transpose() * Lxx_tri;
        V_k.noalias() = E_k.transpose() * Lxx_next;

        M_k = H_k;
        // M_k.selfadjointView<Eigen::Lower>().rankUpdate(V_k, 1.0);
        M_k.noalias() += V_k * V_k.transpose();
        
        L_k = M_k.llt().matrixL();

        ConstVectorRef p_next = lqr.knotpoints[k + 1].lp.tail(nx);
        ConstMatrixRef Luu_k  = L_k.topLeftCorner(nu, nu);
        ConstMatrixRef Lxu_k  = L_k.bottomLeftCorner(nx, nu);
        
        VectorRef Pb_k   = lqr.knotpoints[k].Pb;
        VectorRef Pb_tmp = lqr.knotpoints[k].Pb_tmp;
        VectorRef lp_k   = lqr.knotpoints[k].lp;
        VectorRef lu_k   = lp_k.head(nu);
        VectorRef p_k    = lp_k.tail(nx);

        Pb_tmp.noalias() = Lxx_next.transpose() * c_k;
        Pb_k.noalias()   = Lxx_next * Pb_tmp;
        Pb_k            += p_next;
    
        lp_k             = h_k;
        lp_k.noalias()  += E_k.transpose() * Pb_k;

        Luu_k.triangularView<Eigen::Lower>().solveInPlace(lu_k);
        p_k.noalias()   -= Lxu_k * lu_k;
    }
}

void forward(const VectorXs& x0, LQR& lqr)
{
    lqr.xs[0] = x0;
    int nx = lqr.n;
    int nu = lqr.m;
   
    for (int k = 0; k < lqr.N; ++k) {
        ConstMatrixRef A_k   = lqr.knotpoints[k].E.topRightCorner(nx, nx);
        ConstMatrixRef B_k   = lqr.knotpoints[k].E.topLeftCorner(nx, nu);
        ConstVectorRef c_k   = lqr.knotpoints[k].c;
        MatrixRef      Luu_k = lqr.knotpoints[k].L.topLeftCorner(nu, nu);
        ConstMatrixRef Lxu_k = lqr.knotpoints[k].L.bottomLeftCorner(nx, nu);
        ConstVectorRef lu_k  = lqr.knotpoints[k].lp.head(nu);

        // lqr.us[k].noalias() = - Lxu_k.transpose() * lqr.xs[k] - lu_k;
        lqr.us[k]                = -lu_k;
        lqr.us[k].noalias()     -= Lxu_k.transpose() * lqr.xs[k];
        Luu_k.triangularView<Eigen::Lower>().transpose().solveInPlace(lqr.us[k]);

        // lqr.xs[k + 1].noalias() = A_k * lqr.xs[k] + B_k * lqr.us[k] + c_k;
        lqr.xs[k + 1]            = c_k;
        lqr.xs[k + 1].noalias() += A_k * lqr.xs[k];
        lqr.xs[k + 1].noalias() += B_k * lqr.us[k];
    }
}

void LQR_solve(const VectorXs& x0, LQR* lqr)
{
    backward(*lqr);
    forward(x0, *lqr);
}

} // namespace lqr


