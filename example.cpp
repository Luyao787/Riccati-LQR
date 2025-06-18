
#include <iostream>
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include "eigen_types.h"
#include "knotpoint.h"
#include "lqr.h"
#include "utils.h"

using namespace std;
using namespace Eigen;
using namespace lqr;

int main() {
    
    /* Quadrotor */
    constexpr int n = 12;
    constexpr int m = 4;
    constexpr int N = 100; // horizon length
    cout << "horizon: " << N << endl;

    VectorXs x0(n);
    VectorXs x_ref(n);
    x0 << 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.;
    x_ref << 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.;

    MatrixXs Q(n, n);
    MatrixXs R(m, m);
    MatrixXs S(m, n);
    VectorXs q(n);
    VectorXs r(m);
    MatrixXs A(n, n);
    MatrixXs B(n, m);
    VectorXs c(n);
  
    // Q.diagonal() << 0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.;
    Q.diagonal() << 1e-5, 1e-5, 10., 10., 10., 10., 1e-5, 1e-5, 1e-5, 5., 5., 5.; // N.B. Q_N is required to be positive definite in current implementation

    R.diagonal() << 0.1, 0.1, 0.1, 0.1;
    S.setZero();
    q = -x_ref.transpose() * Q;
    r.setZero();

    A << 
        1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.,
        0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.,
        0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.,
        0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.,
        0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.,
        0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992,
        0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.,
        0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.,
        0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.,
        0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.,
        0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.,
        0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846;
    B <<
        0.,      -0.0726,  0.,     0.0726,
        -0.0726,   0.,      0.0726, 0.,
        -0.0152,   0.0152, -0.0152, 0.0152,
        -0.,      -0.0006, -0.,     0.0006,
        0.0006,   0.,     -0.0006, 0.0000,
        0.0106,   0.0106,  0.0106, 0.0106,
        0.,      -1.4512,  0.,     1.4512,
        -1.4512,   0.,      1.4512, 0.,
        -0.3049,   0.3049, -0.3049, 0.3049,
        -0.,      -0.0236,  0.,     0.0236,
        0.0236,   0.,     -0.0236, 0.,
        0.2107,   0.2107,  0.2107, 0.2107;
        
    c.setZero();

    vector<Knotpoint> knotpoints(N + 1, Knotpoint(n, m));

    for (int k = 0; k < N; ++k) {
        knotpoints[k].E << B, A;
        knotpoints[k].c = c;
        knotpoints[k].H.setZero();
        knotpoints[k].h.setZero();

        knotpoints[k].L.setZero();
        knotpoints[k].V.setZero();
        knotpoints[k].M.setZero();
        knotpoints[k].lp.setZero();
        knotpoints[k].Pb.setZero();
        knotpoints[k].Pb_tmp.setZero();

        knotpoints[k].H.block(0, 0, m, m) = R;
        knotpoints[k].H.block(m, m, n, n) = Q;
        knotpoints[k].H.block(0, m, m, n) = S;
        knotpoints[k].H.block(m, 0, n, m) = S.transpose();
        knotpoints[k].h.head(m) = r;
        knotpoints[k].h.tail(n) = q;
    }
    knotpoints[N].H.setZero();
    knotpoints[N].h.setZero();
    knotpoints[N].H.block(m, m, n, n) = Q;
    knotpoints[N].h.tail(n) = q;
 
    /* ------------------------ LQR ------------------------------- */
    constexpr int num_iter = 5e3;

    LQR* lqr = LQR_setup(n, m, N, knotpoints); 
    
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iter; ++i) 
    {
        LQR_solve(x0, lqr);
    }
    auto end   = chrono::high_resolution_clock::now();
    cout << "Mean solve time: " 
         << chrono::duration_cast<chrono::milliseconds>(end - start).count() / double(num_iter) 
         << " ms" << endl;
    // Print out the firt 10 control inputs
    cout << "First 10 control inputs:\n";
    for (int i = 0; i < 10 && i < lqr->N; ++i) {
        std::cout << "u_" << i << ": " << lqr->us[i].transpose() << std::endl;
    }
    // Print out the last state 
    std::cout << "x_N:\n" << lqr->xs[N].transpose() << std::endl;

    LQR_free(lqr);

    return 0;
}