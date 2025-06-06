#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
from casadi import vertcat, atan, exp, cos, sin, sqrt, SX
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

from plot_utils import plot_rsm_trajectories, plot_hexagon


WITH_ELLIPSOIDAL_CONSTRAINT = True
# WITH_ELLIPSOIDAL_CONSTRAINT = False
WITH_HEXAGON_CONSTRAINT = True
# WITH_HEXAGON_CONSTRAINT = False
USE_RTI = True
# USE_RTI = False

USE_PLANT = True
# USE_PLANT = False

# multiple executions for consistent timings:
N_EXEC = 5

# shooting intervals
N = 2

Ts = 0.0008

i_d_ref = 1.484
i_q_ref = 1.429
w_val   = 200

i_d_ref = -20
i_q_ref = 20
w_val   = 300

udc = 580
u_max = 2/3*udc
Rs = 0.4

w_val_tilde = .5 * w_val

X0 = np.array([0.0, 0.0])


# fitted psi_d map
def psi_d_num(x,y):
    #    This function was generated by the Symbolic Math Toolbox version 8.0.
    #    07-Feb-2018 23:07:49

    psi_d_expression = x*(-4.215858085639979e-3) + \
        exp(y**2*(-8.413493151721978e-5))* \
        atan(x*1.416834085282644e-1)*8.834738694115108e-1

    return psi_d_expression

def psi_q_num(x,y):
    #    This function was generated by the Symbolic Math Toolbox version 8.0.
    #    07-Feb-2018 23:07:50

    psi_q_expression = y*1.04488335702649e-2+ \
        exp(x**2*(-1.0/7.2e1))*atan(y)*6.649036351062812e-2

    return psi_q_expression


def export_rsm_model():
    model_name = 'rsm'

    # set up states
    psi_d = SX.sym('psi_d')
    psi_q = SX.sym('psi_q')
    x = vertcat(psi_d, psi_q)

    # set up controls
    u_d = SX.sym('u_d')
    u_q = SX.sym('u_q')
    u = vertcat(u_d, u_q)

    # set up algebraic variables
    i_d = SX.sym('i_d')
    i_q = SX.sym('i_q')
    z = vertcat(i_d, i_q)

    # set up xdot
    psi_d_dot = SX.sym('psi_d_dot')
    psi_q_dot = SX.sym('psi_q_dot')
    xdot = vertcat(psi_d_dot, psi_q_dot)

    # set up parameters
    w      = SX.sym('w') # speed
    dist_d = SX.sym('dist_d') # d disturbance
    dist_q = SX.sym('dist_q') # q disturbance
    p      = vertcat(w, dist_d, dist_q)

    # build flux expression
    Psi = vertcat(psi_d_num(i_d, i_q), psi_q_num(i_d, i_q))

    # dynamics
    f_impl = vertcat(psi_d_dot - u_d + Rs*i_d - w*psi_q - dist_d,
                     psi_q_dot - u_q + Rs*i_q + w*psi_d - dist_q,
                     psi_d - Psi[0],
                     psi_q - Psi[1])

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name

    if WITH_ELLIPSOIDAL_CONSTRAINT:
        r = SX.sym('r', 2, 1)
        model.con_phi_expr = r[0]**2 + r[1]**2
        model.con_r_expr = vertcat(u_d, u_q)
        model.con_r_in_phi = r

        model.con_phi_expr_0 = model.con_phi_expr
        model.con_r_expr_0 = model.con_r_expr
        model.con_r_in_phi_0 = model.con_r_in_phi

    return model


def compute_y_ref(w_val):
    # compute steady-state u
    psi_d_ref = psi_d_num(i_d_ref, i_q_ref)
    psi_q_ref = psi_q_num(i_d_ref, i_q_ref)
    u_d_ref = Rs*i_d_ref - w_val*psi_q_ref
    u_q_ref = Rs*i_q_ref + w_val*psi_d_ref

    return np.array([psi_d_ref, psi_q_ref, u_d_ref, u_q_ref])


def create_ocp_solver(tol = 1e-3):

    ocp = AcadosOcp()
    model = export_rsm_model()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    nz = model.z.rows()
    ny = nu + nx
    ny_e = nx
    Tf = N*Ts

    # set number of shooting intervals
    ocp.solver_options.N_horizon = N

    # set cost
    Q = np.diag([5e2, 5e2])
    R = np.diag([1e-4, 1e-4])

    ocp.cost.W = scipy.linalg.block_diag(Q, R)

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[2,0] = 1.0
    Vu[3,1] = 1.0
    ocp.cost.Vu = Vu

    ocp.cost.Vz = np.zeros((ny, nz))

    Q_e = np.diag([1e-3, 1e-3])
    ocp.cost.W_e = Q_e

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)

    ocp.cost.Vx_e = Vx_e

    y_ref = compute_y_ref(w_val)
    ocp.cost.yref = y_ref
    ocp.cost.yref_e = y_ref[:ny_e]

    ## setup constraints
    ocp.constraints.x0 = X0

    # bounds on u
    q2 = u_max*sin(np.pi/3)
    lbu = np.array([-q2])
    ubu = np.array([+q2])
    ocp.constraints.idxbu = np.array([1])
    ocp.constraints.lbu = lbu
    ocp.constraints.ubu = ubu

    # polytopic constraint on the input
    x1 = u_max
    y1 = 0
    x2 = u_max*cos(np.pi/3)
    y2 = u_max*sin(np.pi/3)

    q1 = -(y2 - y1/x1*x2)/(1-x2/x1)
    m1 = -(y1 + q1)/x1

    if WITH_ELLIPSOIDAL_CONSTRAINT:
        # to avoid LICQ violations
        eps = 1e-3 # Note: was originally eps = 0.0.
        ocp.constraints.lphi = np.array([-1.0e8])
        ocp.constraints.uphi = (1-eps)*np.array([(u_max*sqrt(3)/2)**2])
        ocp.constraints.lphi_0 = ocp.constraints.lphi
        ocp.constraints.uphi_0 = ocp.constraints.uphi

    if WITH_HEXAGON_CONSTRAINT:
        # lg <= C*x + D*u <= ug
        ocp.constraints.D = np.array([[m1, 1],[-m1, 1]])
        ocp.constraints.C = np.zeros((nx, nx))
        ocp.constraints.ug  = np.array([-q1, -q1])
        ocp.constraints.lg  = np.array([+q1, +q1])

    # setting parameters
    ocp.parameter_values = np.array([w_val, 0.0, 0.0])

    # set QP solver
    # ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_DAQP'
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_num_stages = 2
    ocp.solver_options.sim_method_newton_iter = 20
    ocp.solver_options.sim_method_newton_tol = 1e-6

    # set prediction horizon
    ocp.solver_options.tf = Tf
    ocp.solver_options.tol = tol
    ocp.solver_options.qp_tol = tol/10.
    if USE_RTI:
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    else:
        ocp.solver_options.nlp_solver_type = 'SQP'

    acados_solver = AcadosOcpSolver(ocp, verbose=True)

    return acados_solver


def setup_acados_integrator(ocp=None):

    if ocp is None:
        integrator = AcadosSimSolver(ocp)
    else:
        sim = AcadosSim()
        sim.model = export_rsm_model()
        sim.model.name = 'rsm_integrator'
        sim.solver_options.integrator_type = 'IRK'
        sim.solver_options.sens_forw = False
        sim.solver_options.sens_adj = False
        sim.solver_options.num_stages = 6
        sim.solver_options.num_steps = 3
        sim.solver_options.newton_tol = 1e-10
        sim.solver_options.newton_iter = 50
        sim.solver_options.T = Ts
        sim.parameter_values = np.array([w_val, 0.0, 0.0])
        integrator = AcadosSimSolver(sim)

    return integrator

def main():
    acados_solver = create_ocp_solver()
    ocp = acados_solver.acados_ocp
    nx = ocp.dims.nx
    nu = ocp.dims.nu
    N = ocp.solver_options.N_horizon
    Nsim = 100

    if USE_PLANT:
        plant = setup_acados_integrator(acados_solver.acados_ocp)

    simX = np.zeros((Nsim, nx))
    simU = np.zeros((Nsim, nu))
    simY = np.zeros((Nsim, nu+nx))
    times_prep = np.zeros(Nsim)
    times_feed = np.zeros(Nsim)

    p_val_1 = np.array([w_val, 0, 0])
    y_ref_1 = compute_y_ref(w_val)

    p_val_2 = np.array([w_val_tilde, 0, 0])
    y_ref_2 = compute_y_ref(w_val_tilde)

    for i_exec in range(N_EXEC):
        xcurrent = X0.copy()
        acados_solver.reset()

        # # initialize
        # for i in range(N):
        #     acados_solver.set(i, 'u', -0.9 * u_max * np.ones(nu,))

        # simulation
        for i in range(Nsim):

            # update params and y_ref
            if i > Nsim / 3 and i < Nsim / 2:
                p_val = p_val_2
                y_ref = y_ref_2
            else:
                p_val = p_val_1
                y_ref = y_ref_1

            # set y_ref
            for j in range(N):
                acados_solver.cost_set(j, "yref", y_ref)
            acados_solver.cost_set(N, "yref", y_ref[:nx])

            # set params
            p_flat = np.tile(p_val, N+1)
            acados_solver.set_flat("p", p_flat)

            # preparation rti_phase
            if USE_RTI:
                acados_solver.options_set('rti_phase', 1)
                status = acados_solver.solve()
                time_prep = acados_solver.get_stats('time_tot') * 1e3
                if i_exec == 0:
                    times_prep[i] = time_prep
                else:
                    times_prep[i] = min(times_prep[i], time_prep)

            # update initial condition
            acados_solver.set(0, "lbx", xcurrent)
            acados_solver.set(0, "ubx", xcurrent)

            # feedback rti_phase
            if USE_RTI:
                acados_solver.options_set('rti_phase', 2)

            # solve
            status = acados_solver.solve()
            time_feed = acados_solver.get_stats('time_tot') * 1e3
            if i_exec == 0:
                times_feed[i] = time_feed
            else:
                times_feed[i] = min(times_feed[i], time_feed)
            if status != 0:
                acados_solver.print_statistics()
                # raise Exception(f'acados returned status {status}.')

            # get solution
            u0 = acados_solver.get(0, "u")

            simX[i, :] = xcurrent
            simU[i, :] = u0
            simY[i, :] = y_ref

            # get next state
            if USE_PLANT:
                plant.set('u', u0)
                plant.set('x', xcurrent)
                plant.set('p', p_val)
                plant.solve()
                xcurrent = plant.get('x')
            else:
                xcurrent = acados_solver.get(1, "x")

    # timings
    cpu_times = times_prep + times_feed

    print(f"Ran experiment with {WITH_ELLIPSOIDAL_CONSTRAINT=}, {WITH_HEXAGON_CONSTRAINT=}, {USE_RTI=}")
    print(f"CPU time in ms: min {np.min(cpu_times):.3f}, median {np.median(cpu_times):.3f}, max {np.max(cpu_times):.3f}")

    # plot results
    plot_rsm_trajectories(simX, simU, simY, Ts)
    plot_hexagon(simU, u_max, simY[:, nx:])

    plt.show()

if __name__ == "__main__":
    main()
