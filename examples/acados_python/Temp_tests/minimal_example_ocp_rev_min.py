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

from acados_template import AcadosOcp, AcadosOcpSolver, plot_trajectories
from pendulum_model import export_pendulum_ode_model
import numpy as np
import casadi as ca

def main():
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_pendulum_ode_model()
    ocp.model = model

    Tf = 1.0
    nx = model.x.rows()
    nu = model.u.rows()
    N = 10

    # set prediction horizon
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf

    # cost matrices
    Q_mat = 2*np.diag([1e3, 1e3, 1e-2, 1e-2])
    R_mat = 2*np.diag([1e-2])

    # path cost
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.cost.yref = np.zeros((nx+nu,))
    ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

    # terminal cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.yref_e = np.zeros((nx,))
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.W_e = Q_mat

    # soft bound on x, using constraint h
    theta = ocp.model.x[3]
    x1 = ocp.model.x[0]
    u = ocp.model.u[0]
    theta_max = np.deg2rad(60)
    xmax = 0.5
    Fmax = 80

    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])
    # ocp.constraints.remove_x0_elimination()

    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([Fmax])
    ocp.constraints.idxbu = np.array([0])

    # initial soft constraint on h
    ocp.model.con_h_expr_0 = ca.vertcat(x1, theta)
    ocp.constraints.lh_0 = np.array([-xmax, -theta_max])
    ocp.constraints.uh_0 = np.array([+xmax, +theta_max])
    # ocp.constraints.idxsh_0 = np.array([0, 1]) # indices of slacked constraints within h
    ocp.constraints.idxs_rev_0 = np.array([-1,-1,-1,-1,-1,0,0]) # indices of slacked constraints within h, reversed order (from last to first)

    ocp.cost.zl_0 = np.ones((1,))
    ocp.cost.Zl_0 = np.zeros((1,))
    ocp.cost.zu_0 = np.ones((1,))
    ocp.cost.Zu_0 = np.zeros((1,))

    ########################################################
    # intermidiate soft constraints on h
    ocp.model.con_h_expr = ca.vertcat(x1, theta)
    ocp.constraints.lh = np.array([-xmax, -theta_max])
    ocp.constraints.uh = np.array([+xmax, +theta_max])
    ocp.constraints.idxs_rev = np.array([-1, 0, 0]) # indices of slacked constraints within h, reversed order (from last to first)
    # set penalty weight for slack variables
    ocp.cost.zl = np.ones((1,))
    ocp.cost.Zl = np.zeros((1,))
    ocp.cost.zu = np.ones((1,))
    ocp.cost.Zu = np.zeros((1,))
    #########################################################

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    # ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
    ocp.solver_options.nlp_solver_max_iter = 2
    ocp.solver_options.eval_residual_at_max_iter = True

    ocp_solver = AcadosOcpSolver(ocp)

    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))

    status = ocp_solver.solve()
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    ocp_solver.dump_last_qp_to_json(filename='/home/jingtao/acados/examples/acados_python/casadi_tests/qp_tests/pendulum_rev_min.json', overwrite=True)
    ocp_solver.store_iterate(filename='pendulum_sol.json', overwrite=True)

    # get solution
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")

if __name__ == '__main__':
    main()
