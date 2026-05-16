import os
from acados_template import AcadosOcpQpSolver, AcadosCasadiOcpQpSolver, AcadosOcpQp, AcadosOcpQpOptions
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.abspath(os.path.join(script_dir, 'qp_tests'))
qp_json_files = [('drone_FrenSer_QP.json', 'drone_FrenSer_QP_com.json'),]
def main():
    for qp_json_file, qp_json_file_com in qp_json_files:

        json_name = os.path.basename(qp_json_file)
        qp_json_file = os.path.join(json_dir, qp_json_file)
        qp = AcadosOcpQp.from_json(qp_json_file)

        casadi_solver = AcadosCasadiOcpQpSolver(qp)
        # casadi_solver.set_iterate(iterate_acados) # set initial guess from acados solution
        status = casadi_solver.solve()
        casadi_u = np.array([casadi_solver.get(i, "u") for i in range(qp.N)])
        casadi_x = np.array([casadi_solver.get(i, "x") for i in range(qp.N+1)])
        casadi_sl = np.concatenate([casadi_solver.get(i, "sl") for i in range(qp.N+1)])
        casadi_su = np.concatenate([casadi_solver.get(i, "su") for i in range(qp.N+1)])
        casadi_lam = np.concatenate([casadi_solver.get(i, "lam") for i in range(qp.N+1)])
        casadi_pi = np.concatenate([casadi_solver.get(i, "pi") for i in range(qp.N)])
        iterate_casadi = casadi_solver.get_iterate()
        # casadi_cost = casadi_solver.get_cost()

        json_name = os.path.basename(qp_json_file_com)
        qp_json_file_com = os.path.join(json_dir, qp_json_file_com)
        qp_com = AcadosOcpQp.from_json(qp_json_file_com)

        casadi_solver_com = AcadosCasadiOcpQpSolver(qp_com)
        status = casadi_solver_com.solve()
        casadi_u_com = np.array([casadi_solver_com.get(i, "u") for i in range(qp_com.N)])
        casadi_x_com = np.array([casadi_solver_com.get(i, "x") for i in range(qp_com.N+1)])
        casadi_su_com = np.concatenate([casadi_solver_com.get(i, "su") for i in range(qp_com.N+1)])
        casadi_sl_com = np.concatenate([casadi_solver_com.get(i, "sl") for i in range(qp_com.N+1)])
        casadi_lam_com = np.concatenate([casadi_solver_com.get(i, "lam") for i in range(qp_com.N+1)])
        casadi_pi_com = np.concatenate([casadi_solver_com.get(i, "pi") for i in range(qp_com.N)])
        iterate_casadi_com = casadi_solver_com.get_iterate()
        # casadi_cost_com = casadi_solver_com.get_cost()

        opts = AcadosOcpQpOptions()
        opts.iter_max = 500
        opts.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        acados_solver = AcadosOcpQpSolver(qp, opts=opts)
        status = acados_solver.solve()
        acados_u = np.array([acados_solver.get(i, "u") for i in range(qp.N)])
        acados_x = np.array([acados_solver.get(i, "x") for i in range(qp.N+1)])
        acados_sl = np.concatenate([acados_solver.get(i, "sl") for i in range(qp.N+1)])
        acados_su = np.concatenate([acados_solver.get(i, "su") for i in range(qp.N+1)])
        acados_lam = np.concatenate([acados_solver.get(i, "lam") for i in range(qp.N+1)])
        acados_pi = np.concatenate([acados_solver.get(i, "pi") for i in range(qp.N)])
        iterate_acados = acados_solver.get_iterate()
        # acados_cost = acados_solver.get_cost()

        acados_solver_com = AcadosOcpQpSolver(qp_com, opts=opts)
        status = acados_solver_com.solve()
        acados_u_com = np.array([acados_solver_com.get(i, "u") for i in range(qp_com.N)])
        acados_x_com = np.array([acados_solver_com.get(i, "x") for i in range(qp_com.N+1)])
        acados_sl_com = np.concatenate([acados_solver_com.get(i, "sl") for i in range(qp_com.N+1)])
        acados_su_com = np.concatenate([acados_solver_com.get(i, "su") for i in range(qp_com.N+1)])
        acados_lam_com = np.concatenate([acados_solver_com.get(i, "lam") for i in range(qp_com.N+1)])
        acados_pi_com = np.concatenate([acados_solver_com.get(i, "pi") for i in range(qp_com.N)])
        iterate_acados_com = acados_solver_com.get_iterate()
        # acados_cost_com = acados_solver_com.get_cost()

        print("-------casadi_no_lg_mask and acados_no_lg_mask-------")
        print(f"diff in u matches for {json_name} with error {np.max(np.abs(casadi_u - acados_u))}")
        print(f"diff in x matches for {json_name} with error {np.max(np.abs(casadi_x - acados_x))}")
        print(f"diff in lam matches for {json_name} with error {np.max(np.abs(casadi_lam - acados_lam))}")
        print(f"diff in pi matches for {json_name} with error {np.max(np.abs(casadi_pi - acados_pi))}")
        if casadi_sl.shape[0] != 0 and acados_sl.shape[0] != 0:
            print(f"diff in sl matches for {json_name} with error {np.max(np.abs(casadi_sl - acados_sl))}")
            print(f"diff in su matches for {json_name} with error {np.max(np.abs(casadi_su - acados_su))}")
        # assert np.isclose(casadi_cost, acados_cost, atol=1e-4), f"cost mismatch for {json_name}")

        print("-------casadi_lg_mask and acados_no_lg_mask-------")
        print(f"diff in u matches for {json_name} with error {np.max(np.abs(casadi_u_com - acados_u))}")
        print(f"diff in x matches for {json_name} with error {np.max(np.abs(casadi_x_com - acados_x))}")
        print(f"diff in lam matches for {json_name} with error {np.max(np.abs(casadi_lam_com - acados_lam))}")
        print(f"diff in pi matches for {json_name} with error {np.max(np.abs(casadi_pi_com - acados_pi))}")
        if casadi_sl_com.shape[0] != 0 and acados_sl.shape[0] != 0:
            print(f"diff in sl matches for {json_name} with error {np.max(np.abs(casadi_sl_com - acados_sl))}")
            print(f"diff in su matches for {json_name} with error {np.max(np.abs(casadi_su_com - acados_su))}")
        # assert np.isclose(casadi_cost_com, acados_cost, atol=1e-4), f"cost mismatch for {json_name}")

        print("-------acados_lg_mask and acados_no_lg_mask-------")
        print(f"diff in u matches for {json_name} with error {np.max(np.abs(acados_u_com - acados_u))}")
        print(f"diff in x matches for {json_name} with error {np.max(np.abs(acados_x_com - acados_x))}")
        print(f"diff in lam matches for {json_name} with error {np.max(np.abs(acados_lam_com - acados_lam))}")
        print(f"diff in pi matches for {json_name} with error {np.max(np.abs(acados_pi_com - acados_pi))}")
        if acados_sl_com.shape[0] != 0 and acados_sl.shape[0] != 0:
            print(f"diff in sl matches for {json_name} with error {np.max(np.abs(acados_sl_com - acados_sl))}")
            print(f"diff in su matches for {json_name} with error {np.max(np.abs(acados_su_com - acados_su))}")
        # assert np.isclose(acados_cost_com, acados_cost, atol=1e-4), f"cost mismatch for {json_name}")

if __name__ == "__main__":
    main()