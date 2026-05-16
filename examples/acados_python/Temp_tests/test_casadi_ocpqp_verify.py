import os
from acados_template import AcadosOcpQpSolver, AcadosCasadiOcpQpSolver, AcadosOcpQp, AcadosOcpQpOptions
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.abspath(os.path.join(script_dir, 'qp_tests'))
qp_json_files = [('drone_FrenSer_QP.json', 'drone_FrenSer_QP_com.json'),]
def main():
    for qp_json_file, qp_json_file_com in qp_json_files:

        json_name = os.path.basename(qp_json_file_com)
        qp_json_file_com = os.path.join(json_dir, qp_json_file_com)
        qp = AcadosOcpQp.from_json(qp_json_file_com)

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

        opts = AcadosOcpQpOptions()
        opts.iter_max = 500
        opts.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        acados_hpipm_solver = AcadosOcpQpSolver(qp, opts=opts)
        status = acados_hpipm_solver.solve()
        acados_hpipm_u = np.array([acados_hpipm_solver.get(i, "u") for i in range(qp.N)])
        acados_hpipm_x = np.array([acados_hpipm_solver.get(i, "x") for i in range(qp.N+1)])
        acados_hpipm_sl = np.concatenate([acados_hpipm_solver.get(i, "sl") for i in range(qp.N+1)])
        acados_hpipm_su = np.concatenate([acados_hpipm_solver.get(i, "su") for i in range(qp.N+1)])
        acados_hpipm_lam = np.concatenate([acados_hpipm_solver.get(i, "lam") for i in range(qp.N+1)])
        acados_hpipm_pi = np.concatenate([acados_hpipm_solver.get(i, "pi") for i in range(qp.N)])
        iterate_acados = acados_hpipm_solver.get_iterate()
        # acados_cost = acados_solver.get_cost()

        opts = AcadosOcpQpOptions()
        opts.iter_max = 500
        opts.qp_solver = 'FULL_CONDENSING_QPOASES'
        acados_qpoases_solver = AcadosOcpQpSolver(qp, opts=opts)
        status = acados_qpoases_solver.solve()
        acados_qpoases_u = np.array([acados_qpoases_solver.get(i, "u") for i in range(qp.N)])
        acados_qpoases_x = np.array([acados_qpoases_solver.get(i, "x") for i in range(qp.N+1)])
        acados_qpoases_sl = np.concatenate([acados_qpoases_solver.get(i, "sl") for i in range(qp.N+1)])
        acados_qpoases_su = np.concatenate([acados_qpoases_solver.get(i, "su") for i in range(qp.N+1)])
        acados_qpoases_lam = np.concatenate([acados_qpoases_solver.get(i, "lam") for i in range(qp.N+1)])
        acados_qpoases_pi = np.concatenate([acados_qpoases_solver.get(i, "pi") for i in range(qp.N)])
        iterate_acados_qpoases = acados_qpoases_solver.get_iterate()
        # acados_cost = acados_solver.get_cost()

        print("-------ipopt and hpipm-------")
        print(f"diff in u matches for {json_name} with error {np.max(np.abs(casadi_u - acados_hpipm_u))}")
        print(f"diff in x matches for {json_name} with error {np.max(np.abs(casadi_x - acados_hpipm_x))}")
        print(f"diff in lam matches for {json_name} with error {np.max(np.abs(casadi_lam - acados_hpipm_lam))}")
        print(f"diff in pi matches for {json_name} with error {np.max(np.abs(casadi_pi - acados_hpipm_pi))}")
        if casadi_sl.shape[0] != 0 and acados_hpipm_sl.shape[0] != 0:
            print(f"diff in sl matches for {json_name} with error {np.max(np.abs(casadi_sl - acados_hpipm_sl))}")
            print(f"diff in su matches for {json_name} with error {np.max(np.abs(casadi_su - acados_hpipm_su))}")
        # assert np.isclose(casadi_cost, acados_cost, atol=1e-4), f"cost mismatch for {json_name}")

        print("-------hpipm and qpoases-------")
        print(f"diff in u matches for {json_name} with error {np.max(np.abs(acados_hpipm_u - acados_qpoases_u))}")
        print(f"diff in x matches for {json_name} with error {np.max(np.abs(acados_hpipm_x - acados_qpoases_x))}")
        print(f"diff in lam matches for {json_name} with error {np.max(np.abs(acados_hpipm_lam - acados_qpoases_lam))}")
        print(f"diff in pi matches for {json_name} with error {np.max(np.abs(acados_hpipm_pi - acados_qpoases_pi))}")
        if acados_hpipm_sl.shape[0] != 0 and acados_qpoases_sl.shape[0] != 0:
            print(f"diff in sl matches for {json_name} with error {np.max(np.abs(acados_hpipm_sl - acados_qpoases_sl))}")
            print(f"diff in su matches for {json_name} with error {np.max(np.abs(acados_hpipm_su - acados_qpoases_su))}")
        # assert np.isclose(acados_hpipm_cost, acados_qpoases_cost, atol=1e-4), f"cost mismatch for {json_name}")

        print("-------ipopt and qpoases-------")
        print(f"diff in u matches for {json_name} with error {np.max(np.abs(casadi_u - acados_qpoases_u))}")
        print(f"diff in x matches for {json_name} with error {np.max(np.abs(casadi_x - acados_qpoases_x))}")
        print(f"diff in lam matches for {json_name} with error {np.max(np.abs(casadi_lam - acados_qpoases_lam))}")
        print(f"diff in pi matches for {json_name} with error {np.max(np.abs(casadi_pi - acados_qpoases_pi))}")
        if casadi_sl.shape[0] != 0 and acados_qpoases_sl.shape[0] != 0:
            print(f"diff in sl matches for {json_name} with error {np.max(np.abs(casadi_sl - acados_qpoases_sl))}")
            print(f"diff in su matches for {json_name} with error {np.max(np.abs(casadi_su - acados_qpoases_su))}")
        # assert np.isclose(casadi_cost, acados_qpoases_cost, atol=1e-4), f"cost mismatch for {json_name}")

if __name__ == "__main__":
    main()