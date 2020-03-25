import pandas as pd
import numpy as np
import casadi as ca

class OptimalControl:
    def __init__(self, model, H=30):
        self.model = model
        self.H = H

        # Box-constraints, state
        self.X_lb = np.ones(self.model.x.shape[0]) * -np.inf
        self.X_ub = np.ones(self.model.x.shape[0]) * np.inf

        # Box-constraints, input
        self.U_lb = np.ones(self.model.u.shape[0]) * -np.inf
        self.U_ub = np.ones(self.model.u.shape[0]) * np.inf

    def get_MPC(self):
        opti = ca.Opti()

        # Control trajectory
        U = opti.variable(1, self.H)
        # State trajectory
        X = opti.variable(self.model.x.size, self.H+1)

        # Parameters
        P = opti.parameter(self.model.p.size)
        X0 = opti.parameter(self.model.x.size)

        # Initial conditions
        opti.subject_to(X[:,0] == X0)

        # Get discrete dynamics
        F_RK4 = self.model.get_rk4_single_step()

        # Gap-closing shooting constraints
        for k in range(self.H):
           opti.subject_to(X[:,k+1] == F_RK4(X[:,k], U[:,k], P))


        #opti.subject_to(opti.bounded(0, X[2,:], 0.02))
        #opti.subject_to(opti.bounded(0, U, 0.8))

        opti.subject_to(opti.bounded(self.X_lb, X, self.X_ub))
        opti.subject_to(opti.bounded(self.U_lb, U, self.U_ub))

        opti.minimize(ca.sumsqr(U))

        opti.solver('ipopt', {'ipopt': {'print_level': 0}})

        return opti.to_function("MPC", [X0, P], [U[0]], ["x[k]", "p"], ["u[k]"])