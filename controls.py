import pandas as pd
import numpy as np
import casadi as ca

class OptimalControl:
    def __init__(self, model, H=30):
        self.model = model
        self.H = H

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

        # Capacity
        # TODO: Move this from here
        opti.subject_to(opti.bounded(0, X[2,:], 0.02))
        opti.subject_to(opti.bounded(0, U, 0.8))

        opti.minimize(ca.sumsqr(U))

        opti.solver('ipopt')

        return opti.to_function("MPC", [X0, P], [U[0]], ["x[k]", "p"], ["u[k]"])