import casadi as ca
import casadi.tools


class SEIR:
    def __init__(self):

        # Social distancing control
        u = ca.MX.sym("u")

        # State
        x = ca.tools.struct_symMX(["s", "e", "i", "r"])

        # Parameters
        p = ca.tools.struct_symMX(["alpha", "beta", "gamma"])

        dxdt = ca.tools.struct_MX(x)
        dxdt["s"] = -(1 - u) * p["beta"] * x["s"] * x["i"]
        dxdt["e"] = (1 - u) * p["beta"] * x["s"] * x["i"] - p["alpha"] * x["e"]
        dxdt["i"] = p["alpha"] * x["e"] - p["gamma"] * x["i"]
        dxdt["r"] = p["gamma"] * x["i"]

        self.f = ca.Function("f", [x, u, p], [dxdt.cat], ["x", "u", "p"], ["dx/dt"])
        self.x = x
        self.u = u
        self.p = p

    def get_rk4_single_step(self, dt=1):
        # RK4
        k1 = self.f(self.x, self.u, self.p)
        k2 = self.f(self.x + dt / 2.0 * k1, self.u, self.p)
        k3 = self.f(self.x + dt / 2.0 * k2, self.u, self.p)
        k4 = self.f(self.x + dt * k3, self.u, self.p)
        xf = self.x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Single step time propagation
        return ca.Function("F_RK4", 
            [self.x, self.u, self.p], [xf], ['x[k]', 'u[k]', "p"], ['x[k+1]']
        )