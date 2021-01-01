import numpy as np
from scipy.optimize import minimize


def objective(x):
    return x[0]**3 - x[1]**3 - x[0]*x[1] + 2 * x[0]**2  # 不等式约束


def constraint1(x):
    return -x[0]**2 - x[1]**2 + 6.0  # 等式约束


# 使用约束的时候，要转化为形如ax+by>=c的形式,比如不等约束为x^2+y^2<=6, 则要转化为-x^2-y^2+6>=0的形式
def constraint2(x):
    return 2.0 - x[0] * x[1]


# initial guesses
n = 2
x0 = np.zeros(n)
x0[0] = 1
x0[1] = 2


# show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))


# optimize
b = (0.0, 3.0)
bounds = (b, b)
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'eq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': lambda x: x}
con4 = {'type': 'ineq', 'fun': lambda x: x - 3}
cons = ([con1, con2])
# solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
solution = minimize(objective, x0, method='SLSQP', constraints=cons)
x = solution.x


# show final objective
print('Final SSE Objective: ' + str(objective(x)))

# print solution
print('Solution')
print('x1 = ' + str(x[0]))
print('x2 = ' + str(x[1]))
