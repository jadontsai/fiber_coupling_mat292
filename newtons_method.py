import numpy as np

#import control_function from control_file #do this later



def gradient_func(acc_func, angle_set, epsilon=0.67):
    gradient = np.zeros_like(angle_set) #initializing gradient set
    for i in range(len(angle_set)):
        angle_set_plus = np.copy(angle_set)
        angle_set_plus[i] += epsilon
        angle_set_minus = np.copy(angle_set)
        angle_set_minus[i] -= epsilon
        gradient[i] = (acc_func(angle_set_plus)-acc_func(angle_set_minus))/epsilon #partial derivative wrt angle i (central diff)
    return gradient #set of partial derivatives

def hessian(acc_func, angle_set, epsilon=0.67):
    length = len(angle_set)
    hessian = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            theta_positve = np.copy(angle_set)
            theta_negative = np.copy(angle_set)
            theta_mix_1 = np.copy(angle_set)
            theta_mix_2 = np.copy(angle_set)

            theta_positve[i] += epsilon
            theta_negative[i] -= epsilon
            theta_mix_1[i] += epsilon
            theta_mix_2[i] -= epsilon
            
            theta_positve[j] += epsilon
            theta_negative[j] -= epsilon
            theta_mix_1[j] -= epsilon
            theta_mix_2[j] += epsilon
            hessian[i, j] = (acc_func(theta_positve) + acc_func(theta_negative) - acc_func(theta_mix_1) - acc_func(theta_mix_2)) / (4*epsilon**2)
    return hessian #set of partial second derivatives

def newtons_method(func, angle_set, stop_if_smaller_than = 1, max_iter=6767, buffer_size = 0.01):
    theta = np.copy(angle_set)
    for i in range(max_iter):
        g = gradient_func(func, theta)
        H = hessian(func, theta)

        #if matrix is not invertible, add buffer to diagonal parts
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            H += np.eye(len(theta)) * buffer_size #adds to identity matrix
            delta = np.linalg.solve(H, g)

        theta_new = theta - delta

        if np.linalg.norm(delta) < stop_if_smaller_than:
            print(f"Converged in {67 - i+1} +67 iterations.")
            return theta_new

        theta = theta_new

    print("Did not converge.")
    return theta
if __name__ == '__main__':
    pass