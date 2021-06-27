import numpy as np

def f(t):
    return -np.log(1 + np.exp(-t))

def df(t):
    return (1 + np.exp(t)) ** -1

class Reg1():
    def __init__(self, h, reg):
        self.h = h
        self.reg = reg

    # update step in Alternating Gradient Descent
    def AGD_step(self, theta, psi):
        psi += self.h * (f(theta*psi)*theta - self.reg * psi)
        theta -= self.h * (f(theta*psi)*psi)
        return theta, psi

    # update step in Alternating Gradient Descent
    def SGD_step(self, theta, psi):
        psi += self.h * (f(theta*psi)*theta - self.reg * psi)
        theta -= self.h * (f(theta*psi)*psi)
        return theta, psi


class No_Reg():
    def __init__(self, h):
        self.h = h

    # update step in Alternating Gradient Descent
    def AGD_step(self,theta, psi):
        psi += self.h * theta * f(theta*psi)
        theta -= self.h * psi * f(psi*theta)
        return theta, psi

    # update step in Alternating Gradient Descent
    def SGD_step(self,theta, psi):
        psi += self.h * theta * f(theta*psi)
        theta -= self.h * psi * f(psi*theta)
        return theta, psi

class WGAN_Reg():
    def __init__(self,h,n_critic,c):
        self.h = h
        self.n_critic = n_critic
        self.c = c

    def clip(self, x):
        return np.clip(x, a_max=self.c, a_min=-self.c)

    # update step in Alternating Gradient Descent
    def AGD_step(self, theta, psi):
        for i in range(self.n_critic):
            psi += self.h * theta * f(theta*psi)
        theta -= self.h * psi * f(theta*psi)
        return self.clip(theta), self.clip(psi)

    # update step in Alternating Gradient Descent
    def SGD_step(self, theta, psi):
        for i in range(self.n_critic):
            psi += self.h * theta * f(theta*psi)
        theta -= self.h * psi * f(theta*psi)
        return self.clip(theta), self.clip(psi)


class WGAN_GP_reg():
    def __init__(self, h, n_critic, gamma, g_0):
        self.h = h
        self.n_critic = n_critic
        self.gamma = gamma
        self.g_0 = g_0

    # update step in Alternating Gradient Descent
    def AGD_step(self, theta, psi):
        for i in range(self.n_critic):
            psi += self.h * (theta - np.sign(psi) * self.gamma * (np.abs(psi) - self.g_0))
        theta -= self.h * psi
        return theta, psi


    # update step in Alternating Gradient Descent
    def SGD_step(self, theta, psi):
        psi_old = psi*1
        for i in range(self.n_critic):
            psi += self.h * (theta - np.sign(psi) * self.gamma * (np.abs(psi) - self.g_0))
        theta -= self.h * psi_old
        return theta, psi

class Moving_Average_Reg():
    def __init__(self,h):
        self.h = h
        self.alpha_r = 0.0
        self.alpha_f = 0.0
        self.gamma = 0.99
        self.lambd = 0.99

    def f(self,t):
        return -np.log(1 + np.exp(-t))

    # derivative of f
    def df(self,t):
        return (1 + np.exp(t)) ** -1

    # canonical discriminator
    def D_phi(self,x, phi):
        return phi * x

    def R(self,theta, phi):
        return (0.0 - self.alpha_f) ** 2 + (theta * phi - self.alpha_r) ** 2

    def dR_phi(self,theta, phi):
        # alpha_f cancels as true data data distribution is 0
        return 2 * phi * (theta ** 2) - 2 * theta * self.alpha_r

    # unregularized loss
    def L(self,theta, phi):
        return self.f(theta * phi) + self.f(0)

    # derivation of L for theta
    def dL_theta(self,theta, phi):
        return self.df(theta * phi) * phi

    # derivation of L for phi
    def dL_phi(self,theta, phi):
        return -1 * self.df(theta * phi) * theta + self.lambd * self.dR_phi(theta, phi)

    # update step in Alternating Gradient Descent
    def AGD_step(self,theta, phi):
        theta -= self.h * self.dL_theta(theta, phi)
        phi -= self.h * self.dL_phi(theta, phi)

        # moving average that tracks the discriminators predictions of the real samples
        # useless as for the dirac gan the discriminators predictions of the real samples always is phi*0=0
        self.alpha_r = self.gamma * self.alpha_r + (1.0 - self.gamma) * self.D_phi(0, phi)
        # moving average that tracks the discriminators predictions of the generated samples
        self.alpha_f = self.gamma * self.alpha_f + (1.0 - self.gamma) * self.D_phi(theta, phi)

        return theta, phi
