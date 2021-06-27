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
    def __init__(self,h, alpha_r, alpha_f, gamma, lambd):
        self.h = h
        self.alpha_r = alpha_r
        self.alpha_f = alpha_f
        self.gamma = gamma
        self.lambd = lambd

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


class No_Reg_Non_Sat():
    def __init__(self, h):
        self.h = h

    # f function as defined in paper
    def f(self, t):
        return -np.log(1 + np.exp(-t))

    # derivative of f
    def df(self, t):
        return (1 + np.exp(t)) ** -1

    # canonical discriminator
    def D_phi(self, x, phi):
        return phi * x

    def R(self, phi):
        return self.h / 2 * phi ** 2

    # unregularized loss
    def L(self, theta, phi):
        return self.f(theta * phi) + self.f(0)

    # derivation of L for theta
    def dL_theta(self, theta, phi):
        return self.df(theta * phi) * phi

    # derivation of L for phi
    def dL_phi(self, theta, phi):
        return self.df(theta * phi) * theta

    # update step in Alternating Gradient Descent
    def AGD_step(self, theta, phi):
        theta -= self.h * self.dL_theta(-theta, phi)
        phi += self.h * self.dL_phi(theta, phi)
        return theta, phi


class Reg_Cons_Opt():
    def __init__(self, h):
        self.h = h

    # f function as defined in paper
    def f(self, t):
        return -np.log(1 + np.exp(-t))

    # derivative of f
    def df(self, t):
        return (1 + np.exp(t)) ** -1

    # canonical discriminator
    def D_phi(self, x, phi):
        return phi * x

    # see paper: formula 101
    def R(self, theta, phi):
        # return self.h / 2 * (self.dL_theta(theta, phi) ** 2 + self.dL_phi(theta, phi) ** 2)
        return self.h / 2 * (((self.df(theta * phi) * phi) ** 2) + (self.df(theta * phi) * theta) ** 2)

    # gradient for theta (by hand)
    def dR_theta(self, theta, phi):
        # gradient: -(h * theta * ((theta * phi - 1) * np.exp(theta * phi) - 1)
        #            + h * (phi ** 3) * np.exp(theta * phi)) / ((np.exp(theta * phi)+1) ** 3)
        return -(theta * ((theta * phi - 1) * np.exp(theta * phi) - 1)
                 + (phi ** 3) * np.exp(theta * phi)) / ((np.exp(theta * phi) + 1) ** 3)

    # gradient for phi (by hand)
    def dR_phi(self, theta, phi):
        # gradient: -(h * phi * ((theta * phi - 1) * np.exp(theta * phi) - 1)
        #            + h * (theta ** 3) * np.exp(theta * phi)) / ((np.exp(theta * phi)+1) ** 3)
        return -(phi * ((theta * phi - 1) * np.exp(theta * phi) - 1)
                 + (theta ** 3) * np.exp(theta * phi)) / ((np.exp(theta * phi) + 1) ** 3)

    # unregularized loss
    def L(self, theta, phi):
        return self.f(theta * phi) + self.f(0)

    # derivation of L for theta
    def dL_theta(self, theta, phi):
        return self.df(theta * phi) * phi - self.dR_theta(theta, phi)

    # derivation of L for phi
    def dL_phi(self, theta, phi):
        return self.df(theta * phi) * theta - self.dR_phi(theta, phi)

    # update step in Alternating Gradient Descent
    def AGD_step(self, theta, phi):
        theta -= self.h * self.dL_theta(theta, phi)
        phi += self.h * self.dL_phi(theta, phi)
        return theta, phi


class Reg_Inst_Noise():
    def __init__(self, h):
        self.h = h

    # f function as defined in paper
    def f(self, t):
        return -np.log(1 + np.exp(-t))

    # derivative of f
    def df(self, t):
        return (1 + np.exp(t)) ** -1

    # canonical discriminator
    def D_phi(self, x, phi):
        return phi * x

    # unregularized loss
    def L(self, theta, phi, noise):
        return self.f((theta + noise) * phi) + self.f(0)

    # derivation of L for theta
    def dL_theta(self, theta, phi, noise):
        return self.df((theta + noise) * phi) * phi  # ... * -phi

    # derivation of L for phi
    def dL_phi(self, theta, phi, t_noise, x_noise):
        return self.df((theta + t_noise) * phi) * (theta + t_noise) - self.df(-x_noise * phi) * x_noise

    # update step in Alternating Gradient Descent
    def AGD_step(self, theta, phi, std=1):
        t_noise = np.random.normal(scale=std, size=1000)
        x_noise = np.random.normal(scale=std, size=1000)
        theta = np.full_like(t_noise, theta)
        phi_vec = np.full_like(theta, phi)
        theta -= self.h * self.dL_theta(theta, phi, t_noise)
        phi_vec += self.h * self.dL_phi(theta, phi, t_noise, x_noise)
        return np.mean(theta), np.mean(phi_vec)