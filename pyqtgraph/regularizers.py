import numpy as np

class Reg1():
    def __init__(self,h):
        self.h = h

    # f function as defined in paper
    def f(self,t):
        return -np.log(1 + np.exp(-t))

    # derivative of f
    def df(self,t):
        return (1 + np.exp(t)) ** -1

    # canonical discriminator
    def D_phi(self,x, phi):
        return phi * x

    def R1(self,phi):
        return self.h / 2 * phi ** 2

    # for Dirac GAN R1 and R2 regulizers are equivalent, since the derivation does not depend on x
    def dR1_phi(self,phi):
        return self.h * phi

    # unregularized loss
    def L(self,theta, phi):
        return self.f(theta * phi) + self.f(0)

    # derivation of L for theta
    def dL_theta(self,theta, phi):
        return self.df(theta * phi) * phi

    # derivation of L for phi
    def dL_phi(self,theta, phi):
        return self.df(theta * phi) * theta - self.dR1_phi(phi)

    # update step in Alternating Gradient Descent
    def AGD_step(self,theta, phi):
        theta -= self.h * self.dL_theta(theta, phi)
        phi += self.h * self.dL_phi(theta, phi)
        return theta, phi


class No_Reg():
    def __init__(self,h):
        self.h = h

    # f function as defined in paper
    def f(self,t):
        return -np.log(1 + np.exp(-t))

    # derivative of f
    def df(self,t):
        return (1 + np.exp(t)) ** -1

    # canonical discriminator
    def D_phi(self,x, phi):
        return phi * x

    def R(self,phi):
        return self.h / 2 * phi ** 2

    # unregularized loss
    def L(self,theta, phi):
        return self.f(theta * phi) + self.f(0)

    # derivation of L for theta
    def dL_theta(self,theta, phi):
        return self.df(theta * phi) * phi

    # derivation of L for phi
    def dL_phi(self,theta, phi):
        return self.df(theta * phi) * theta

    # update step in Alternating Gradient Descent
    def AGD_step(self,theta, phi):
        theta -= self.h * self.dL_theta(theta, phi)
        phi += self.h * self.dL_phi(theta, phi)
        return theta, phi


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
