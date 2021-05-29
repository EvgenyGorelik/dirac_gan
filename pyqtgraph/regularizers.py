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
