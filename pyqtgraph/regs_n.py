# -*- coding: utf-8 -*-
import numpy as np

""" 
Should be the same as No_Reg / standard GAN, except for "-theta" in the 
theta update.
"""
class No_Reg_Non_Sat():
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
        theta -= self.h * self.dL_theta(-theta, phi)
        phi += self.h * self.dL_phi(theta, phi)
        return theta, phi

"""
Learning rate in R wrong? (already in AGD - remove in AGD?)
"""
class Reg_Cons_Opt():
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

    # see paper: formula 101
    def R(self,theta,phi):
        #return self.h / 2 * (self.dL_theta(theta, phi) ** 2 + self.dL_phi(theta, phi) ** 2)
        return self.h / 2 * (((self.df(theta * phi) * phi) ** 2) + (self.df(theta * phi) * theta) ** 2)

    # gradient for theta (by hand)
    def dR_theta(self,theta,phi):
        return -(self.h * theta * ((theta * phi - 1) * np.exp(theta * phi) - 1) 
                 + self.h * (phi ** 3) * np.exp(theta * phi)) / ((np.exp(theta * phi)+1) ** 3)

    # gradient for phi (by hand)
    def dR_phi(self,theta,phi):
        return -(self.h * phi * ((theta * phi - 1) * np.exp(theta * phi) - 1) 
                 + self.h * (theta ** 3) * np.exp(theta * phi)) / ((np.exp(theta * phi)+1) ** 3)

    # unregularized loss
    def L(self,theta, phi):
        return self.f(theta * phi) + self.f(0)

    # derivation of L for theta
    def dL_theta(self,theta, phi):
        return self.df(theta * phi) * phi - self.dR_theta(theta, phi)

    # derivation of L for phi
    def dL_phi(self,theta, phi):
        return self.df(theta * phi) * theta - self.dR_phi(theta, phi)

    # update step in Alternating Gradient Descent
    def AGD_step(self,theta, phi):
        theta -= self.h * self.dL_theta(theta, phi)
        phi += self.h * self.dL_phi(theta, phi)
        return theta, phi


"""
NOT WORKING YET.
Not sure if learning rate is included in R.
"""
class Reg_Inst_Noise():
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

    """ Not needed here?
    def R(self,theta):
        return self.h * np.random.normal() * theta

    # 
    def dR_phi(self,phi):
        return self.h * np.random.normal()
    """

    # unregularized loss
    def L(self,theta, phi, noise):
        return self.f((theta + noise) * phi) + self.f(0)

    # derivation of L for theta
    def dL_theta(self,theta, phi, noise):
        return self.df((theta + noise) * phi) * phi # ... * -phi

    # derivation of L for phi
    def dL_phi(self,theta, phi, t_noise, x_noise):
        #return self.df(theta * phi) * theta + self.df(noise * phi) * noise
        return self.df((theta + t_noise) * phi) * (theta + t_noise) - self.df(-x_noise * phi) * x_noise

    # update step in Alternating Gradient Descent
    def AGD_step(self,theta, phi, std=0.3):
        t_noise = std * np.random.normal()
        x_noise = std * np.random.normal()
        theta -= self.h * self.dL_theta(theta, phi, t_noise)
        phi += self.h * self.dL_phi(theta, phi, t_noise, x_noise)
        return theta, phi