import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque


# f function as defined in paper
def f(t):
    return -np.log(1 + np.exp(-t))

# derivative of f
def df(t):
    return (1 + np.exp(t)) ** -1

# canonical discriminator
def D_phi(x, phi):
    return phi * x

# see paper: formula 101
def R(theta,phi):
    #return self.h / 2 * (self.dL_theta(theta, phi) ** 2 + self.dL_phi(theta, phi) ** 2)
    return h / 2 * (((df(theta * phi) * phi) ** 2) + (df(theta * phi) * theta) ** 2)

# gradient for theta (by hand)
def dR_theta(theta,phi):
    return -(h * theta * ((theta * phi - 1) * np.exp(theta * phi) - 1) 
             + h * (phi ** 3) * np.exp(theta * phi)) / ((np.exp(theta * phi)+1) ** 3)

# gradient for phi (by hand)
def dR_phi(theta,phi):
    return -(h * phi * ((theta * phi - 1) * np.exp(theta * phi) - 1) 
             + h * (theta ** 3) * np.exp(theta * phi)) / ((np.exp(theta * phi)+1) ** 3)

# unregularized loss
def L(theta, phi):
    return f(theta * phi) + f(0)

# derivation of L for theta
def dL_theta(theta, phi):
    return df(theta * phi) * phi - dR_theta(theta, phi)

# derivation of L for phi
def dL_phi(theta, phi):
    return df(theta * phi) * theta - dR_phi(theta, phi)

# update step in Alternating Gradient Descent
def AGD_step(theta, phi, h):
    theta -= h * dL_theta(theta, phi)
    phi += h * dL_phi(theta, phi)
    return theta, phi


history_len = 100  # how many trajectory points to display

h = 1.2 #1.4
theta = [np.random.rand()]
phi = [np.random.rand()]

for i in range(history_len):
    theta_i, phi_i = AGD_step(theta[i],phi[i],h)
    theta.append(theta_i)
    phi.append(phi_i)

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('\u03B8')
ax.set_ylabel('\u03C6')

ax.scatter([0],[0],c='k')
line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], ',-', lw=1)
time_template = 'time = %.1f'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)


def animate(i):
    current_theta, current_phi = theta[i], phi[i]

    if i == 0:
        history_x.clear()
        history_y.clear()

    history_x.append(current_theta)
    history_y.append(current_phi)

    line.set_data(current_theta, current_phi)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % i)
    return line, trace, time_text


ani = FuncAnimation(fig, animate, 100, interval=100, blit=True)
plt.show()
