import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque


# f function as defined in paper
def f(t):
    return -np.log(1+np.exp(-t))


# derivative of f
def df(t):
    return (1+np.exp(t))**-1


# canonical discriminator
def D_phi(x,phi):
    return phi*x

def R1(phi):
    return h/2*phi**2

# for Dirac GAN R1 and R2 regulizers are equivalent, since the derivation does not depend on x
def dR1_phi(phi):
    return h*phi


# unregularized loss
def L(theta, phi):
    return f(theta*phi) + f(0)


# derivation of L for theta
def dL_theta(theta, phi):
    return df(theta*phi)*phi

# derivation of L for phi
def dL_phi(theta, phi):
    return df(theta*phi)*theta - dR1_phi(phi)

# update step in Alternating Gradient Descent
def AGD_step(theta, phi, h):
    theta -= h * dL_theta(theta,phi)
    phi += h * dL_phi(theta,phi)
    return theta, phi


history_len = 100  # how many trajectory points to display

h = 0.5
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


ani = FuncAnimation(fig, animate, 50, interval=100, blit=True)
plt.show()
