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

def R(theta, phi, alpha_r, alpha_f):
    return (0.0 - alpha_f)**2 + (theta*phi - alpha_r)**2

def dR_phi(theta, phi, alpha_r, alpha_f):
    # alpha_f cancels as true data data distribution is 0
    return 2*phi*(theta**2) - 2*theta*alpha_r


# unregularized loss
def L(theta, phi):
    return f(theta*phi) + f(0)


# derivation of L for theta
def dL_theta(theta, phi):
    return df(theta*phi)*phi

# derivation of L for phi
def dL_phi(theta, phi, alpha_r, alpha_f, lambd=0.99):
    return -1*df(theta*phi)*theta + lambd*dR_phi(theta,phi,alpha_r,alpha_f)

# update step in Alternating Gradient Descent
def AGD_step(theta, phi, h, alpha_r, alpha_f):
    theta -= h * dL_theta(theta,phi)
    phi -= h * dL_phi(theta,phi,alpha_r,alpha_f)
    return theta, phi


history_len = 1000 # how many trajectory points to display

h = 0.5
gamma = 0.90

theta = [np.random.rand()]
phi = [np.random.rand()]

alpha_r = 0.0
alpha_f = 0.0

for i in range(history_len):
    theta_i, phi_i = AGD_step(theta[i],phi[i],h,alpha_r,alpha_f)
    theta.append(theta_i)
    phi.append(phi_i)

    # moving average that tracks the discriminators predictions of the real samples
    # useless as for the dirac gan the discriminators predictions of the real samples always is phi*0=0
    alpha_r = gamma * alpha_r + (1.0 - gamma) * D_phi(0, phi_i)

    # moving average that tracks the discriminators predictions of the generated samples
    alpha_f = gamma * alpha_f + (1.0 - gamma) * D_phi(theta_i, phi_i)

    #print(alpha_r, alpha_f)
    print(theta_i, phi_i)


fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.plot([0, 0], [0, 1], 'k')
line, = ax.plot([], [], ',-', lw=2)
deriv, = ax.plot([], [], ',-', lw=1)
time_template = 'time = %.0f'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

x = np.linspace(-2,2)


def animate(i):
    current_theta, current_phi = theta[i], phi[i]
    y = D_phi(x, current_phi)
    line.set_data([current_theta,current_theta],[0,1])
    deriv.set_data(x,y)
    time_text.set_text(time_template % i)
    return line, deriv, time_text


ani = FuncAnimation(fig, animate, history_len, interval=100, blit=True)
plt.show()
