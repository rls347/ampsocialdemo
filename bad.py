from numpy import *
import matplotlib.pyplot as plt

def lif(T):
    dt = 0.125
    time = arange(0,T+dt,dt)
    time_for_refractory_period=0
    Vs = zeros(len(time))       # potential (V) trace over time
    Rm = 1                      # Rawr! Resistance! (kOhm)
    Cm = 10                     # capacitance (uF)
    tau_m = Rm*Cm               # time constant (msec)
    tau_refractory = 4          # refractory period (msec)
    Vth = 1
    add_pointy_thing = 0.5      # spike (V)
    I = 1.5                     # input current (A)
    for i, t in enumerate(time):
        if t > time_for_refractory_period:
            Vs[i] = Vs[i-1] + (-Vs[i-1] + I*Rm) / tau_m * dt
            if Vs[i] >= Vth:
                Vs[i] += add_pointy_thing
                time_for_refractory_period = t + tau_refractory

    return time, Vs


# Run and plot membrane potential trace
time, Vs = lif(100)
plt.plot(time, Vs); plt.title('Leaky Integrate-and-Fire Example');plt.ylabel('Membrane Potential (V)'); plt.xlabel('Time (msec)'); plt.ylim([0, 2]);
plt.show()

