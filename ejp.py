import numpy as np
import matplotlib.pyplot as plt


def lif(T):
    """A leaky integrate and fire model neuron.

    Params
    ------
    T : scalar
        The number of time steps to simulate.

    Returns
    -------
    time : array-like
        Time steps
    voltages : array-like
        The voltage at each step
    """

    # -- Initialize
    # Timing
    dt = 0.125                          # Time step
    times = np.arange(0, T + dt, dt)    # A list of all the steps
    refrac_remaining = 0                # How much time remains in refrac

    # Neuron properties
    Vs = np.zeros(len(times))       # potential (V) trace over time
    Rm = 1                          # rawr! resistance! (kOhm)
    Cm = 10                         # capacitance (uF)
    tau_m = Rm * Cm                 # time constant (msec)
    tau_refrac = 4                  # refractory period (msec)
    Vth = 1                         # spike threshold (V)
    spike = 0.5                     # spike (V)
    I = 1.5                         # input current (A)

    # -- Run model
    for i, t in enumerate(times):
        # Not it a refrac period? Update the neuron.
        if t > refrac_remaining:
            # Update Vs using Euler intergration
            Vs[i] = Vs[i - 1] + (-Vs[i - 1] + I * Rm) / tau_m * dt

            # Should it spike?
            if Vs[i] >= Vth:
                Vs[i] += spike
                refrac_remaining = t + tau_refrac

    return times, Vs


def test_lif():
    """Compare lif(1) to known good values"""

    times, test_Vs = lif(1)

    # Known good values from a short run
    good_Vs = np.array([
        0.0, 0.01875, 0.03726562, 0.0555498, 0.07360543,
        0.09143536,  0.10904242, 0.12642939, 0.14359902
    ])

    # compare good and test, using allclose() to take
    # care of floating point inconsistencies.
    assert np.allclose(test_Vs, good_Vs), "Not consistent!"


if __name__ == "__main__":
    # Run and plot membrane potential trace
    times, Vs = lif(100)

    plt.plot(times, Vs)
    plt.title('Leaky Integrate-and-Fire Example')
    plt.ylabel('Membrane Potential (V)')
    plt.xlabel('Time (msec)')
    plt.ylim([0, 2])
    plt.show()
