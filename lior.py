import numpy as np
import matplotlib.pyplot as plt

# Constants
k_B = 1.380658e-23  # Boltzmann constant in joule/K
q = 1.60217733e-19  # Electron charge in coulombs
No = 5e12  # Density (per cm^2) of initial Si-H bonds at the interface
Beta = 1e17  # Prefactor for A
F_acc = 0.6  # Field acceleration factor
n_NBTI = 0.167  # NBTI time exponent
n_HCI = 0.45  # HCI time exponent
EA = 0.55  # Activation energy for diffusion (eV)
Tox = 2.0e-7  # Oxide thickness in cm
Tcorr = 1e-7  # Correction due to inversion layer thickness
Vt = 0.3  # Threshold voltage in volts
Vacpt = 0.1  # Acceptable bond breakage fraction
voltage = 1.2  # Operating voltage in volts

# Transistor dimensions
Length = 0.1e-4  # Effective channel length in cm
Width = 0.02e-4  # Width of the channel in cm
NoDev = No * Length * Width  # Average Si-H bonds in a given device

# Time range
time = np.logspace(0, 8, 100)  # From 1 second to 10^8 seconds

# Temperatures (Kelvin)
temps = [80,90,100,110, 120,130]

# Initialize plot
plt.figure(figsize=(10, 6))

# Plot combined models for each temperature
for T in temps:
    # Thermal voltage
    VThermal = (k_B / q) * T  # in volts

    # Oxide electric field
    Eox = (voltage - Vt) / (Tox + Tcorr) / 1e6  # in MV/cm

    # Calculate A using field acceleration and Arrhenius model
    A = Beta * (Eox**(2/3)) * np.exp(F_acc * 2 * Eox / 3) * np.exp(-EA / VThermal)

    # NBTI model: Delta Vth ~ A * t^n
    DeltaVth_NBTI = A * time**n_NBTI

    # HCI model: Delta Vth ~ A * t^n
    DeltaVth_HCI = A * time**n_HCI

    # Combined degradation
    DeltaVth_Combined = DeltaVth_NBTI + DeltaVth_HCI

    # Plot combined model
    plt.plot(time, DeltaVth_Combined, label=f'Combined @ T={T}K')

# Customize plot
plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel(r'$\Delta V_{th}$ (V)')
plt.title('Combined NBTI + HCI Degradation at Different Temperatures')
plt.legend()
plt.grid(True)
plt.show()
