
import matplotlib.pyplot as plt
import numpy as np

def calc_flux_ratio(packing_ratio, sav_ratio) -> float:
    """_summary_

    Args:
        fuel (Fuel): _description_

    Returns:
        float: _description_
    """

    # rho_b = rho_b
    # rho_p = rho_p
    # sav_ratio = sav_ratio

    # packing_ratio = rho_b / rho_p
    flux_ratio = (192 + 0.2595*sav_ratio)**(-1) * np.exp((0.792 + 0.681*sav_ratio**0.5)*(packing_ratio + 0.1))

    return flux_ratio


sav_ratio = np.linspace(20, 3500, 1000)
packing_ratio = [0.02, 0.01, 0.005, 0.001]


for pr in packing_ratio:
    flux_ratios = [calc_flux_ratio(pr, sav) for sav in sav_ratio]
    plt.plot(sav_ratio, flux_ratios, label=f"Packing Ratio: {pr}")


plt.figure(figsize=(10, 6))
for pr in packing_ratio:
    flux_ratios = [calc_flux_ratio(pr, sav) for sav in sav_ratio]
    plt.plot(sav_ratio, flux_ratios, label=f"Packing Ratio: {pr}")

plt.xlabel("SAV Ratio")
plt.ylabel("Flux Ratio")
plt.title("Flux Ratio vs SAV Ratio for Different Packing Ratios")
plt.legend()
plt.grid(True)
plt.show()

