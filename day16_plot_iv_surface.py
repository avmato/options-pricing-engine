from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/SPY_iv_surface_grid_wide.csv"
)

FIGURES_DIRECTORY = Path(
    "figures"
)

FIGURES_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


HEATMAP_PATH = (
    FIGURES_DIRECTORY
    / "SPY_iv_surface_heatmap.png"
)

CONTOUR_PATH = (
    FIGURES_DIRECTORY
    / "SPY_iv_surface_contour.png"
)

SURFACE_3D_PATH = (
    FIGURES_DIRECTORY
    / "SPY_iv_surface_3d.png"
)


surface_wide = pd.read_csv(
    INPUT_PATH,
    index_col=0,
)


surface_wide.index = (
    surface_wide.index.astype(float)
)

surface_wide.columns = (
    surface_wide.columns.astype(float)
)


days_to_expiry_values = (
    surface_wide.index.to_numpy()
)

log_moneyness_values = (
    surface_wide.columns.to_numpy()
)

implied_volatility_values = (
    surface_wide.to_numpy()
)


log_moneyness_grid, days_to_expiry_grid = (
    np.meshgrid(
        log_moneyness_values,
        days_to_expiry_values,
    )
)


print("Surface matrix shape:")
print(
    implied_volatility_values.shape
)

print()
print("Days to expiry values:")
print(
    days_to_expiry_values
)

print()
print("Log-moneyness range:")
print(
    (
        log_moneyness_values.min(),
        log_moneyness_values.max(),
    )
)

print()
print("Missing values in surface matrix:")
print(
    np.isnan(
        implied_volatility_values
    ).sum()
)

print()
print("Implied volatility range:")
print(
    (
        np.nanmin(
            implied_volatility_values
        ),
        np.nanmax(
            implied_volatility_values
        ),
    )
)


plt.figure(
    figsize=(11, 7)
)

plt.imshow(
    implied_volatility_values,
    aspect="auto",
    origin="lower",
    extent=[
        log_moneyness_values.min(),
        log_moneyness_values.max(),
        days_to_expiry_values.min(),
        days_to_expiry_values.max(),
    ],
)

plt.colorbar(
    label="Implied Volatility"
)

plt.axvline(
    x=0.0,
    linestyle="--",
    label="ATM",
)

plt.xlabel(
    "Log-Moneyness: log(K / S)"
)

plt.ylabel(
    "Days to Expiry"
)

plt.title(
    "SPY Implied Volatility Surface Heatmap"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    HEATMAP_PATH,
    dpi=300,
)

plt.show()


plt.figure(
    figsize=(11, 7)
)

contour = plt.contourf(
    log_moneyness_grid,
    days_to_expiry_grid,
    implied_volatility_values,
    levels=20,
)

plt.colorbar(
    contour,
    label="Implied Volatility",
)

plt.axvline(
    x=0.0,
    linestyle="--",
    label="ATM",
)

plt.xlabel(
    "Log-Moneyness: log(K / S)"
)

plt.ylabel(
    "Days to Expiry"
)

plt.title(
    "SPY Implied Volatility Surface Contour"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    CONTOUR_PATH,
    dpi=300,
)

plt.show()


figure = plt.figure(
    figsize=(12, 8)
)

axis = figure.add_subplot(
    111,
    projection="3d",
)

axis.plot_surface(
    log_moneyness_grid,
    days_to_expiry_grid,
    implied_volatility_values,
)

axis.set_xlabel(
    "Log-Moneyness: log(K / S)"
)

axis.set_ylabel(
    "Days to Expiry"
)

axis.set_zlabel(
    "Implied Volatility"
)

axis.set_title(
    "SPY Implied Volatility Surface"
)

figure.tight_layout()

figure.savefig(
    SURFACE_3D_PATH,
    dpi=300,
)

plt.show()


print()
print(
    f"Saved heatmap to: "
    f"{HEATMAP_PATH}"
)

print(
    f"Saved contour plot to: "
    f"{CONTOUR_PATH}"
)

print(
    f"Saved 3D surface to: "
    f"{SURFACE_3D_PATH}"
)