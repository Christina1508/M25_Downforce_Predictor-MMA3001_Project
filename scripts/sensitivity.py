"""Aerodynamic sensitivity analysis of the M25 downforce models.

Uses second-order centred finite differences to estimate the sensitivity
of predicted downforce area to airspeed and yaw angle.

Compares quadratic regression and Gaussian RBF predictions across the
experimental operating range and verifies finite-difference step sizes.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif"
})

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR/"figures"

# -------- Part 10 - Aerodynamic Sensitivity Analysis

def run_sensitivity(predict_model, x_train):
    """Calculate aerodynamic sensitivities using centred finite differences.

    Evaluates quadratic and Gaussian RBF predictions across a velocity-yaw
    grid within the experimental range. Calculates partial derivatives,
    generates contour plots and compares results using halved step sizes.

    Args:
        predict_model (callable): Function returning CLA predictions for
            a model name and an array of input conditions.
        x_train (np.ndarray): Training inputs containing airspeed (m/s)
            and yaw angle (deg), shape (n, 2).

    Returns:
        dict: Predicted CLA (m^2), velocity sensitivity (m^2/(m/s))
            and yaw sensitivity (m^2/deg) for each model.
    """
    # Finite difference step sizes
    h_v = 0.1       # m/s
    h_yaw = 0.25    # deg 

    # Prediction range within experimental conditions
    v_min = x_train[:, 0].min()
    v_max = x_train[:, 0].max()
    yaw_min = x_train[:, 1].min()
    yaw_max = x_train[:, 1].max()

    # Exclude boundaries for centred differences
    velocities = np.linspace(v_min + h_v, v_max - h_v, 80)
    yaws = np.linspace(yaw_min + h_yaw, yaw_max - h_yaw, 100)

    V, yaw = np.meshgrid(velocities, yaws)


    # Predict CLA across velocity-yaw grid
    def predict_surface(model, V, yaw):
        """Evaluate model predictions over a velocity-yaw grid.

        Args:
            model (str): Name of the regression model.
            V (np.ndarray): Grid of airspeeds (m/s).
            yaw (np.ndarray): Grid of yaw angles (deg).

        Returns:
            np.ndarray: Predicted CLA values (m^2), shaped like V.
        """
        X = np.column_stack((V.ravel(), yaw.ravel()))

        return predict_model(model, X).reshape(V.shape)


    # Calculate partial derivatives using centred finite differences
    def calculate_sensitivity(model, V, yaw, hv, hyaw):
        """Calculate partial derivatives using centred finite differences.

        Args:
            model (str): Name of the regression model.
            V (np.ndarray): Airspeed grid (m/s).
            yaw (np.ndarray): Yaw angle grid (deg).
            hv (float): Velocity perturbation (m/s).
            hyaw (float): Yaw perturbation (deg).

        Returns:
            tuple[np.ndarray, np.ndarray]: Derivatives of CLA with
                respect to velocity (m^2/(m/s)) and yaw (m^2/deg).
        """
        dCLA_dV = (
            predict_surface(model, V + hv, yaw)
            - predict_surface(model, V - hv, yaw)
        ) / (2 * hv)

        dCLA_dyaw = (
            predict_surface(model, V, yaw + hyaw)
            - predict_surface(model, V, yaw - hyaw)
        ) / (2 * hyaw)

        return dCLA_dV, dCLA_dyaw


    # Calculate surfaces for both models
    models = ["Quadratic", "Gaussian RBF"]
    sensitivity_results = {}

    for model in models:

        CLA = predict_surface(model, V, yaw)

        dCLA_dV, dCLA_dyaw = calculate_sensitivity(
            model, V, yaw, h_v, h_yaw
        )

        sensitivity_results[model] = {
            "CLA": CLA,
            "dCLA_dV": dCLA_dV,
            "dCLA_dyaw": dCLA_dyaw
        }


    # -------- Plot Predicted CLA and Sensitivity Contours

    fig, axes = plt.subplots(
        2, 3, figsize=(17, 10), dpi=400,
        constrained_layout=True
    )

    fields = ["CLA", "dCLA_dV", "dCLA_dyaw"]

    titles = [
        r"Predicted $C_LA$",
        r"Velocity Sensitivity, $\partial(C_LA)/\partial V$",
        r"Yaw Sensitivity, $\partial(C_LA)/\partial\beta$"
    ]

    labels = [
        r"$C_LA$ ($\mathrm{m^2}$)",
        r"$\partial(C_LA)/\partial V$ ($\mathrm{m^2/(m/s)}$)",
        r"$\partial(C_LA)/\partial\beta$ ($\mathrm{m^2/deg}$)"
    ]

    for j, field in enumerate(fields):

        # Consistent colour scales between models
        values = [sensitivity_results[m][field] for m in models]

        vmin = min(a.min() for a in values)
        vmax = max(a.max() for a in values)

        if j > 0:
            max_abs = max(abs(vmin), abs(vmax))
            vmin, vmax = -max_abs, max_abs

        levels = np.linspace(vmin, vmax, 21)
        cmap = "viridis" if j == 0 else "coolwarm"

        for i, model in enumerate(models):

            ax = axes[i, j]

            contour = ax.contourf(
                V, yaw,
                sensitivity_results[model][field],
                levels=levels,
                cmap=cmap
            )

            # Show original experimental training locations
            if j == 0:
                ax.scatter(
                    x_train[:, 0],
                    x_train[:, 1],
                    c="black",
                    s=12,
                    label="Training Data"
                )
                ax.legend(loc="lower left", fontsize=9)

            ax.set_xlabel(r"Velocity, $V$ ($\mathrm{m/s}$)")
            ax.set_ylabel(r"Yaw Angle, $\beta$ ($^\circ$)")
            ax.set_title(model + "\n" + titles[j])

            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label(labels[j])

    plt.savefig(FIG_DIR/"sensitivities.png")
    plt.show()


    # -------- Sensitivity at Reference Operating Condition

    V_ref = np.array([[17.0]])
    yaw_ref = np.array([[0.0]])

    print("\nAerodynamic Sensitivity at V = 17 m/s, yaw = 0 deg")

    for model in models:

        CLA_ref = predict_surface(model, V_ref, yaw_ref)[0, 0]

        dV, dyaw = calculate_sensitivity(
            model, V_ref, yaw_ref, h_v, h_yaw
        )

        print(f"\n{model}")
        print(f"Predicted CLA: {CLA_ref:.5f} m^2")
        print(f"dCLA/dV:   {dV[0, 0]:+.6f} m^2/(m/s)")
        print(f"dCLA/dyaw: {dyaw[0, 0]:+.6f} m^2/deg")


    # -------- Finite Difference Step Size Verification

    print("\nFinite Difference Step Size Verification")

    for model in models:

        dV_1, dyaw_1 = calculate_sensitivity(
            model, V_ref, yaw_ref, h_v, h_yaw
        )

        dV_2, dyaw_2 = calculate_sensitivity(
            model, V_ref, yaw_ref, h_v / 2, h_yaw / 2
        )

        print(f"\n{model}")
        print(f"Change in dCLA/dV:   {abs(dV_1[0,0] - dV_2[0,0]):.8f}")
        print(f"Change in dCLA/dyaw: {abs(dyaw_1[0,0] - dyaw_2[0,0]):.8f}")

    return sensitivity_results
