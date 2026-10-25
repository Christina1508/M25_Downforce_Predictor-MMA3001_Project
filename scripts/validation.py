"""Independent validation of the M25 downforce prediction models.

Evaluates regression predictions against repeated wind tunnel measurements
and CFD reference data using RMSE, MAE and mean prediction bias.

Generates comparison plots in the figures directory.
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

# -------- Part 9 - Independent Model Validation

def run_validation(predict_model, model_names, train, val1, val2):
    """Evaluate fitted models against independent validation datasets.

    Calculates RMSE, MAE and mean prediction bias for wind tunnel repeat
    measurements and CFD results. Generates validation comparison plots.

    Args:
        predict_model (callable): Function returning CLA predictions for
            a model name and an array of input conditions.
        model_names (list[str]): Names of regression models to evaluate.
        train (pd.DataFrame): Training wind tunnel measurements.
        val1 (pd.DataFrame): Independent wind tunnel repeat measurements.
        val2 (pd.DataFrame): CFD reference measurements.

    Returns:
        dict: Validation results for each dataset and model, containing
            predictions, RMSE, MAE and bias in m^2.
    """
    # Independent validation datasets
    validation_sets = {
        "Wind Tunnel Repeat Measurements": (
            val1[["airspeed", "yaw"]].to_numpy(),
            val1["CLA"].to_numpy()
        ),
        "CFD Yaw Sweep at 17m/s": (
            val2[["airspeed", "yaw"]].to_numpy(),
            val2["CLA"].to_numpy()
        )
    }

    validation_results = {}

    # Calculate validation errors
    for dataset_name, (X_val, Y_val) in validation_sets.items():

        validation_results[dataset_name] = {}

        print(f"\n{dataset_name}")
        print("-" * 45)

        for model in model_names:

            predictions = predict_model(model, X_val)

            errors = predictions - Y_val
            rmse = np.sqrt(np.mean(errors**2))
            mae = np.mean(np.abs(errors))
            bias = np.mean(errors)

            validation_results[dataset_name][model] = {
                "predictions": predictions,
                "rmse": rmse,
                "mae": mae,
                "bias": bias
            }

            print(
                f"{model:15s} "
                f"RMSE = {rmse:.5f} m^2, "
                f"MAE = {mae:.5f} m^2, "
                f"Bias = {bias:+.5f} m^2"
            )


    # -------- Plot Independent Validation Results
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=400)

    all_values = []

    for dataset_name, (X_val, Y_val) in validation_sets.items():
        all_values.extend(Y_val)

        for model in model_names:
            all_values.extend(
                validation_results[dataset_name][model]["predictions"]
            )

    limits = [min(all_values), max(all_values)]

    for ax, (dataset_name, (X_val, Y_val)) in zip(axes, validation_sets.items()):

        ax.plot(limits, limits, "k--", label="Ideal Prediction")
        for model in model_names:

            results = validation_results[dataset_name][model]
            ax.scatter(
                Y_val,
                results["predictions"],
                s=45,
                label=f"{model} (RMSE = {results['rmse']:.4f})"
            )

        ax.set_xlabel(r"Reference $C_LA$ ($\mathrm{m^2}$)")
        ax.set_ylabel(r"Predicted $C_LA$ ($\mathrm{m^2}$)")
        ax.set_title(dataset_name)

        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR/"validation_fits.png")
    plt.show()


    # -------- Part 9A - Investigating Validation Discrepancies

    # Select nominal 17 m/s wind tunnel measurements
    train_17 = train[
        np.abs(train["airspeed"] - 17) < 0.5
    ].sort_values("yaw")

    val_17 = val1[
        np.abs(val1["airspeed"] - 17) < 0.5
    ].sort_values("yaw")

    cfd_17 = val2.sort_values("yaw")

    fig, ax = plt.subplots(figsize=(9, 6), dpi=400)

    # Experimental training data
    ax.plot(
        train_17["yaw"],
        train_17["CLA"],
        "o-",
        label="Wind Tunnel - Training"
    )

    # Independent experimental validation data
    ax.plot(
        val_17["yaw"],
        val_17["CLA"],
        "s-",
        label="Wind Tunnel - Validation"
    )

    # CFD reference data
    ax.plot(
        cfd_17["yaw"],
        cfd_17["CLA"],
        "^-",
        label="CFD"
    )

    # Evaluate models at 17 m/s
    yaw_plot = np.linspace(-12, 12, 200)

    X_plot = np.column_stack((
        np.full(len(yaw_plot), 17.0),
        yaw_plot
    ))

    for model in ["Quadratic", "Gaussian RBF"]:
        ax.plot(
            yaw_plot,
            predict_model(model, X_plot),
            "--",
            label=model
        )

    ax.set_xlabel(r"Yaw Angle, $\beta$ ($^\circ$)")
    ax.set_ylabel(r"Downforce Area, $C_LA$ ($\mathrm{m^2}$)")
    ax.set_title(
        r"Downforce Response to Yaw at $V = 17\ \mathrm{m/s}$"
    )

    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR/"validation_descrepancies.png")

    plt.show()

    return validation_results
