"""Numerical prediction of M25 full-vehicle downforce area.

Fits linear, quadratic and Gaussian RBF regression models using wind tunnel
airspeed and yaw measurements. Performs cross-validation, hyperparameter
optimisation, independent validation and aerodynamic sensitivity analysis.

Inputs: Airspeed (m/s) and yaw angle (deg).
Output: Predicted downforce area, CLA (m^2).

Run after data/data_extract.py to generate the required CSV dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif"})

# -------- Part 1 - Importing and Preparing Data
BASE_DIR = Path(__file__).resolve().parent.parent
data_fn = BASE_DIR/"data"/"m25_re_yaw_sweep.csv"
data_cfd_fn = BASE_DIR/"data"/"cfd_m25_yaw_sweep.txt"

data = pd.read_csv(data_fn) # wind tunnel data
data_cfd = pd.read_csv(data_cfd_fn, sep="\t")

# renaming columns in CFD data
data_cfd = data_cfd.rename(columns={
    "Full CLA": "CLA",
    "Lift Force": "L",
})

# separating training and validation data
train = data[data["description"] == "M25_FW_tap"].copy() 
val1 = data[data["description"] == "M25_RW_Tap_RPT2"].copy() 
val2 = data_cfd.copy()

# extracting yaws from CFD results
val2[["yaw", "airspeed", "steer"]] = val2["Run Name"].str.extract(
    r"^(-?\d+)y(\d+)v(-?\d+)s$"
).astype(int)

# Convert CLA to positive values in WT to match cfd data
train["CLA"] = -train["CLA"]
val1["CLA"] = -val1["CLA"]

# inputs - velocity (m/s) and yaw (deg)
x_train = train[["airspeed", "yaw"]].to_numpy()
X_val1 = val1[["airspeed", "yaw"]].to_numpy()
X_val2 = val2[["airspeed", "yaw"]].to_numpy()

# outputs - CLA
y_train = train["CLA"].to_numpy()
Y_val1 = val1["CLA"].to_numpy()
Y_val2 = val2["CLA"].to_numpy()


# -------- Part 2 - Baseline Model - Linear Regression
A = np.column_stack((
    np.ones(len(x_train)),
    x_train[:, 0],
    x_train[:, 1]
))

# Solve least-squares problem
coeff, residuals, rank, s = np.linalg.lstsq(A, y_train, rcond=None)

# Extract coefficients
a0, a1, a2 = coeff

print("\nLinear Regression Model:")
print(f"CLA = {a0:.4f} + ({a1:.4f})V + ({a2:.4f})yaw")

# Predict CLA for training measurements
y_pred = A @ coeff

# Calculate training errors
errors = y_pred - y_train

rmse = np.sqrt(np.mean(errors**2))
mae = np.mean(np.abs(errors))

print(f"Training RMSE: {rmse:.5f} m^2")
print(f"Training MAE:  {mae:.5f} m^2")

# Plot predicted versus measured CLA
plt.figure(dpi=400)
plt.scatter(y_train, y_pred, label="Training data")
limits = [
    min(y_train.min(), y_pred.min()),
    max(y_train.max(), y_pred.max())
]
plt.plot(limits, limits, "k--", label="Regression")
plt.xlabel(r"Measured $C_LA$ ($\mathrm{m^2}$)")
plt.ylabel(r"Predicted Area $C_LA$ ($\mathrm{m^2}$)")
plt.text(
    0.05, 0.95,
    rf"$\mathrm{{RMSE}} = {rmse:.4f}\ \mathrm{{m^2}}$" "\n"
    rf"$\mathrm{{MAE}} = {mae:.4f}\ \mathrm{{m^2}}$",
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
)
plt.title(
    rf"Linear Regression: $\widehat{{C_LA}}$ vs $C_LA$" "\n"
    rf"$C_LA = {a0:.4f} + ({a1:.4f})V + ({a2:.4f})\beta$"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
FIG_DIR = BASE_DIR/"figures"
plt.savefig(FIG_DIR/"Baseline_LinReg.png")
plt.show()

# -------- Part 3 - Quadratic Polynomial Regression
# Function to construct quadratic design matrix
def quadratic_features(X):
    """Construct the design matrix for quadratic polynomial regression.

    Args:
        X (np.ndarray): Airspeed and yaw inputs, shape (n, 2).

    Returns:
        np.ndarray: Matrix containing [1, V, yaw, V^2, V*yaw, yaw^2].
    """
    
    V = X[:, 0]
    yaw = X[:, 1]

    return np.column_stack((
        np.ones(len(X)),
        V,
        yaw,
        V**2,
        V*yaw,
        yaw**2
    ))
A_quad = quadratic_features(x_train)

# Solve least-squares problem
coeff_quad, _, _, _ = np.linalg.lstsq(
    A_quad, y_train, rcond=None
)

# Function to predict continuous CLA 
def predict_quadratic(X):
    """Predict downforce area using the fitted quadratic regression model.

    Args:
        X (np.ndarray): Airspeed and yaw inputs, shape (n, 2).

    Returns:
        np.ndarray: Predicted CLA values (m^2).
    """
    return quadratic_features(X) @ coeff_quad

# Predict training measurements
y_pred_quad = predict_quadratic(x_train)

# Calculate errors
errors_quad = y_pred_quad - y_train
rmse_quad = np.sqrt(np.mean(errors_quad**2))
mae_quad = np.mean(np.abs(errors_quad))

print("\nQuadratic Regression Model:")
print("Coefficients:", coeff_quad)
print(f"Training RMSE: {rmse_quad:.5f} m^2")
print(f"Training MAE:  {mae_quad:.5f} m^2")


# -------- Part 4 - Gaussian RBF Regression
x_mean = x_train.mean(axis=0)
x_std = x_train.std(axis=0)

# RBF parameters
length_scale = 1.0
regularisation = 0.001

# Gaussian RBF kernel function
def rbf_kernel(X1, X2):
    """Calculate the Gaussian RBF kernel using standardised inputs.

    Args:
        X1 (np.ndarray): First set of airspeed and yaw inputs.
        X2 (np.ndarray): Second set of airspeed and yaw inputs.

    Returns:
        np.ndarray: Gaussian kernel matrix, shape (len(X1), len(X2)).
    """
    X1_scaled = (X1 - x_mean) / x_std
    X2_scaled = (X2 - x_mean) / x_std

    # Squared distances between all pairs of points
    dist_sq = np.sum(
        (X1_scaled[:, None, :] - X2_scaled[None, :, :])**2,
        axis=2
    )

    return np.exp(-dist_sq / (2 * length_scale**2))

# RBF kernel matrix
K = rbf_kernel(x_train, x_train)

# Calculate weights with regularisation
y_mean = y_train.mean()

weights_rbf = np.linalg.solve(
    K + regularisation * np.eye(len(x_train)),
    y_train - y_mean
)

# CLA prediction function
def predict_rbf(X):
    """Predict downforce area using the initial Gaussian RBF model.

    Args:
        X (np.ndarray): Airspeed and yaw inputs, shape (n, 2).

    Returns:
        np.ndarray: Predicted CLA values (m^2).
    """
    return y_mean + rbf_kernel(X, x_train) @ weights_rbf

# Predict training measurements
y_pred_rbf = predict_rbf(x_train)

# Calculate errors
errors_rbf = y_pred_rbf - y_train
rmse_rbf = np.sqrt(np.mean(errors_rbf**2))
mae_rbf = np.mean(np.abs(errors_rbf))

print("\nGaussian RBF Regression Model:")
print(f"Length scale: {length_scale}")
print(f"Regularisation: {regularisation}")
print(f"Training RMSE: {rmse_rbf:.4f} m^2")
print(f"Training MAE:  {mae_rbf:.4f} m^2")


# -------- Part 5 - Training Model Comparison
a0q, a1q, a2q, a3q, a4q, a5q = coeff_quad

linear_eq = (
    rf"$\widehat{{C_{{LA}}}} = {a0:.3f}"
    rf"{a1:+.3f}V"
    rf"{a2:+.3f}\beta$"
)

quadratic_eq = (
    rf"$\widehat{{C_{{LA}}}} = {a0q:.3f}"
    rf"{a1q:+.3f}V"
    rf"{a2q:+.3f}\beta"
    rf"{a3q:+.4f}V^2"
    rf"{a4q:+.4f}V\beta"
    rf"{a5q:+.4f}\beta^2$"
)

rbf_eq = (
    rf"$\widehat{{C_{{LA}}}} = \bar{{y}} + \sum_{{j=1}}^n w_j \phi(\mathbf{{x}}, \mathbf{{x}}_j)$" "\n"
    rf"$\bar{{y}} = {y_mean:.3f},\ \ell = {length_scale:.1f},\ \lambda = {regularisation:.3f}$"
)

fig, axes = plt.subplots(1, 3, figsize=(15, 7), dpi=400)

models = [
    ("Linear Regression", linear_eq, y_pred, rmse, mae),
    ("Quadratic Regression", quadratic_eq, y_pred_quad, rmse_quad, mae_quad),
    ("Gaussian RBF", rbf_eq, y_pred_rbf, rmse_rbf, mae_rbf)
]

limits = [
    min(y_train.min(), y_pred.min(),
        y_pred_quad.min(), y_pred_rbf.min()),
    max(y_train.max(), y_pred.max(),
        y_pred_quad.max(), y_pred_rbf.max())
]

for ax, (name, equation, predictions, model_rmse, model_mae) in zip(axes, models):
    ax.scatter(y_train, predictions)
    ax.plot(limits, limits, "k--", label="Prediction Model")

    ax.set_xlabel(r"Measured $C_LA$ ($\mathrm{m^2}$)")
    ax.set_ylabel(r"Predicted $C_LA$ ($\mathrm{m^2}$)")
    ax.set_title(name + "\n" + equation, fontsize=11)

    ax.text(
        0.05, 0.95,
        rf"$\mathrm{{RMSE}} = {model_rmse:.4f}\ \mathrm{{m^2}}$" "\n"
        rf"$\mathrm{{MAE}} = {model_mae:.4f}\ \mathrm{{m^2}}$",
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
    )

    ax.grid(True)
    ax.legend()
    ax.set_xlim(limits)
    ax.set_ylim(limits)

plt.tight_layout()
plt.savefig(FIG_DIR/"Baseline_ModelComparison.png")
plt.show()



# -------- Part 6 - Cross-Validation and Model Selection

# Fit a model and predict unseen inputs
def fit_predict(model, X_fit, y_fit, X_test):
    """Fit a regression model and predict withheld measurements.

    Used during cross-validation to train on the selected subset without
    including withheld measurements.

    Args:
        model (str): "Linear", "Quadratic", or "Gaussian RBF".
        X_fit (np.ndarray): Training airspeed and yaw inputs.
        y_fit (np.ndarray): Training CLA values (m^2).
        X_test (np.ndarray): Inputs at which predictions are required.

    Returns:
        np.ndarray: Predicted CLA values (m^2), or None for an
            unsupported model name.
    """

    if model == "Linear":
        A_fit = np.column_stack((np.ones(len(X_fit)), X_fit))
        A_test = np.column_stack((np.ones(len(X_test)), X_test))

        coeff, _, _, _ = np.linalg.lstsq(A_fit, y_fit, rcond=None)
        return A_test @ coeff

    elif model == "Quadratic":
        A_fit = quadratic_features(X_fit)
        A_test = quadratic_features(X_test)

        coeff, _, _, _ = np.linalg.lstsq(A_fit, y_fit, rcond=None)
        return A_test @ coeff

    elif model == "Gaussian RBF":
        # Standardise using ONLY the current training fold
        mean = X_fit.mean(axis=0)
        std = X_fit.std(axis=0)
        std = np.where(std == 0, 1, std)

        def kernel(X1, X2):
            """Calculate the Gaussian kernel using training-fold scaling."""
            X1 = (X1 - mean) / std
            X2 = (X2 - mean) / std

            dist_sq = np.sum(
                (X1[:, None, :] - X2[None, :, :])**2,
                axis=2
            )
            return np.exp(-dist_sq / (2 * length_scale**2))

        K = kernel(X_fit, X_fit)
        y_mean_fold = y_fit.mean()

        weights = np.linalg.solve(
            K + regularisation * np.eye(len(X_fit)),
            y_fit - y_mean_fold
        )

        return y_mean_fold + kernel(X_test, X_fit) @ weights


# Group measurements by nominal yaw angle
yaw_groups = 4 * np.rint(x_train[:, 1] / 4).astype(int)

model_names = ["Linear", "Quadratic", "Gaussian RBF"]

# Store predictions and results for each model
cv_predictions = {}
cv_results = {}

for model in model_names:

    predictions = np.zeros(len(y_train))

    # Withhold one yaw group at a time
    for yaw in np.unique(yaw_groups):

        test_mask = yaw_groups == yaw
        train_mask = ~test_mask

        X_fit = x_train[train_mask]
        y_fit = y_train[train_mask]
        X_test = x_train[test_mask]

        # Fit model and predict withheld measurements
        predictions[test_mask] = fit_predict(
            model, X_fit, y_fit, X_test
        )

    # Calculate cross-validation errors
    errors_cv = predictions - y_train
    rmse_cv = np.sqrt(np.mean(errors_cv**2))
    mae_cv = np.mean(np.abs(errors_cv))

    cv_predictions[model] = predictions
    cv_results[model] = (rmse_cv, mae_cv)

    print(f"\n{model} Cross-Validation:")
    print(f"RMSE: {rmse_cv:.5f} m^2")
    print(f"MAE:  {mae_cv:.5f} m^2")

# Mdel with lowest cross-validation RMSE
best_model = min(cv_results, key=lambda m: cv_results[m][0])

print(f"\nBest model based on CV RMSE: {best_model}")


# Plotting
equations = {
    "Linear": linear_eq,
    "Quadratic": quadratic_eq,
    "Gaussian RBF": rbf_eq
}

fig, axes = plt.subplots(1, 3, figsize=(15, 8), dpi=400)

limits = [
    min(y_train.min(), *(p.min() for p in cv_predictions.values())),
    max(y_train.max(), *(p.max() for p in cv_predictions.values()))
]

for ax, model in zip(axes, model_names):

    predictions = cv_predictions[model]
    rmse_cv, mae_cv = cv_results[model]

    ax.scatter(y_train, predictions, s=40)
    ax.plot(limits, limits, "k--", label="Prediction Model")

    ax.set_xlabel(r"Measured $C_LA$ ($\mathrm{m^2}$)")
    ax.set_ylabel(r"Predicted $C_LA$ ($\mathrm{m^2}$)")
    ax.set_title(
        model + "\nLeave-One-Yaw-Out CV\n" + equations[model],
        fontsize=11,
        pad=12
    )

    ax.text(
        0.05, 0.95,
        rf"$\mathrm{{RMSE}} = {rmse_cv:.4f}\ \mathrm{{m^2}}$" "\n"
        rf"$\mathrm{{MAE}} = {mae_cv:.4f}\ \mathrm{{m^2}}$",
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
    )

    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.grid(True)
    ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(FIG_DIR/"ConstCV_models.png")
plt.show()

# -------- Part 7 - Velocity Cross-Validation
# Group by airspeeds
nominal_speeds = [11, 17, 19]
speed_groups = np.array([
    min(nominal_speeds, key=lambda s: abs(v - s))
    for v in x_train[:, 0]
])

# Withhold the 17 m/s test condition
test_mask = speed_groups == 17
train_mask = ~test_mask

# Split data
X_fit = x_train[train_mask]
y_fit = y_train[train_mask]

X_test = x_train[test_mask]
y_test = y_train[test_mask]

# Store results
velocity_predictions = {}
velocity_results = {}

for model in model_names:

    # Fit using 11 and 19 m/s, predict 17 m/s
    predictions = fit_predict(model, X_fit, y_fit, X_test)

    # Calculate prediction errors
    errors = predictions - y_test
    rmse_vel = np.sqrt(np.mean(errors**2))
    mae_vel = np.mean(np.abs(errors))

    velocity_predictions[model] = predictions
    velocity_results[model] = (rmse_vel, mae_vel)

    print(f"\n{model} - Velocity Interpolation:")
    print(f"RMSE: {rmse_vel:.5f} m^2")
    print(f"MAE:  {mae_vel:.5f} m^2")

# Plot predictions 
fig, ax = plt.subplots(figsize=(8, 5), dpi=400)

# Sort measurements by yaw
order = np.argsort(X_test[:, 1])
yaw = X_test[order, 1]

# Experimental measurements
ax.plot(
    yaw, y_test[order], "ko-",
    linewidth=2, label="Wind Tunnel"
)

# Model predictions
for model in model_names:
    predictions = velocity_predictions[model]
    rmse_vel, mae_vel = velocity_results[model]

    ax.plot(
        yaw, predictions[order], "o--",
        label=f"{model} (RMSE = {rmse_vel:.4f})"
    )

ax.set_xlabel(r"Yaw Angle, $\beta$ ($^\circ$)")
ax.set_ylabel(r"Downforce Area, $C_LA$ ($\mathrm{m^2}$)")
ax.set_title("Velocity Cross-Validation at 17 m/s")

ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR/"velocity_cross_val.png")
plt.show()



# -------- Part 8 - Gaussian RBF Hyperparameter Optimisation
# Hyperparameter ranges
length_scales = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
regularisations = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# Function to fit and predict hyperparams
def rbf_cv_predict(X_fit, y_fit, X_test, ls, reg):
    """Fit and evaluate a Gaussian RBF model with specified parameters.

    Standardises inputs using the training fold to avoid data leakage
    during hyperparameter optimisation.

    Args:
        X_fit (np.ndarray): Training airspeed and yaw inputs.
        y_fit (np.ndarray): Training CLA values (m^2).
        X_test (np.ndarray): Withheld inputs for prediction.
        ls (float): Gaussian kernel length scale.
        reg (float): Regularisation parameter.

    Returns:
        np.ndarray: Predicted CLA values (m^2).
    """

    # Standardise inputs using training fold only
    mean = X_fit.mean(axis=0)
    std = X_fit.std(axis=0)
    std = np.where(std == 0, 1, std)

    def kernel(X1, X2):
        """Calculate the Gaussian kernel using the specified length scale."""
        X1 = (X1 - mean) / std
        X2 = (X2 - mean) / std

        dist_sq = np.sum(
            (X1[:, None, :] - X2[None, :, :])**2,
            axis=2
        )

        return np.exp(-dist_sq / (2 * ls**2))

    # Fit RBF model
    K = kernel(X_fit, X_fit)
    y_mean_fold = y_fit.mean()

    weights = np.linalg.solve(
        K + reg * np.eye(len(X_fit)),
        y_fit - y_mean_fold
    )

    # Predict withheld measurements
    return y_mean_fold + kernel(X_test, X_fit) @ weights


# Grid search using leave-one-yaw-out cross-validation
rmse_grid = np.zeros((len(length_scales), len(regularisations)))

for i, ls in enumerate(length_scales):
    for j, reg in enumerate(regularisations):

        predictions = np.zeros(len(y_train))

        for yaw in np.unique(yaw_groups):

            test_mask = yaw_groups == yaw
            train_mask = ~test_mask

            predictions[test_mask] = rbf_cv_predict(
                x_train[train_mask],
                y_train[train_mask],
                x_train[test_mask],
                ls,
                reg
            )

        rmse_grid[i, j] = np.sqrt(
            np.mean((predictions - y_train)**2)
        )


# Identify hyperparameters
best_idx = np.unravel_index(np.argmin(rmse_grid), rmse_grid.shape)

best_length_scale = length_scales[best_idx[0]]
best_regularisation = regularisations[best_idx[1]]
best_rmse = rmse_grid[best_idx]

print("\nGaussian RBF Hyperparameter Optimisation")
print(f"Optimal length scale: {best_length_scale}")
print(f"Optimal regularisation: {best_regularisation}")
print(f"Minimum CV RMSE: {best_rmse:.5f} m^2")


# Plot hyperparameter sensitivity
fig, ax = plt.subplots(figsize=(9, 6), dpi=400)

im = ax.imshow(
    rmse_grid,
    origin="lower",
    aspect="auto",
    cmap="viridis"
)

ax.set_xticks(range(len(regularisations)))
ax.set_xticklabels([f"{r:g}" for r in regularisations])

ax.set_yticks(range(len(length_scales)))
ax.set_yticklabels(length_scales)

ax.set_xlabel(r"Regularisation, $\lambda$")
ax.set_ylabel(r"Length Scale, $\ell$")
ax.set_title(
    "Gaussian RBF Hyperparameter Optimisation\n"
    "Leave-One-Yaw-Out Cross-Validation"
)

# Displaying RMSE
for i in range(len(length_scales)):
    for j in range(len(regularisations)):
        ax.text(
            j, i, f"{rmse_grid[i, j]:.4f}",
            ha="center",
            va="center",
            color="white",
            fontsize=9
        )

# Optimal hyperparameters
ax.scatter(
    best_idx[1], best_idx[0],
    marker="o",
    s=220,
    facecolors="none",
    edgecolors="red",
    linewidths=2.5
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"Cross-Validation RMSE ($\mathrm{m^2}$)")

plt.tight_layout()
plt.savefig(FIG_DIR/"Gaussian_param_opti.png")
plt.show()


# -------- Part 9 - Independent Model Validation

# Final Optimised Gaussian RBF Model
x_mean_tuned = x_train.mean(axis=0)
x_std_tuned = x_train.std(axis=0)
x_std_tuned = np.where(x_std_tuned == 0, 1, x_std_tuned)

def tuned_rbf_kernel(X1, X2):
    """Calculate the Gaussian RBF kernel using optimised hyperparameters.

    Args:
        X1 (np.ndarray): First set of airspeed and yaw inputs.
        X2 (np.ndarray): Second set of airspeed and yaw inputs.

    Returns:
        np.ndarray: Gaussian kernel matrix using the optimised length scale.
    """
    X1_scaled = (X1 - x_mean_tuned) / x_std_tuned
    X2_scaled = (X2 - x_mean_tuned) / x_std_tuned

    dist_sq = np.sum(
        (X1_scaled[:, None, :] - X2_scaled[None, :, :])**2,
        axis=2
    )

    return np.exp(-dist_sq / (2 * best_length_scale**2))


# Fit final model using all training measurements
K_tuned = tuned_rbf_kernel(x_train, x_train)
y_mean_tuned = y_train.mean()

weights_tuned = np.linalg.solve(
    K_tuned + best_regularisation * np.eye(len(x_train)),
    y_train - y_mean_tuned
)


# Prediction functions for the three fitted models
def predict_model(model, X):
    """Predict downforce area using a fitted regression model.

    The Gaussian RBF option uses the optimised length scale and
    regularisation determined during cross-validation.

    Args:
        model (str): "Linear", "Quadratic", or "Gaussian RBF".
        X (np.ndarray): Airspeed and yaw inputs, shape (n, 2).

    Returns:
        np.ndarray: Predicted CLA values (m^2), or None for an
            unsupported model name.
    """
    if model == "Linear":
        A = np.column_stack((np.ones(len(X)), X))
        return A @ coeff

    elif model == "Quadratic":
        return quadratic_features(X) @ coeff_quad

    elif model == "Gaussian RBF":
        return (
            y_mean_tuned
            + tuned_rbf_kernel(X, x_train) @ weights_tuned
        )


# Testing validation dataset into the model
from validation import run_validation

validation_results = run_validation(
    predict_model=predict_model,
    model_names=model_names,
    train=train,
    val1=val1,
    val2=val2
)



# -------- Part 10 - Aerodynamic Sensitivity Analysis
from sensitivity import run_sensitivity

sensitivity_results = run_sensitivity(
    predict_model=predict_model,
    x_train=x_train
)

