"This file is used to create the model downforce model of M25"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif"})

# -------- Part 1 - Importing and Preparing Data
data_fn = "data/m25_re_yaw_sweep.csv"
data_cfd_fn = "data/cfd_m25_yaw_sweep.txt"
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
plt.figure()
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
plt.show()


# -------- Part 3 - Quadratic Polynomial Regression
# Function to construct quadratic design matrix
def quadratic_features(X):
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

fig, axes = plt.subplots(1, 3, figsize=(15, 7), dpi=120)

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
plt.show()
