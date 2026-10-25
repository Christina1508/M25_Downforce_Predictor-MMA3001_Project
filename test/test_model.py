"""Pytest checks for the M25 wind-tunnel downforce prediction workflow.

Run from the repository root:
    python tests/test_model.py

A readable pytest transcript will be written to tests/test_results.txt.
"""

import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")  # Do not open interactive plot windows during tests.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))


def close_plots(*args, **kwargs):
    """Close figures instead of showing them during automated tests."""
    plt.close("all")


@pytest.fixture(scope="module")
def model():
    """Run the actual model once, without displaying or overwriting figures."""
    with patch.object(plt, "savefig"), patch.object(plt, "show", side_effect=close_plots):
        with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
            return runpy.run_path(str(SCRIPTS / "downforce_model.py"))


def test_quadratic_features(model):
    """The quadratic design matrix contains the six intended terms."""
    X = np.array([[2.0, -3.0], [4.0, 1.0]])
    expected = [[1, 2, -3, 4, -6, 9], [1, 4, 1, 16, 4, 1]]
    assert_allclose(model["quadratic_features"](X), expected)


def test_linear_regression(model):
    """Linear regression recovers predictions from a known linear equation."""
    X = np.array([[11, -8], [11, 8], [17, 0], [19, 4]], dtype=float)
    y = 1.2 + 0.3 * X[:, 0] - 0.05 * X[:, 1]
    X_test = np.array([[15.0, 2.0], [18.0, -4.0]])

    predicted = model["fit_predict"]("Linear", X, y, X_test)
    expected = 1.2 + 0.3 * X_test[:, 0] - 0.05 * X_test[:, 1]
    assert_allclose(predicted, expected, atol=1e-10)


def test_quadratic_regression(model):
    """Quadratic regression recovers predictions from a known polynomial."""
    X = np.array([[v, a] for v in (11., 17., 19.) for a in (-8., 0., 8.)])

    def polynomial(points):
        v, yaw = points[:, 0], points[:, 1]
        return 3 + 0.2*v - 0.01*yaw + 0.002*v**2 + 0.003*v*yaw - 0.0005*yaw**2

    X_test = np.array([[15., 2.], [18., -4.]])
    predicted = model["fit_predict"]("Quadratic", X, polynomial(X), X_test)
    assert_allclose(predicted, polynomial(X_test), atol=1e-8)


def test_rbf_constant_response(model):
    """An RBF model predicts a constant when all training outputs are equal."""
    X = np.array([[11., -4.], [11., 4.], [17., -4.], [17., 4.]])
    X_test = np.array([[14., 0.], [16., 2.]])
    predicted = model["rbf_cv_predict"](
        X, np.full(len(X), 3.7), X_test, ls=1.0, reg=0.001
    )
    assert_allclose(predicted, [3.7, 3.7], atol=1e-10)


def test_gaussian_kernel(model):
    """The Gaussian kernel is symmetric and equals one on its diagonal."""
    X = model["x_train"]
    K = model["tuned_rbf_kernel"](X, X)
    assert K.shape == (len(X), len(X))
    assert_allclose(K, K.T, atol=1e-12)
    assert_allclose(np.diag(K), 1.0, atol=1e-12)
    assert np.all((K >= 0) & (K <= 1))


def test_end_to_end_pipeline(model):
    """The full model produces finite CV metrics and validation outputs."""
    assert set(model["model_names"]) == {"Linear", "Quadratic", "Gaussian RBF"}
    assert all(
        np.isfinite(rmse) and np.isfinite(mae)
        for rmse, mae in model["cv_results"].values()
    )
    assert model["validation_results"]
    assert model["sensitivity_results"]


def test_finite_difference_sensitivity():
    """Centred differences recover known velocity and yaw derivatives."""
    from scripts.sensitivity import run_sensitivity

    def linear_predict(_model, X):
        return 5 + 2 * X[:, 0] + 3 * X[:, 1]

    X = np.array([[11., -12.], [11., 12.], [19., -12.], [19., 12.]])
    with patch.object(plt, "savefig"), patch.object(plt, "show", side_effect=close_plots):
        results = run_sensitivity(linear_predict, X)

    for result in results.values():
        assert_allclose(result["dCLA_dV"], 2.0, atol=1e-10)
        assert_allclose(result["dCLA_dyaw"], 3.0, atol=1e-10)


def test_data_extraction(tmp_path, monkeypatch):
    """Data extraction reproduces the supplied Day 2 training and repeat rows."""
    import shutil

    shutil.copy2(ROOT / "data" / "MMS_Time_Averaged.xlsx", tmp_path)
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(ROOT / "data" / "data_extract.py"))

    actual = pd.read_csv(tmp_path / "m25_re_yaw_sweep.csv")
    expected = pd.read_csv(ROOT / "data" / "m25_re_yaw_sweep.csv")
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-10, atol=1e-10)
    assert len(actual) == 42
    assert actual["description"].value_counts().to_dict() == {
        "M25_FW_tap": 21, "M25_RW_Tap_RPT2": 21
    }
    assert (actual["day"] == 2).all()


def test_validation_metrics():
    """Independent validation calculates RMSE, MAE and bias correctly."""
    from scripts.validation import run_validation

    train = pd.DataFrame({"airspeed": [17.], "yaw": [0.], "CLA": [3.]})
    repeat = pd.DataFrame({
        "airspeed": [17., 19.], "yaw": [-4., 4.], "CLA": [2., 4.]
    })
    cfd = pd.DataFrame({
        "airspeed": [17., 17.], "yaw": [-4., 4.], "CLA": [3., 5.]
    })

    def predict(_model, X):
        return np.full(len(X), 3.0)

    with patch.object(plt, "savefig"), patch.object(plt, "show", side_effect=close_plots):
        results = run_validation(predict, ["Quadratic"], train, repeat, cfd)

    wt = results["Wind Tunnel Repeat Measurements"]["Quadratic"]
    assert_allclose([wt["rmse"], wt["mae"], wt["bias"]], [1., 1., 0.])

    cfd_result = results["CFD Yaw Sweep at 17m/s"]["Quadratic"]
    assert_allclose(
        [cfd_result["rmse"], cfd_result["mae"], cfd_result["bias"]],
        [np.sqrt(2), 1., -1.]
    )


if __name__ == "__main__":
    # Run pytest and save the same readable results shown in the terminal.
    TEST_DIR = Path(__file__).resolve().parent
    report_path = TEST_DIR / "test_results.txt"
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", str(__file__),
            "-v", "--tb=short", "--color=no", "-p", "no:cacheprovider",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    transcript = result.stdout + result.stderr
    print(transcript, end="")
    report_path.write_text(transcript, encoding="utf-8")
    print(f"\nTest report saved to: {report_path}")
    sys.exit(result.returncode)
