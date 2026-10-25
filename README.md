# Numerical Prediction of Downforce for the Monash Motorsport M25 Using Wind Tunnel Data

**MMA3001 – Numerical Methods and Machine Learning**

This project develops numerical models to predict the aerodynamic downforce of the Monash Motorsport M25 Formula Student vehicle using experimental wind tunnel data.

Linear regression, quadratic polynomial regression, and Gaussian Radial Basis Function (RBF) regression are compared using cross-validation. The models are independently evaluated against repeated wind tunnel measurements and CFD results. Aerodynamic sensitivity to velocity and yaw angle is also investigated using finite differences.

An interactive browser-based GUI allows users to enter operating conditions and obtain downforce predictions and empirical uncertainty intervals using the fitted models.

## Repository Structure

```text
MMA3001-Project-ChristinaDevaDoss/
├── data/
│   ├── MMS_Time_Averaged.xlsx
│   ├── data_extract.py
│   ├── m25_re_yaw_sweep.csv
│   └── cfd_m25_yaw_sweep.txt
├── scripts/
│   ├── downforce_model.py
│   ├── validation.py
│   ├── sensitivity.py
│   └── model_gui.py
├── tests/
│   ├── test_model.py
│   └── test_results.txt
├── figures/
├── Open M25 Predictor.bat
├── README.md
└── LICENSE
```

## Dependencies

Python 3 with the following libraries:

- NumPy
- Pandas
- Matplotlib
- OpenPyXL
- Pytest (for automated testing)

Install using:

```bash
python -m pip install numpy pandas matplotlib openpyxl pytest
```

The GUI runs locally in a standard web browser and does not require additional Python packages beyond those listed above.

## Input Data

The model uses wind tunnel measurements from the Monash Motorsport M25 vehicle.

**Inputs:**
- Airspeed, V (m/s)
- Yaw angle, β (degrees)

**Output:**
- Downforce area, CLA (m²)

The experimental dataset includes nominal airspeeds of 11, 17 and 19 m/s, with yaw angles from −12° to +12° in 4° increments.

The model predicts the full-vehicle downforce area rather than individual aerodynamic component contributions. Wind tunnel CLA values are sign-converted to represent positive downforce area.

## Running the Project

### 1. Data Extraction

The `data/` directory contains the original wind tunnel workbook, CFD reference data, and the data extraction script.

Before running the model, execute `data_extract.py` from within the `data/` directory:

```bash
cd data
python data_extract.py
cd ..
```

This extracts the relevant Day 2 measurements from `MMS_Time_Averaged.xlsx` and generates `m25_re_yaw_sweep.csv`.

The CSV contains:
- `M25_FW_tap`: 21 training measurements.
- `M25_RW_Tap_RPT2`: 21 independent repeat measurements for validation.

The generated CSV is required by the main model script and the GUI.

Data extraction only needs to be repeated if the CSV is missing or the source workbook changes.

### 2. Main Downforce Model

From the repository root, run:

```bash
python scripts/downforce_model.py
```

**This is the main script required to execute the numerical modelling and analysis.**

It performs:
1. Data import and preprocessing.
2. Baseline linear regression.
3. Quadratic polynomial and Gaussian RBF regression.
4. Training error comparison.
5. Leave-one-yaw-out cross-validation and model comparison.
6. Velocity interpolation validation.
7. Gaussian RBF hyperparameter optimisation.
8. Independent model validation.
9. Aerodynamic sensitivity analysis.

The main script automatically imports and executes the functions defined in `validation.py` and `sensitivity.py`. These scripts do not need to be run separately.

Results are printed to the terminal, and generated plots are saved in the `figures/` directory.

### 3. Interactive Downforce Predictor (GUI)

The GUI provides a browser-based interface for predicting the Monash Motorsport M25's whole-vehicle aerodynamic downforce at user-defined operating conditions.

**Windows — recommended:** From the repository root, double-click:

```text
Open M25 Predictor.bat
```

**Alternative — any platform with Python 3:** Run the following from the repository root:

```bash
python scripts/model_gui.py
```

The launcher starts a local Python server and opens the predictor in the default web browser. Keep the terminal or command window open while using the application; press **Ctrl+C** in that window to stop the server. The local web address uses an available port and may differ between launches.

The GUI loads and executes the existing `downforce_model.py` and uses its fitted `predict_model()` function. **There is no need to run `downforce_model.py` separately before launching the GUI.** The model coefficients and RBF weights are not recreated in the browser.

**Using the interface:**
1. Select **Linear**, **Quadratic**, or **Gaussian RBF (optimised)** regression.
2. Enter the desired **airspeed (m/s)** and **yaw angle (degrees)** using the fields or sliders.
3. Adjust **air density (kg/m³)** if required; the default is 1.225 kg/m³.
4. Select an **80%, 90%, or 95% empirical interval**.
5. Read the predicted **downforce area, CLA (m²)** and **downforce force (N)**. The page also shows the independent wind tunnel RMSE, mean signed error (bias), and a plot of the predicted yaw response.

Downforce force is calculated from the predicted downforce area using:

\[
D = \tfrac{1}{2}\rho V^2 C_LA
\]

The GUI restricts inputs to the measured training-data bounds in velocity and yaw. The training sweep comprises nominal velocities of 11, 17 and 19 m/s and yaw angles between −12° and +12°; predictions at intermediate conditions are model estimates rather than additional experimental measurements.

**Uncertainty interpretation:** The displayed intervals are empirical error bands derived from the absolute residuals of the 21 independent wind tunnel validation measurements. They should **not** be interpreted as rigorously established per-prediction confidence probabilities. In particular, the training and validation runs used different pressure-tap configurations, and their systematic offset may affect the intervals. The GUI does not correct the model predictions for this offset.

The GUI operates locally on `127.0.0.1` and requires the repository's existing data and Python scripts; opening an HTML file by itself is not sufficient to run the Python model.

## Automated Testing

Automated tests are implemented using Pytest in `tests/test_model.py`.

To run the tests, execute the following from the repository root:

```bash
python tests/test_model.py
```

The test script checks numerical regression methods, kernel properties, finite-difference sensitivities, data extraction, validation metrics and execution of the overall model.

Test results are displayed in the terminal and automatically saved as:

`tests/test_results.txt`

The text file contains a readable transcript of the test results, including passed or failed tests and any error messages.

The report is generated automatically when the test script is executed.

## Validation

Model performance is assessed using:
- Training RMSE and MAE.
- Leave-one-yaw-out cross-validation.
- Velocity interpolation validation.
- Independent wind tunnel repeat measurements.
- CFD yaw sweep comparison at 17 m/s.

Sensitivity analysis uses centred finite differences to estimate the partial derivatives of downforce area with respect to velocity and yaw angle.

The model is intended for prediction and interpolation within the experimentally investigated operating range. Predictions outside this range are not independently validated.

## Data Provenance

The experimental dataset originates from wind tunnel testing of the Monash Motorsport M25 vehicle at the Australian Centre for Sports Aerodynamics (ACSA) on 24 February 2026.

`MMS_Time_Averaged.xlsx` contains the original experimental measurements. The generated `m25_re_yaw_sweep.csv` contains only the selected Day 2 measurements relevant to model training and validation.

`cfd_m25_yaw_sweep.txt` provides CFD reference data used for comparison.

`CLiftA, m^2` in the source workbook is signed lift area, rather than a dimensionless lift coefficient. The model reverses the sign of the wind tunnel CLA measurements to represent positive downforce area.

The underlying datasets should only be redistributed with appropriate permission.

## AI Acknowledgement

This project may make use of generative Artificial Intelligence (AI) tools, such as ChatGPT, to support aspects of the development process. AI assistance may be used for tasks including:

- Explaining programming concepts and debugging strategies.
- Providing suggestions for code structure and documentation.
- Assisting with brainstorming ideas and planning the project.
- Assist in generating code for the model gui
- Improving the clarity and grammar of written documentation.
- Explaining error messages and recommending potential solutions.

All AI-generated content has been critically reviewed, tested, and modified where necessary by the project authors. Final design decisions, implementation, testing, and submitted work are the responsibility of the authors.

AI has not been used to generate work that is submitted without verification or understanding. Its use is intended to support learning and productivity while adhering to Monash University's guidelines on the responsible use of Artificial Intelligence.
