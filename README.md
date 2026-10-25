
# Numerical Prediction of Downforce for the Monash Motorsport M25 Using Wind Tunnel Data

MMA3001 - Numerical Methods and Machine Learning

This project develops numerical models to predict the aerodynamic downforce of the Monash Motorsport M25 Formula Student vehicle using experimental wind tunnel data.

Linear regression, quadratic polynomial regression, and Gaussian Radial Basis Function (RBF) regression are compared using cross-validation. The models are independently evaluated against repeated wind tunnel measurements and CFD results. Aerodynamic sensitivity to velocity and yaw angle is also investigated using finite differences.

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
│   └── sensitivity.py
├── tests/
│   ├── test_model.py
│   └── test_results.txt
├── figures/
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

## Input Data

The model uses wind tunnel measurements from the Monash Motorsport M25 vehicle.

**Inputs:**
- Airspeed, V (m/s)
- Yaw angle, β (degrees)

**Output:**
- Downforce area, CLA (m²)

The experimental dataset includes nominal airspeeds of 11, 17 and 19 m/s, with yaw angles from -12° to +12° in 4° increments.

The model predicts the full-vehicle downforce area rather than individual aerodynamic component contributions. Wind tunnel CLA values are sign-converted to represent positive downforce area.

## Running the Project

Scripts should be executed in the following order.

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

The generated CSV is required by the main model script.

Data extraction only needs to be repeated if the CSV is missing or the source workbook changes.

### 2. Main Downforce Model

From the repository root, run:

```bash
python scripts/downforce_model.py
```

**This is the main script required to execute the project.**

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

The experimental dataset originates from wind tunnel testing of the Monash Motorsport M25 vehicle.

`MMS_Time_Averaged.xlsx` contains the original experimental measurements. The generated `m25_re_yaw_sweep.csv` contains only the selected Day 2 measurements relevant to model training and validation.

`cfd_m25_yaw_sweep.txt` provides CFD reference data used for comparison.

`CLiftA, m^2` in the source workbook is signed lift area, rather than a dimensionless lift coefficient. The model reverses the sign of the wind tunnel CLA measurements to represent positive downforce area.

The underlying datasets should only be redistributed with appropriate permission.

## AI Acknowledgement

This project may make use of generative Artificial Intelligence (AI) tools, such as ChatGPT, to support aspects of the development process. AI assistance may be used for tasks including:

- Explaining programming concepts and debugging strategies.
- Providing suggestions for code structure and documentation.
- Assisting with brainstorming ideas and planning the project.
- Improving the clarity and grammar of written documentation.
- Explaining error messages and recommending potential solutions.

All AI-generated content has been critically reviewed, tested, and modified where necessary by the project authors. Final design decisions, implementation, testing, and submitted work are the responsibility of the authors.

AI has not been used to generate work that is submitted without verification or understanding. Its use is intended to support learning and productivity while adhering to Monash University's guidelines on the responsible use of Artificial Intelligence.
