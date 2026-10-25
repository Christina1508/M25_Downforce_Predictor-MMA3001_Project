# M25 Downforce Area Prediction | MMA3001

Numerical regression of **full-vehicle aerodynamic downforce area** ($C_LA$) for the Monash Motorsport M25 Formula Student vehicle using wind-tunnel measurements of airspeed and yaw. The project compares linear regression, quadratic regression and Gaussian radial basis function (RBF) regression, then assesses cross-validation, independent validation and aerodynamic sensitivity.

**Scope:** This implementation predicts the M25's *full-vehicle* $C_LA$ from force-balance data, **not** rear-wing-mainplane pressure distributions or mainplane-only downforce. The model output is downforce area in **m²**, not force in newtons. Downforce force can be calculated separately as $F_{down} = q C_LA$ using dynamic pressure $q$ (Pa).

## Repository structure

```text
data/
  MMS_Time_Averaged.xlsx        Source wind-tunnel workbook
  data_extract.py               Extract relevant Day 2 measurements
  m25_re_yaw_sweep.csv          Generated wind-tunnel training/validation CSV
  cfd_m25_yaw_sweep.txt         CFD reference for independent comparison
scripts/
  downforce_model.py            MAIN SCRIPT: models, CV, optimisation and plots
  validation.py                 Called by the main script for independent validation
  sensitivity.py                Called by the main script for finite differences
figures/                        Model and validation plots (written by scripts)
tests/                          pytest test functions
reports/                        pytest result files
tools/generate_docs.py          Generates HTML source documentation
docs/index.html                 Generated HTML source documentation
requirements.txt                Python packages
LICENSE                         Repository license
```

## Setup and run order

Requires **Python 3.10+**. From the repository root, install dependencies:

```bash
python -m pip install -r requirements.txt
```

**1. Prepare the data (run this from inside the `data/` folder):**

```bash
cd data
python data_extract.py
cd ..
```

`data_extract.py` reads `MMS_Time_Averaged.xlsx` from its **current working directory**, filters Day 2 to the `M25_FW_tap` and `M25_RW_Tap_RPT2` runs, and writes/overwrites `data/m25_re_yaw_sweep.csv`. The CSV is included in this repository, but **rerun extraction to rebuild it** from the source Excel workbook. Running `python data/data_extract.py` from the repository root does **not** work with the present relative paths.

**2. Run the main model (from the repository root):**

```bash
python scripts/downforce_model.py
```

This is the **only modelling script that needs to be run manually**. It automatically calls `validation.py` and `sensitivity.py`. It prints training, cross-validation, optimisation, validation and sensitivity results and writes PNG plots to `figures/`. Keep the `figures/` directory present. This workflow is also intended to be run from the repository root; do not execute the helper scripts individually for the full analysis.

**3. Run tests and produce a test report:**

```bash
python -m pytest -v --junitxml=reports/pytest-results.xml
```

Tests are in `tests/`. The JUnit XML file is the machine-readable test report; a readable example is included in `reports/pytest-report.txt`. Tests exercise actual regression, extraction, validation and finite-difference functions. See the instructions below for interpreting a result.

**4. Regenerate HTML function documentation:**

```bash
python tools/generate_docs.py
```

Open `docs/index.html` in a browser. Documentation is extracted from the source docstrings without importing or rerunning the plotting-intensive main script.

## Input data and outputs

- **Wind-tunnel training:** 21 Day 2 `M25_FW_tap` rows from `m25_re_yaw_sweep.csv`.
- **Independent wind-tunnel validation:** 21 Day 2 `M25_RW_Tap_RPT2` rows from the same CSV, withheld from fitting.
- **Independent CFD comparison:** 7 yaw conditions at 17 m/s from `cfd_m25_yaw_sweep.txt`; this is a computational reference, not direct experimental ground truth.
- **Model inputs:** measured `airspeed` in m/s and `yaw` in degrees. Training nominal airspeeds are 11, 17 and 19 m/s, with yaw angles of -12, -8, -4, 0, 4, 8 and 12 degrees.
- **Model output:** positive downforce area, `CLA` ($C_LA$, m²). The wind-tunnel source stores a negative lift-area sign; the main script reverses the sign for training and wind-tunnel validation to match the CFD downforce convention.
- **Results:** terminal model metrics (RMSE/MAE/bias in m²), selected hyperparameters, and figures for model comparison, cross-validation, validation, and sensitivity. The finite-difference outputs are $\partial(C_LA)/\partial V$ (m²/(m/s)) and $\partial(C_LA)/\partial\beta$ (m²/deg).

**Limitations:** The code does not explicitly validate missing columns, malformed inputs, unseen model names, or out-of-domain velocities/yaws. Missing files or columns cause ordinary Python/pandas errors; an unsupported model name returns `None` in the prediction dispatchers. Predictions outside the measured domain are extrapolations and have not been validated. CFD/wind-tunnel discrepancies include differences in modelling assumptions and measurement conditions. Pytest checks software implementation, **not** physical prediction accuracy; quantitative engineering validation is reported separately by the model script.

## Testing and documentation

`pytest` looks for functions named `test_*` in files named `test_*.py`. A passed test means a *specific expected behaviour* was met. The report shows passed/failed/error counts; any failure should be resolved before submitting the repository. Numerical checks use `numpy.testing.assert_allclose` to account for floating-point rounding.

The HTML documentation is built from the actual module/function docstrings. For assessment, the test report, HTML pages, README, functional source and license should all be committed together.

## AI Acknowledgement

This project may make use of generative Artificial Intelligence (AI) tools, such as ChatGPT, to support aspects of the development process. AI assistance may be used for tasks including:

- Explaining programming concepts and debugging strategies.
- Providing suggestions for code structure and documentation.
- Assisting with brainstorming ideas and planning the project.
- Improving the clarity and grammar of written documentation.
- Explaining error messages and recommending potential solutions.

All AI-generated content has been critically reviewed, tested, and modified where necessary by the project authors. Final design decisions, implementation, testing, and submitted work are the responsibility of the authors.

AI has not been used to generate work that is submitted without verification or understanding. Its use is intended to support learning and productivity while adhering to Monash University's guidelines on the responsible use of Artificial Intelligence.

## Data provenance and permissions

`m25_re_yaw_sweep.csv` is extracted from the provided `MMS_Time_Averaged.xlsx` using `data/data_extract.py` (it is not a synthetic dataset). Only the two Day 2 runs named above are retained; other days and configurations are excluded. The extraction retains the original measurement and metadata columns after renaming selected headers. **The current extraction does not include** the separate `M25_RW_Tap_RPT` single-point run discussed in an earlier README version.

`CLiftA, m^2` in the source workbook is a signed *lift area*, not a dimensionless lift coefficient. The script negates wind-tunnel `CLA` for its positive-downforce convention. Dynamic pressure and measured airspeed are retained in the CSV. Confirm that you have permission to publish/distribute the wind-tunnel workbook and CFD data before making the repository public.
