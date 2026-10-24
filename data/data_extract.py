"""Extracting relevant data points from the input wind tunnel data excel workbook

The given data contains data that is not relevant to the model such as various geometry changes 
and different test subjects. The relevant data is from day2 with the descriptions M25_FW_tap 
and M25_RW_Tap_RPT2

This script should be run first before any training is done to the model. 
The output put is a csv file named m25_re_yaw_sweep.csv that contains the relevant
training and validation data.

Please ensure the relevant libraries have been installed such as openpyxl
"""

import os
import pandas as pd

# File names
input_fn = "MMS_Time_Averaged.xlsx"
output_fn = "m25_re_yaw_sweep.csv"

df = pd.read_excel(input_fn)

descriptions = ["M25_FW_tap", "M25_RW_Tap_RPT2"]

# Validation - only extraction data from day 2 and the relevant runs
data = df[
    (df["Day"] ==2)&
    (df["Description"].isin(descriptions))
].copy()

#Renaming column items
data = data.rename(columns={
    "Day": "day",
    "Seq IDs": "sequence_id",
    "Run no": "run",
    "Point no": "point",
    "Description": "description",
    "Airspeed, m/sec": "airspeed",
    "Yaw angle, deg": "yaw",
    "Dynamic pressure, Pa": "q",
    "CLiftA, m^2": "CLA",
    "Lift force, N": "L"
})

data.to_csv(output_fn, index=False)


