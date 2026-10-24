# MMA3001-Project-ChristinaDevaDoss
Repository containing the firmware for my MMA3001 Project
## AI Acknowledgement

This project may make use of generative Artificial Intelligence (AI) tools, such as ChatGPT, to support aspects of the development process. AI assistance may be used for tasks including:

- Explaining programming concepts and debugging strategies.
- Providing suggestions for code structure and documentation.
- Assisting with brainstorming ideas and planning the project.
- Improving the clarity and grammar of written documentation.
- Explaining error messages and recommending potential solutions.

All AI-generated content has been critically reviewed, tested, and modified where necessary by the project authors. Final design decisions, implementation, testing, and submitted work are the responsibility of the authors.

AI has not been used to generate work that is submitted without verification or understanding. Its use is intended to support learning and productivity while adhering to Monash University's guidelines on the responsible use of Artificial Intelligence.


# Data provenance and permissions

`selected_day2.csv` is a faithful extraction of rows from the user-provided
`MMS_Time_Averaged.xlsx`, not a synthetic dataset. `source_excel_row` identifies
its source Excel row. All other dates and altered-geometry configurations were
excluded. Day 2 `M25_FW_tap` (21 points) is the training run;
`M25_RW_Tap_RPT2` (21 points) is the independent complete repeat, and
`M25_RW_Tap_RPT` (1 point at 11 m/s and 0 deg) is an additional repeat.

`CLiftA, m^2` is a signed *lift area*, not a dimensionless lift coefficient.
Positive downforce is `-Lift force, N`. Dynamic pressure and measured speed are
retained for the force and density checks.


