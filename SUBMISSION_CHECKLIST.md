# Final project rubric review

**Primary submission** is the [executed Jupyter notebook](notebooks/Finance_with_ML_CSCI_E278.ipynb). The course rubric assigns up to ten points to each of the four criteria. This review identifies the evidence for the highest band. The instructor awards the grade.

| Rubric criterion | Evidence in the notebook | Assessment |
| --- | --- | --- |
| Notebook organisation | The opening gives the research question and answer. Sections 1–5 trace prices into forecasts. Sections 6–12 follow trades through costs, risk and interpretation. | A reader can follow one investment decision from signal to exit and find the result before reading the code. |
| Description of findings and outputs | Sections 7–11 explain wealth, drawdowns, forecast bins, feature importance, costs and uncertainty. Tables distinguish annual compound returns, monthly percentage-point differences and the nine-month 2026 return. | The text explains why the forest can beat momentum in dollars while its picks win less often than the random reference. It gives the sample sizes behind the calibration chart. |
| Topics covered | Six price-based features, probabilistic classification, walk-forward fitting, calendar-based leakage checks, portfolio construction, fees, drawdowns and paired block resampling. | Each method addresses a specific investment or measurement question. Model selection uses only the validation period. |
| Creative implementation | The experiment tests the same rankings as both forecasts and trades. It pairs past-frequency and random-selection references with two investable benchmarks. | The results isolate different questions about probability accuracy, sector selection and realised wealth. The cost sensitivity shows how trading affects the conclusion. |

## Verification on 23 November 2026

- [x] Verified the SHA-256 hashes of the saved Yahoo Finance CSVs.
- [x] Recomputed 1,269 sector-month observations and audited 513 out-of-sample probabilities over 57 months.
- [x] Executed the full notebook in a fresh Python kernel using the saved snapshot. Verified the four figures, the result tables and the absence of cell errors.
- [x] Ran the project test suite, including regression tests for the number of completed training months and the excess-return Sharpe ratio.
- [x] Checked portfolio fees, total returns, cost sensitivity and paired uncertainty intervals. The 15 prespecified research decisions are recorded in [PROJECT_DESIGN.md](PROJECT_DESIGN.md).
- [x] Included references, an AI acknowledgement and the limits of inference from 57 investment months.

The academic review and GitHub publication are separate from submitting the notebook to the course.
