# Can Machine Learning Improve Stock-Sector Selection?

Harvard CSCI E-278 · Applied Quantitative Finance and Machine Learning

## Approved project abstract

The following is the approved proposal for this project.

> **Can Machine Learning Improve Stock-Sector Selection?**
>
> I will test whether a random forest can select US stock-market sectors more effectively than a simple momentum rule. I will study nine long-established sector ETFs using historical data from 2015 to September 2026. And for each sector, I will calculate recent returns, moving-average trends and volatility.
>
> The model will predict whether a sector is likely to outperform the S&P 500 during the next month. Each month, a portfolio will hold the three sectors with the highest predicted probabilities, equally weighted. I am thinking that I will then compare it with a six-month momentum strategy selecting three sectors and with a buy-and-hold investment in SPY.
>
> I will use walk-forward testing so that each prediction uses only information available at the time.
> I will include trading costs and compare returns, volatility, Sharpe ratios and also maximum drawdowns.
>
> The final aim is to see whether the machine-learning strategy adds value in unseen historical periods, and this is while recognising that past results cannot guarantee future returns.

## What we implemented

We implemented this approved abstract in the [executed Jupyter notebook](notebooks/Finance_with_ML_CSCI_E278.ipynb). The forest ranked nine sector ETFs each month by their chance of beating SPY, our proxy for the S&P 500. We invested equally in the first three and compared the portfolio with the three strongest six-month momentum sectors and with holding SPY. We selected the model settings using earlier data, refitted the forest each month using completed outcomes, and assessed all three portfolios over the same 57 months after trading costs.

[HTML reading copy](notebooks/Finance_with_ML_CSCI_E278.html) · [The 15 research decisions](PROJECT_DESIGN.md) · [Rubric review](SUBMISSION_CHECKLIST.md)

## Did machine learning add value in unseen months?

**Aim.** Could a monthly investor earn more after trading costs by following the forest than by following six-month momentum or holding SPY? We tested the 57 months from January 2022 through September 2026. Each forecast used only information available before that month's trade.

**Answer.** In the 57 unseen months, the forest earned **8.07% a year** after trading costs, compared with **6.18%** for six-month momentum. Machine learning therefore added value for an investor choosing between those two sector strategies. An investor who simply held SPY earned **11.94% a year**, more than either strategy.

All three portfolios used the same opening dates and an assumed cost of 7.5 basis points per dollar bought or sold. The September holding was valued at the October 1 opening.

| Strategy | Annual compound return | Final value of $10,000 | Annual volatility | Sharpe | Maximum monthly drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random forest | 8.07% | $14,455 | 15.62% | 0.332 | −26.52% |
| Six-month momentum | 6.18% | $13,293 | 14.89% | 0.220 | −12.20% |
| SPY buy-and-hold | 11.94% | $17,085 | 15.25% | 0.572 | −23.37% |

The forest overtook momentum in January–September 2026. Its top three picks beat SPY 79 times in 171 selections (46.2%). Picking three sectors at random would have averaged 46.6% over the same months. Its probability forecasts also trailed the past-frequency benchmark on Brier score, 0.2502 versus 0.2494. A lower score is better.

**Would the lead over momentum carry over to another period?** The forest earned **0.16 percentage points more per month** than momentum on average. The approximate 95% interval for that difference runs from **0.74 points behind** momentum to **1.11 points ahead**. Zero falls inside that range, so an average advantage of zero is also consistent with the data. The forest beat momentum over these particular 57 months. We cannot conclude from this test that its sector picks will beat momentum in another period. **Past results cannot guarantee future returns.**

## Run locally or in Colab

1. Use a Python environment with the packages in `requirements.txt`. For local JupyterLab, open a prompt in `notebooks`, start `python -m jupyterlab`, and open `Finance_with_ML_CSCI_E278.ipynb`. The same notebook can be uploaded directly to Colab.
2. Run all cells in order. The notebook uses `data/` relative to the kernel's working directory. A normal local launch from `notebooks/` uses `notebooks/data/`. The first run downloads adjusted daily opens and closes for the nine ETFs and SPY and Treasury yield quotes for `^IRX` from Yahoo Finance.
3. Keep the three CSVs and their manifest together. A rerun verifies their SHA-256 hashes before use. The notebook records the hashes of the reported run in section 2. A later vendor download can differ. The repository contains the executed notebook and aggregate results. The full vendor files stay on the local machine.
4. From the project root, run `python -m pytest -q tests`. The tests use artificial histories to check forecast timing, the rolling training window, cash accounting, risk measures and snapshot integrity.

The reported run used Python 3.12.14, NumPy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, Matplotlib 3.11.2 and yfinance 1.8.0. For implementation changes, edit `sector_backtest.py` and `build_notebook.py`. Run `python build_notebook.py` to regenerate the notebook, then execute and save it again. The builder also accepts `--output PATH` for a separate working copy.

The notebook includes method references. Course materials and the full market-data snapshots are excluded from the repository.
