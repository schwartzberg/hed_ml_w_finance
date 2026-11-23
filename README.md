# Can Machine Learning Improve Stock-Sector Selection?

CSCI E-278 *Applied Quantitative Finance and Machine Learning* — final project.

## Run

1. Open `notebooks/Finance_with_ML_CSCI_E278.ipynb` in Google Colab or Jupyter with internet access.
2. Run all cells in order. The notebook downloads Yahoo Finance daily adjusted OHLC for nine original US Select Sector SPDR ETFs and SPY (plus a quoted 13-week Treasury yield, `^IRX`).
3. It saves CSV snapshots and hashes in the local `data/` folder. Retain your own snapshot if you need byte-for-byte reproduction; it is excluded from public GitHub.
4. The final 2022–September 2026 return table, charts and written interpretation are generated from *executed historical data*, never invented.

## Pre-specified design and date discipline

| Item | Rule |
|---|---|
| ETFs | XLB XLE XLF XLI XLK XLP XLU XLV XLY; benchmark SPY |
| Preparatory data | 2014 daily adjusted prices for technical lookbacks |
| Study period | January 2015–September 2026; first October 2026 opening for exit valuation |
| Model inputs | 1m, 3m, 6m adjusted close returns; 63-day volatility; 200-day MA distance; 3m relative SPY return |
| Target | Sector beats SPY over the *following* first-open to first-open investment month |
| Signal / execution | Prior month final close -> next month first open |
| Data leakage | Training label's exit opening must have occurred by signal close; no unfinished outcome used |
| Training | Expanding monthly window (model re-fitted every month); historical results only |
| Tuning | Four predefined forests, chosen by Jan 2019–Nov 2021 walk-forward Brier score |
| Freeze | As of last trading close December 2021; **December 2021 outcome unavailable**, so excluded from tuning |
| Final period | Jan 2022–Sep 2026 (57 investment months), model settings fixed, past completed final-period labels may train later forecasts |
| Portfolio | Top three by predicted probability, equal 1/3 target weights; benchmark six-month momentum top three and passive SPY |
| Cost | 7.5 bps *per dollar traded*, buy and sell; 0/5/7.5/10/20 bps sensitivity |
| Metrics | CAGR, monthly annualised volatility, ^IRX-proxy Sharpe, monthly maximum drawdown, turnover, Brier, classification and top-three accuracy, calibration |

**Important caveats:** The nine original funds are not the current full eleven-sector GICS universe. Adjusted Open is a total-return accounting construction, not a guaranteed transaction price. The 7.5 bps is an assumed cost, not a measured historical cost. `^IRX` approximates the risk-free rate, but is not an investable total-return asset. Monthly max drawdown misses some intra-month declines. Consecutive months and sectors are correlated, limiting statistical confidence. No favourable result is guaranteed.

## Checks

Run `python -m pytest -q tests/` in a Python environment with `numpy`, `pandas`, `scikit-learn`, and `pytest`. Tests use artificial price histories **only** to check program behaviour; they are separate from the research data and its findings.

## Integrity and sources

Original notebook implementation; no copied course slides, notebooks, sample-project code or answer keys are in this repository. The syllabus prohibits publishing course materials without permission. GitHub may be public for **original project work**, but never for restricted class materials. Acknowledge AI-assisted work and independently validate outputs according to the course policy.

## Source control and running from Windows

`notebooks/Finance_with_ML_CSCI_E278.ipynb` is the main submission. The standalone source in `sector_backtest.py` generates the notebook through `python build_notebook.py`; if code changes, rebuild and rerun the notebook, then commit both files.

The local project is already initialised as a Git repository and has its `origin` pointing to `https://github.com/schwartzberg/hed_ml_w_finance.git`. The **initial local commit is not yet pushed** to GitHub. From this folder, run `git push -u origin main` after ensuring GitHub authentication is configured. Do not push restricted course materials or personal data.

A real historical run with Yahoo Finance prices, followed by review of the outputs, is **still required** for final submission. This package includes tested program logic, not a verified historical investment result.
