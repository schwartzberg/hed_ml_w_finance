# Research design — decisions locked before the final test

**Question:** Can a random forest choose US stock-market sectors more effectively than a simple six-month momentum rule or buying and holding SPY?

The following choices were agreed before reviewing real historical test results.

1. **Universe:** The nine original Select Sector SPDR ETFs: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY. SPY is the broad-market benchmark. We do not pretend this is the modern 11-sector universe.
2. **Dates:** Study investment months January 2015 through September 2026; 2014 is an indicator warm-up. Use the first October 2026 trading-session opening solely to complete September's last holding-period valuation.
3. **Prices:** Yahoo Finance daily, split- and dividend-adjusted Open/Close, with downloaded snapshot hashes retained locally.
4. **Inputs:** 1-, 3-, 6-month returns; last 63 trading days' volatility; distance from 200-day moving average; 3-month performance relative to SPY. All known at prior month-end close.
5. **Prediction:** Binary outcome: a sector's next investment month's first-opening-to-first-opening total return exceeds SPY's return over the same dates. The model predicts a probability for each sector.
6. **Portfolio:** Buy the three highest-probability sectors, regardless of whether their probabilities exceed 50%, and target equal 1/3 weights at monthly openings.
7. **Comparisons:** Equal-weight top-three six-month momentum and passive SPY buy-and-hold, over exactly the same holding periods.
8. **Costs:** Assumed 7.5 basis points *per dollar bought or sold*, including every rebalance. Test 0, 5, 7.5, 10 and 20 basis points. This is a scenario assumption, **not** a validated historical cost.
9. **Training window:** Expanding, refitting monthly; compare a four-year rolling window during validation as a robustness check.
10. **Model tuning:** Four predefined modest random forests. Choose by lowest walk-forward Brier score in the pre-final validation window, not by the best realised portfolio profits.
11. **Validation:** Investment months January 2019–November 2021. The December 2021 holding outcome is not available by December 2021's signal and must not inform model selection.
12. **Final test:** January 2022–September 2026 (57 investment months). Freeze the model and selection policy before this test. Earlier **completed** test-period labels may train later walk-forward models; no final-period performance may be used to retune the original hyperparameters.
13. **Forecast timing:** Use the final market close of the preceding month to form each month's features; place the hypothetical trade at the next session's opening. Adjusted openings are accounting constructions, not guaranteed execution prices.
14. **No future outcomes:** A training label may be used only if its holding-period exit opening has occurred **by the forecast's signal close**; tests enforce this chronological rule.
15. **Evaluation:** Report CAGR, monthly annualised volatility, risk-free proxy Sharpe, monthly maximum drawdown, turnover, Brier score vs historical-frequency baseline, top-three hit-rate and probability calibration. Inspect different market environments and distinguish predictive skill from realised returns.

## Verification standards

- Apply the identical date calendar to the forest, momentum and SPY.
- Check missing or suspect vendor prices; refuse to fill fabricated observations silently.
- Preserve downloaded data snapshots with hashes locally; do not publish course materials without instructor permission.
- Run the tests and verify the entire notebook with real vendor data before claiming the work is complete.
- Describe statistical uncertainty and unfavourable outcomes honestly; do not optimise away losses after seeing the final test.
- Acknowledge AI assistance and check the submitted work against the course's originality and citation rules.
