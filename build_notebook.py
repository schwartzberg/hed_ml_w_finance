"""Build a self-contained notebook; no repository-local imports required in Colab."""
from pathlib import Path
import argparse
import nbformat as nbf
root=Path(__file__).resolve().parent
code_source=(root/'sector_backtest.py').read_text(encoding='utf-8')
nb=nbf.v4.new_notebook()
nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
nb.metadata.language_info={'name':'python'}
nb.metadata.colab={'provenance':[]}
def md(s):nb.cells.append(nbf.v4.new_markdown_cell(s))
def code(s):nb.cells.append(nbf.v4.new_code_cell(s))
md('''# Can Machine Learning Improve Stock-Sector Selection?

**CSCI E-278 — Applied Quantitative Finance and Machine Learning | Final project**

**Research question.** Can a random forest select the three US stock-market sectors most likely to outperform SPY in the next month, more successfully than six-month momentum or simply holding SPY?

**Result on the saved Yahoo Finance snapshot.** The forest earned 8.07% a year after costs. The momentum rule earned 6.18% and SPY earned 11.94%. The forest picked a winning sector 79 times in 171 attempts. The 57-month test gives little evidence that it can consistently identify next month's winners.

The universe consists of nine original Select Sector SPDR ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV and XLY). SPY represents the broad market. The study covers January 2015 through September 2026. Prices from 2014 supply the history needed for the first signal.

Sections 1–5 cover the data and forecast timing. Sections 6–8 compare investment returns and prediction quality. Sections 9–12 examine costs, uncertainty and the conclusion. See the sources and AI acknowledgement at the end.''')
md('''## 1. Time and experiment design

| Period | Investment months | Use |
|:--|:--|:--|
| Initial training history | Jan 2015–Dec 2018 | Completed labels used to train the classifier |
| Walk-forward validation | Jan 2019–**Nov 2021** | Select among four pre-defined forest settings by Brier score |
| Freeze | Dec 2021 closing bell | Fix features, model settings, and benchmarks |
| Final unseen evaluation | Jan 2022–Sep 2026 | Compare investment outcomes without retuning |

**Why November 2021 is the last validation investment month.** December's *investment outcome* is only known at the first trading day's opening in January 2022. It was unavailable when deciding the January 2022 trade at the final December 2021 close. We therefore cannot use December 2021's realised outcome to choose hyperparameters.

The model produces a signal **after the last trading day's close** of a month. It buys at the **first market open of the next month**, holds until the **first open of the following month**, then rebalances. Each training label must have completed its exit opening before a later signal; an unfinished previous-month label is excluded. The expanding window grows as outcomes become available. The final-period model is refitted on past completed results, but settings remain locked.

The final September 2026 holding is valued using the first market opening in **October 2026**. No October observation is used to predict September's holdings.''')
md('''## 2. Data and reproducibility

Daily Yahoo Finance **dividend- and split-adjusted Open and Close** are used. We download data from January 2014. `auto_adjust=True` is stated explicitly so a change in `yfinance` defaults cannot silently change our analysis. Adjusted opens are total-return accounting prices, not literal execution prices; a fixed cost is an approximation for dealing costs. The 13-week Treasury quote, `^IRX`, is an imperfect **risk-free-rate proxy**, not a tradable portfolio return.

This cell requires internet access on first execution. It writes a reproducible price snapshot to local `data/`. Rerunning with the same snapshot avoids market-vendor revisions. If downloading fails, execution stops: **we do not substitute synthetic returns**.''')
code('''# Colab bootstrap. With a verified local price snapshot, no network is needed.
from pathlib import Path
import sys, subprocess
snapshot_files=['adjusted_open.csv','adjusted_close.csv','treasury_yield.csv','manifest.json']
if not all((Path('data')/p).is_file() for p in snapshot_files):
    try:
        import yfinance
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable,'-m','pip','install','-q','yfinance>=0.2.40'])
''')
code(code_source)
nb.cells[-1].metadata.jupyter = {'source_hidden': True}
code('''import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import sklearn
print('pandas',pd.__version__,'scikit-learn',sklearn.__version__)
open_px, close_px, yield_pct = load_prices('data')
manifest=json.loads(Path('data/manifest.json').read_text(encoding='utf-8'))
print('Snapshot source:', manifest['source'], '| yfinance', manifest['yfinance_version'])
for filename, digest in manifest['sha256'].items():
    print(filename, digest)
print('Rows:',len(close_px),'available:',close_px.index.min().date(),'-',close_px.index.max().date())
display(close_px.loc[['2026-09-30']].round(2))
print('First October opening for the September exit:', open_px.loc['2026-10':].index[0].date())''')
md('''## 3. Six features and the prediction target

All features are calculated from information available at the **prior month-end close**: sector returns over 1, 3 and 6 calendar months; annualised volatility of the last 63 daily returns; the distance from the 200-trading-day moving average; and the sector's 3-month return relative to SPY.

The label is **1** when the sector's upcoming first-open-to-first-open return exceeds SPY's over the same dates; otherwise it is **0**. This relative-performance label can be 1 even when the sector loses money. Each month contains nine related observations, not nine independent market regimes.''')
code('''panel=make_panel(open_px,close_px,yield_pct)
print('Complete rows:',len(panel),'months:',panel.month.nunique())
assert len(panel)==1269 and panel.month.nunique()==141
assert np.isfinite(panel[list(FEATURES)].to_numpy()).all()
display(panel[['month','ticker','signal_date','entry_date','exit_date',*FEATURES,'label']].head(12))''')
md('''## 4. Walk-forward model selection: prediction quality first

We compare four modest forests (100/200 trees; depth 3/5; minimum leaf sizes 20/15). Their settings were fixed before looking at final test outcomes. For each month in the 2019–November 2021 validation period, we train only on labels whose outcomes are already known, then forecast the nine sectors.

The **Brier score** evaluates probability quality: lower is better. We compare it with a no-lookahead forecast that simply uses the historical frequency of positive labels. A four-year rolling training window is reported as a robustness check alongside our primary **expanding window**, but it is **not** selected based on the unseen 2022–2026 returns. A fixed random seed makes fitting reproducible.''')
code('''name,chosen_params,validation_table=choose_model(panel)
print('Locked model:',name,chosen_params)
display(validation_table.round(4))
valid_forecasts,_=walkforward(panel,*VALIDATION,chosen_params)
rolling_forecasts,_=walkforward(panel,*VALIDATION,chosen_params,rolling_months=48)
window_results=pd.DataFrame({'training_window':['Expanding','Rolling 48 months'],
    'validation_Brier':[brier_score_loss(valid_forecasts.label,valid_forecasts.probability),
                        brier_score_loss(rolling_forecasts.label,rolling_forecasts.probability)]})
display(window_results.round(4))
assert valid_forecasts.month.max()=='2021-11'
assert rolling_forecasts.training_months.max()==48
print('Lowest validation Brier among the four forests:', round(validation_table.brier.min(),4))
print('Historical-frequency benchmark:', round(validation_table.naive_brier.iloc[0],4))''')
md('''All four forests produced larger validation errors than the historical-frequency benchmark. The shallow, 100-tree model was the best of the four. The original model-selection rule carried that forest into the final test.\n\n## 5. Lock the model; predict January 2022–September 2026

No hyperparameters, features or investment rules may be changed using the final results. For every forecast, the calendar audit verifies that the latest training label completed **before the forecast signal**. A January 2022 signal therefore cannot train on the December 2021 position's unrealised return.''')
code('''final_forecasts,feature_importance=walkforward(panel,*TEST,chosen_params)
assert audit(panel,final_forecasts)
print('PASS: 57 final monthly forecasts, 513 sector-level predictions; no future labels')
display(final_forecasts.head(9))''')
md('''## 6. Three investment strategies and transaction costs

**Forest:** buy the three sectors with the highest predicted probabilities, even if all are below 50%.

**Momentum:** buy the three sectors with the highest **prior six-month** adjusted close-to-close returns.

**SPY:** invest once and hold. No subsequent trading.

Sector strategies target **one-third in each selected ETF** at each monthly opening. Weights drift as sector prices change. Each month the strategy pays fees from its own capital, then invests what remains at the target weights. Holding the same names can still require trades. We charge **7.5 basis points (0.075%) on the dollar value bought *and* sold**, including the initial purchases. The final observation marks the open positions to market. The simulation charges no terminal sale. Turnover includes purchases and sales, each measured against wealth immediately before the trade. We separately test 0, 5, 10 and 20 basis points.''')
code('''forest_picks=select(panel,final_forecasts,'forest')
momentum_picks=select(panel,final_forecasts,'momentum')
spy_picks={mo:('SPY',) for mo in forest_picks}
strategies={
 'Random forest':simulate(panel,forest_picks,bps=COST_BPS),
 'Six-month momentum':simulate(panel,momentum_picks,bps=COST_BPS),
 'SPY buy-and-hold':simulate(panel,spy_picks,bps=COST_BPS,spy=True)
}
assert all(len(x)==57 for x in strategies.values())
assert all(x.index.equals(strategies['Random forest'].index) for x in strategies.values())
print('PASS: identical 57-month investment horizons, cost and drift accounting')
display(pd.DataFrame({'Forest':pd.Series(forest_picks),'Momentum':pd.Series(momentum_picks)}).head(12))''')
md('''## 7. Performance and downside risk

CAGR is the annual compound growth rate. Annual volatility equals the standard deviation of monthly portfolio returns times the square root of 12. For the Sharpe ratio, subtract the monthly Treasury proxy from each portfolio return, divide the mean of these excess returns by their standard deviation, and multiply by the square root of 12. The Treasury proxy converts the quoted annual yield at the signal date to a monthly effective rate. Maximum drawdown measures the largest decline from an earlier peak at the monthly valuations.

Dollar charts show the growth of a **$10,000** initial investment, after the assumed costs. The chart includes initial capital at the first entry opening, ensuring that first-month losses count towards the drawdown.''')
code('''performance=pd.DataFrame({k:metrics(v) for k,v in strategies.items()}).T
performance_table = performance.rename(columns={
    'CAGR': 'Annual return', 'volatility': 'Annual volatility', 'sharpe': 'Sharpe',
    'max_monthly_drawdown': 'Maximum drawdown', 'total_return': 'Total return',
    'annual_turnover': 'Annual turnover (x)'})
display(performance_table.style.format({
    'Annual return':'{:.2%}', 'Annual volatility':'{:.2%}', 'Sharpe':'{:.3f}',
    'Maximum drawdown':'{:.2%}', 'Total return':'{:.2%}', 'Annual turnover (x)':'{:.2f}'}))
values=pd.DataFrame({k:v.nav.to_numpy()*10000 for k,v in strategies.items()},
                    index=strategies['Random forest'].valuation_date)
values.index.name='Valuation date'
first_open=panel.loc[panel.month=='2022-01','entry_date'].iloc[0]
values=pd.concat([pd.DataFrame({k:[10000.] for k in values.columns},
                               index=pd.DatetimeIndex([first_open],name='Valuation date')),values])
assert np.allclose(values.iloc[0].to_numpy(),10000)
ax=values.plot(figsize=(11,5),title='Growth of $10,000, after costs')
ax.set_ylabel('Portfolio value (USD)');ax.grid(alpha=.2);plt.tight_layout();plt.show()
drawdowns=values.div(values.cummax()).sub(1)*100
assert all(np.isclose(drawdowns[k].min()/100,performance.loc[k,'max_monthly_drawdown'])
           for k in values.columns), 'Drawdown chart differs from risk table'
ax=drawdowns.plot(figsize=(11,4),title='Drawdown at monthly valuation dates')
ax.set_ylabel('Drawdown (%)');ax.grid(alpha=.2);plt.tight_layout();plt.show()''')
md('''SPY finished with the most wealth and the highest Sharpe ratio. Momentum had the shallowest drawdown. The forest lost about a quarter of its value from its peak and traded about 12.7 dollars per year for each dollar invested, counting buys and sells. SPY turnover consists of the initial purchase spread over the full test period.

## 8. Does the forest actually predict better?

The Brier score measures probability quality, not investment profit and not ranking quality. We also report the **top-three hit rate** (share of selected sector-months that outperform SPY), compare it with the expected rate of choosing three of the nine sectors at random, and inspect **calibration** in five probability bins. The random baseline is a *diagnostic*, not an investable trading strategy.''')
code('''quality,calibration=forecast_diagnostics(final_forecasts)
quality['brier_skill_vs_historical_frequency']=1-quality['brier']/quality['naive_brier']
display(pd.DataFrame([quality],index=['Final test']).round(4))
display(calibration.round(3))
valid_bins=calibration.dropna(subset=['mean_probability','observed_frequency'])
fig,ax=plt.subplots(figsize=(7,5))
ax.plot([0,1],[0,1],ls='--',color='grey',label='Perfect calibration')
ax.scatter(valid_bins.mean_probability,valid_bins.observed_frequency,s=60,label='Forest')
for row in valid_bins.itertuples():
    ax.annotate(f'n={row.n}', (row.mean_probability,row.observed_frequency),
                xytext=(7,-12), textcoords='offset points')
ax.set(xlim=(0,1),ylim=(-.02,1.04),xlabel='Mean forecast probability',
       ylabel='Observed outperform frequency',title='Probability calibration, final period')
ax.legend();ax.grid(alpha=.2);plt.tight_layout();plt.show()''')
md('''Of 513 forecasts, 509 fell in the 40–60% bin. Their average predicted probability was 48.3%, and sectors beat SPY in 46.4% of those cases. Each of the two other occupied bins contained just two observations. The plot provides little information about calibration at those extremes.

## 9. Cost sensitivity and stability

Here we keep the actual selections unchanged and calculate portfolio outcomes at five costs. We do **not** choose the cost with the best result. Feature-importance values average the sequentially fitted final-period forests. Correlated features may share importance; impurity-based scores are descriptive, not proof of cause.''')
code('''grid=[]
for bps in [0,5,7.5,10,20]:
    for name,picks,is_spy in [('Random forest',forest_picks,False),
                              ('Six-month momentum',momentum_picks,False),
                              ('SPY buy-and-hold',spy_picks,True)]:
        res=metrics(simulate(panel,picks,bps=bps,spy=is_spy))
        grid.append({'one_way_cost_bps':bps,'strategy':name,**res})
cost_table=pd.DataFrame(grid)
display(cost_table.pivot(index='one_way_cost_bps',columns='strategy',values='CAGR').style.format('{:.2%}'))
ax=feature_importance[list(FEATURES)].mean().sort_values().plot.barh(figsize=(8,4),
      title='Average sequential forest feature importance')
ax.set_xlabel('Mean impurity-based importance');plt.tight_layout();plt.show()''')
md('''SPY led at every tested cost, even with free trading. The forest beat momentum at all five rates. Trading costs reduced the forest's return most because it traded the most. The forest assigned the greatest average importance to one-month return and 63-day volatility. Their high rank describes how the trees split observations and says little about forecast accuracy on its own.

## 10. How uncertain are the differences?

The 57 investment months are not independent observations. We therefore use a **six-month circular block bootstrap** to estimate approximate 95% intervals for the difference in mean monthly returns between the forest and each benchmark. We also bootstrap the month-level top-three hit-rate advantage over a random three-sector selection. Each resample preserves contiguous groups of months and the cross-sector outcomes within each month. We never use the resulting intervals to redesign the model.

The table reports percentage-point differences per month. Return differences use the arithmetic mean of paired monthly net returns. The hit-rate difference compares the selected three with the fraction of all nine sectors beating SPY that month. These units differ from the annual compound returns above.''')
code('''lift_by_month=top3_monthly_lift(final_forecasts)
bootstrap_tests={
 'Forest minus momentum return':block_bootstrap_ci(
      strategies['Random forest'].return_net.to_numpy()-
      strategies['Six-month momentum'].return_net.to_numpy()),
 'Forest minus SPY return':block_bootstrap_ci(
      strategies['Random forest'].return_net.to_numpy()-
      strategies['SPY buy-and-hold'].return_net.to_numpy()),
 'Top-three hit-rate lift vs random':block_bootstrap_ci(lift_by_month.to_numpy())
}
uncertainty=pd.DataFrame(bootstrap_tests).T
intervals_pp = 100 * uncertainty[['mean_monthly_difference','ci_low','ci_high']]
intervals_pp.columns = ['Mean (pp)', 'Lower 95% (pp)', 'Upper 95% (pp)']
display(intervals_pp.round(2))
print('Approximate 95% intervals from 2,000 draws of six-month blocks.')''')
md('''## 11. Changing market conditions and what the findings mean

We examine each year's realised performance to avoid overlooking a strategy that only works in one kind of market. This analysis is **descriptive**: after seeing these values we do not change the test set or any model setting.

Interpret the investment results alongside prediction quality and turnover. A model can forecast relative performance reasonably well and still fail to earn more than SPY. Conversely, higher realised return over a short sample need not mean the model learned a persistent signal.''')
code('''monthly=pd.DataFrame({name:out.return_net.to_numpy() for name,out in strategies.items()},
    index=pd.PeriodIndex(strategies['Random forest'].index,freq='M'))
annual=monthly.groupby(monthly.index.year).agg(lambda x:(1+x).prod()-1)
display(annual.rename_axis('Investment year').style.format('{:.2%}'))
print('The 2026 entry covers January–September only; it is not a full-year return.')
fr=performance.loc['Random forest','CAGR']; mo=performance.loc['Six-month momentum','CAGR']
spy=performance.loc['SPY buy-and-hold','CAGR']
forest_before_2026 = (1 + monthly.loc[:'2025-12', 'Random forest']).prod() - 1
momentum_before_2026 = (1 + monthly.loc[:'2025-12', 'Six-month momentum']).prod() - 1
conclusion = f"""**Was the forest worth using?** Over the full test it earned **{fr:.2%}** a year after costs. Momentum earned **{mo:.2%}** and SPY earned **{spy:.2%}**. The timing matters. Through December 2025 the forest had gained **{forest_before_2026:.1%}**, while momentum had gained **{momentum_before_2026:.1%}**. The forest overtook momentum in the final nine months, when it earned **{annual.loc[2026, 'Random forest']:.1%}**.

**Could it pick winners?** The three selected sectors beat SPY in **{quality['top3_hit_rate']:.1%}** of cases. Random selection would have averaged **{quality['random_top3_reference']:.1%}** across the same months. The Brier score was **{quality['brier']:.4f}**, slightly worse than the historical-frequency forecast of **{quality['naive_brier']:.4f}**. A sector can miss SPY by a small amount or beat it by a large one. The hit rate counts both events equally, while portfolio returns depend on the size of each gain or loss.

**How much confidence is warranted?** All three block-bootstrap intervals include zero. This sample gives little evidence of a repeatable sector-selection advantage. The difference in realised wealth is meaningful to an investor who lived through this period, but 57 monthly decisions provide limited evidence about the next period."""
display(Markdown(conclusion))''')
md('''## 12. Limitations, evidence and conclusion

Our nine fixed ETFs do not cover the modern full eleven-sector SPDR universe. Historical GICS classification changes affect comparisons. ETFs remain diversified and liquid, but first-open prices need not match an actual order's fill. Dividend-adjusted prices approximate reinvestment; spreads, taxes, market impact and commissions are not independently reconstructed. The 7.5-bps transaction cost is only an assumption, tested at alternatives. The `^IRX` quoted yield is an approximation to a risk-free monthly rate. Drawdowns sampled monthly understate some intramonth pain. Market conditions, selected thresholds and the relatively small number of independent months limit confidence. The project tests historical value added; it does not promise future returns.

**Reproduce and validate.** Run every notebook cell in order, retain the downloaded `data` snapshot and its SHA-256 manifest, and run the accompanying `pytest` tests. Examine errors and any outliers rather than silently deleting dates. Do not interpret charts until the data and timing assertions pass. The unit tests use synthetic fixtures *only* to test code behaviour, never to report an empirical result.

**Sources.** CSCI E-278 Fall 2026 syllabus; Modules I–IV course notes; lecture notebooks on walk-forward testing, random forests, portfolio risk; Yahoo Finance historical data and `yfinance` price adjustment; scikit-learn `RandomForestClassifier` and `brier_score_loss` documentation. The final-project sample was reviewed for expected notebook presentation and was not copied.

**Sources for implementation.** Yahoo Finance daily prices and 13-week Treasury quotes, retrieved with yfinance 1.8.0 for the stored run. The manifest gives the three CSV hashes and the requested date range. The original snapshot has no retrieval timestamp. Section 2 prints the hashes for comparison. [Yahoo Finance](https://finance.yahoo.com/) · [yfinance download](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html) · [scikit-learn random forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) · [scikit-learn Brier score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html) · [William F. Sharpe, The Sharpe Ratio (1994)](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm).

**AI acknowledgement.** OpenAI. ChatGPT, 23 November 2026. Used to discuss the research design, draft and revise code and prose, write tests and check calculations against the saved price data. [ChatGPT](https://chatgpt.com/).''')
nbf.validate(nb)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=root/'notebooks'/'Finance_with_ML_CSCI_E278.ipynb')
out = parser.parse_args().output
nbf.write(nb,out)
print('Created',out,len(nb.cells),'cells')
