"""Original, time-aligned CSCI E-278 sector ETF experiment.

Research data must come from an actual Yahoo Finance snapshot; test fixtures
are allowed in automated tests but must not be included in reported results.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

SECTORS = ('XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY')
TICKERS = (*SECTORS,'SPY')
FEATURES = ('ret1','ret3','ret6','vol63','ma200_distance','relative3')
CANDIDATES = {
 'small_100':dict(n_estimators=100,max_depth=3,min_samples_leaf=20),
 'small_200':dict(n_estimators=200,max_depth=3,min_samples_leaf=20),
 'medium_100':dict(n_estimators=100,max_depth=5,min_samples_leaf=15),
 'medium_200':dict(n_estimators=200,max_depth=5,min_samples_leaf=15)
}
COST_BPS = 7.5
VALIDATION = ('2019-01','2021-11')
TEST = ('2022-01','2026-09')


def _extract(raw, field, tickers):
    if not isinstance(raw.columns,pd.MultiIndex):
        if len(tickers) != 1: raise ValueError('Multi-ticker Yahoo result not shaped as expected')
        out=raw[[field]].rename(columns={field:tickers[0]})
    elif field in raw.columns.get_level_values(0):
        out=raw.xs(field,axis=1,level=0)
    elif field in raw.columns.get_level_values(1):
        out=raw.xs(field,axis=1,level=1)
    else: raise ValueError('Missing OHLC field '+field)
    out=out.reindex(columns=list(tickers)).astype(float).sort_index()
    out.index=pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out.index.name='Date'
    return out


def load_prices(folder='data',refresh=False):
    """Use saved snapshots when present; no fabricated replacement data."""
    d=Path(folder); d.mkdir(parents=True,exist_ok=True)
    paths=[d/'adjusted_open.csv',d/'adjusted_close.csv',d/'treasury_yield.csv']
    if all(p.exists() for p in paths) and not refresh:
        # Refuse silently altered snapshots rather than mixing vendor revisions.
        manifest_file=d/'manifest.json'
        if not manifest_file.exists():
            raise RuntimeError('Existing CSV snapshot lacks its SHA-256 manifest; remove it and refresh data')
        saved=json.loads(manifest_file.read_text(encoding='utf-8'))
        for file in paths:
            expected=saved.get('sha256',{}).get(file.name)
            if not expected or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f'Data snapshot integrity check failed for {file.name}')
        opened,closed=(pd.read_csv(p,index_col=0,parse_dates=True) for p in paths[:2])
        yield_pct=pd.read_csv(paths[2],index_col=0,parse_dates=True).iloc[:,0]
    else:
        import yfinance as yf
        raw=yf.download(list(TICKERS),start='2014-01-01',end='2026-10-10',
                        auto_adjust=True,interval='1d',progress=False,threads=False)
        if raw is None or raw.empty: raise RuntimeError('Yahoo ETF price download failed')
        opened=_extract(raw,'Open',TICKERS); closed=_extract(raw,'Close',TICKERS)
        y=yf.download('^IRX',start='2014-01-01',end='2026-10-10',
                      auto_adjust=False,interval='1d',progress=False,threads=False)
        if y is None or y.empty: raise RuntimeError('Yahoo 13-week yield download failed')
        yield_pct=_extract(y,'Close',('^IRX',)).iloc[:,0].rename('annual_yield_pct')
        opened.to_csv(paths[0]);closed.to_csv(paths[1]);yield_pct.to_frame().to_csv(paths[2])
        manifest={'source':'Yahoo Finance using yfinance','adjustment':'auto_adjust=True',
                  'download_start':'2014-01-01','download_end_exclusive':'2026-10-10',
                  'yfinance_version':getattr(yf,'__version__','unknown'),
                  'tickers':list(TICKERS),'yield_proxy':'^IRX',
                  'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
        (d/'manifest.json').write_text(json.dumps(manifest,indent=2))
    validate_prices(opened,closed,yield_pct)
    return opened,closed,yield_pct


def validate_prices(opened,closed,yields):
    for label,p in [('open',opened),('close',closed)]:
        assert set(TICKERS).issubset(p.columns),label+' missing ticker'
        assert p.index.is_unique and p.index.is_monotonic_increasing
        assert isinstance(p.index,pd.DatetimeIndex)
        assert (p.index.dayofweek<5).all(), label+' contains a weekend date'
        assert p.index.min() <= pd.Timestamp('2014-01-10')
        assert p.index.max() >= pd.Timestamp('2026-10-01')
        assert p.index.equals(opened.index),label+' date mismatch'
        assert p[list(TICKERS)].notna().all().all(),label+' incomplete data'
        assert (p[list(TICKERS)]>0).all().all()
        assert np.isfinite(p[list(TICKERS)].to_numpy()).all()
    assert yields.index.is_unique and yields.index.is_monotonic_increasing
    assert yields.dropna().between(-3,25).all(), 'Treasury yield units surprising'
    assert yields.loc[:'2026-09-30'].notna().any(), 'Missing risk-free-rate data'
    # Flag suspect outliers for inspection without assuming all crises are errors.
    jumps=closed[list(TICKERS)].pct_change().abs().max().max()
    assert jumps < 0.5, 'Price jump >50%; inspect corporate action adjustment'


def make_panel(opened,closed,yields):
    """Month m uses signal at end of m-1 and exit at first opening of m+1."""
    validate_prices(opened,closed,yields)
    month_close=closed.groupby(closed.index.to_period('M')).last()
    month_last=closed.index.to_series().groupby(closed.index.to_period('M')).last()
    month_first=opened.index.to_series().groupby(opened.index.to_period('M')).first()
    changes={k:month_close.pct_change(k) for k in (1,3,6)}
    vol=closed.pct_change().rolling(63,min_periods=63).std()*np.sqrt(252)
    dist=closed/closed.rolling(200,min_periods=200).mean()-1
    rows=[]
    for month in pd.period_range('2015-01','2026-09',freq='M'):
        prev=month-1; nxt=month+1
        assert all(x in month_first.index for x in (month,nxt))
        signal=pd.Timestamp(month_last.loc[prev])
        entry=pd.Timestamp(month_first.loc[month]); exit_=pd.Timestamp(month_first.loc[nxt])
        assert signal<entry<exit_
        known=yields.loc[:signal].dropna()
        assert len(known) and (signal-known.index[-1]).days<=7
        month_rf=(1+float(known.iloc[-1])/100)**(1/12)-1
        spy_return=opened.at[exit_,'SPY']/opened.at[entry,'SPY']-1
        for ticker in SECTORS:
            prev_sector=changes[3].at[prev,ticker]
            prev_spy=changes[3].at[prev,'SPY']
            sector_ret=opened.at[exit_,ticker]/opened.at[entry,ticker]-1
            row=dict(month=str(month),ticker=ticker,signal_date=signal,entry_date=entry,
                     exit_date=exit_,label=int(sector_ret>spy_return),
                     realised=sector_ret,spy_return=spy_return,riskfree=month_rf,
                     ret1=changes[1].at[prev,ticker],ret3=prev_sector,
                     ret6=changes[6].at[prev,ticker],vol63=vol.at[signal,ticker],
                     ma200_distance=dist.at[signal,ticker],
                     relative3=(1+prev_sector)/(1+prev_spy)-1)
            rows.append(row)
    panel=pd.DataFrame(rows)
    assert panel.month.nunique()==141 and len(panel)==141*9
    assert np.isfinite(panel[list(FEATURES)].to_numpy()).all()
    assert panel[['ret1','ret3','ret6','vol63']].notna().all().all()
    assert (panel.vol63>0).all()
    assert panel.label.isin([0,1]).all()
    assert (panel.signal_date<panel.entry_date).all() and (panel.entry_date<panel.exit_date).all()
    assert panel.groupby('month').ticker.nunique().eq(9).all()
    return panel


def walkforward(panel,start,end,params,rolling_months=None):
    forecasts=[]; importances=[]
    for p in pd.period_range(start,end,freq='M'):
        current=panel[panel.month==str(p)].sort_values('ticker')
        assert len(current)==9
        signal=current.signal_date.iloc[0]
        train=panel[(panel.exit_date<=signal)&(panel.month<str(p))].copy()
        if rolling_months is not None: train=train[train.month>=str(p-rolling_months)]
        assert len(train)>=216 and train.label.nunique()==2
        assert (train.exit_date<=signal).all(), 'Future label leaked into training!'
        forest=RandomForestClassifier(**params,max_features='sqrt',random_state=42,n_jobs=1)
        forest.fit(train[list(FEATURES)],train.label)
        positive=list(forest.classes_).index(1)
        prob=forest.predict_proba(current[list(FEATURES)])[:,positive]
        prior=float(train.label.mean())
        for (_,r),q in zip(current.iterrows(),prob):
            forecasts.append(dict(month=str(p),ticker=r.ticker,probability=float(q),
                                  reference_probability=prior,label=int(r.label),
                                  signal_date=signal,entry_date=r.entry_date,exit_date=r.exit_date,
                                  latest_training_exit=train.exit_date.max()))
        importances.append(dict(month=str(p),**dict(zip(FEATURES,forest.feature_importances_))))
    out=pd.DataFrame(forecasts)
    assert out.groupby('month').ticker.nunique().eq(9).all()
    assert (out.latest_training_exit<=out.signal_date).all()
    return out,pd.DataFrame(importances)


def choose_model(panel):
    rows=[]
    for name,params in CANDIDATES.items():
        pred,_=walkforward(panel,*VALIDATION,params)
        rows.append(dict(name=name,brier=brier_score_loss(pred.label,pred.probability),
                         naive_brier=brier_score_loss(pred.label,pred.reference_probability)))
    table=pd.DataFrame(rows).sort_values(['brier','name']).reset_index(drop=True)
    name=table.iloc[0]['name']
    return name,CANDIDATES[name],table


def select(panel,pred,method):
    choices={}
    for month,pred_month in pred.groupby('month',sort=True):
        if method=='forest':
            ranked=pred_month.sort_values(['probability','ticker'],ascending=[False,True])
        elif method=='momentum':
            ranked=panel[panel.month==month].sort_values(['ret6','ticker'],ascending=[False,True])
        else: raise ValueError('Unknown selection rule')
        tickers=tuple(ranked.ticker.head(3))
        assert len(tickers)==len(set(tickers))==3
        choices[month]=tickers
    return choices


def simulate(panel,choices,bps=7.5,spy=False):
    """Monthly drift, then rebalance. Charge both buys and sells by dollar traded."""
    universe=('SPY',) if spy else SECTORS
    before=pd.Series(0.,index=universe)
    wealth=1.; rows=[]
    for month in sorted(choices):
        data=panel[panel.month==month]
        weights=pd.Series(0.,index=universe)
        if spy: weights['SPY']=1.
        else:
            assert len(choices[month])==3
            weights.loc[list(choices[month])]=1/3
        traded=float((weights-before).abs().sum())
        cost=traded*bps/10000
        rs=(pd.Series({'SPY':data.spy_return.iloc[0]}) if spy
            else data.set_index('ticker').realised.reindex(universe))
        gross=1+float(np.dot(weights,rs))
        assert gross>0 and cost<0.01
        net=(1-cost)*gross-1
        wealth*=1+net
        before=weights*(1+rs)/gross
        assert abs(before.sum()-1)<1e-10
        rows.append(dict(month=month,valuation_date=data.exit_date.iloc[0],
                         return_net=net,nav=wealth,turnover=traded,cost=cost,
                         riskfree=data.riskfree.iloc[0],selected=choices[month]))
    out=pd.DataFrame(rows).set_index('month')
    if spy:
        assert out.turnover.iloc[0]==1
        assert (out.turnover.iloc[1:]==0).all()
    return out


def metrics(result):
    r=result.return_net
    vol=float(r.std(ddof=1)*np.sqrt(12))
    excess=r-result.riskfree
    nav=pd.concat([pd.Series([1.],index=['initial']),result.nav])
    dd=float((nav/nav.cummax()-1).min())
    return dict(CAGR=float(result.nav.iloc[-1]**(12/len(r))-1),
                volatility=vol,sharpe=float(excess.mean()*12/vol) if vol>0 else float('nan'),
                max_monthly_drawdown=dd,total_return=float(result.nav.iloc[-1]-1),
                annual_turnover=float(result.turnover.sum()*12/len(r)))


def forecast_diagnostics(pred):
    top=pred.sort_values(['month','probability','ticker'],ascending=[True,False,True]).groupby('month').head(3)
    stats={'brier':float(brier_score_loss(pred.label,pred.probability)),
           'naive_brier':float(brier_score_loss(pred.label,pred.reference_probability)),
           'classification_accuracy':float(((pred.probability>=.5)==pred.label).mean()),
           'top3_hit_rate':float(top.label.mean()),
           'random_top3_reference':float(pred.label.mean())}
    bins=pd.cut(pred.probability,np.linspace(0,1,6),include_lowest=True)
    reliability=pred.groupby(bins,observed=False).agg(n=('label','size'),
         mean_probability=('probability','mean'),observed_frequency=('label','mean'))
    return stats,reliability


def audit(panel,pred):
    assert pred.month.min()=='2022-01' and pred.month.max()=='2026-09'
    assert pred.month.nunique()==57 and len(pred)==513
    assert (pred.latest_training_exit<=pred.signal_date).all()
    assert (pred.signal_date<pred.entry_date).all()
    assert (pred.entry_date<pred.exit_date).all()
    dec=panel[panel.month=='2021-12'].iloc[0]
    jan=panel[panel.month=='2022-01'].iloc[0]
    assert dec.exit_date>jan.signal_date, 'December 2021 label must not tune Jan 2022 model'
    sep=panel[panel.month=='2026-09'].iloc[0]
    assert sep.exit_date.month==10 and sep.exit_date.year==2026
    return True


def block_bootstrap_ci(monthly_differences, block_months=6, draws=2000, seed=2026):
    """Circular block-bootstrap CI for a time-ordered monthly mean difference.

    We resample contiguous months, not individual ETF observations. This is
    an uncertainty diagnostic, not a formal guarantee of future performance.
    """
    x=np.asarray(monthly_differences,dtype=float)
    assert x.ndim==1 and len(x)>=24 and np.isfinite(x).all()
    assert 1<=block_months<=len(x) and 100<=draws<=10000
    rng=np.random.default_rng(seed)
    count=int(np.ceil(len(x)/block_months))
    starts=rng.integers(0,len(x),size=(draws,count))
    indices=((starts[:,:,None]+np.arange(block_months))%len(x)).reshape(draws,-1)[:,:len(x)]
    sampled_means=x[indices].mean(axis=1)
    low,high=np.quantile(sampled_means,[.025,.975])
    return {'mean_monthly_difference':float(x.mean()),
            'ci_low':float(low),'ci_high':float(high),
            'block_months':block_months,'draws':draws}


def top3_monthly_lift(pred):
    """Per-month uplift above same-month random sector selection baseline."""
    assert pred.groupby('month').ticker.nunique().eq(9).all()
    top=(pred.sort_values(['month','probability','ticker'],ascending=[True,False,True])
         .groupby('month',sort=True).head(3))
    best=top.groupby('month').label.mean()
    naive=pred.groupby('month').label.mean()
    assert best.index.equals(naive.index)
    return (best-naive).rename('top3_lift')
