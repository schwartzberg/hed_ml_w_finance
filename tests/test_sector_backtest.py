import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sector_backtest as m


def sample():
    dates=pd.bdate_range('2014-01-01','2026-10-09')
    rng=np.random.default_rng(11)
    market=rng.normal(.0002,.008,len(dates))
    daily=np.clip(market[:,None]+rng.normal(0,.006,(len(dates),10)),-.25,.25)
    closes=pd.DataFrame(100*np.exp(np.cumsum(np.log1p(daily),axis=0)),index=dates,columns=m.TICKERS)
    opens=closes*(1+rng.normal(0,.0008,closes.shape))
    yields=pd.Series(4.,index=dates)
    return opens,closes,yields


def test_calendar():
    opened,closed,yields=sample()
    panel=m.make_panel(opened,closed,yields)
    assert panel.month.nunique()==141
    row=panel.query("month == '2022-01'").iloc[0]
    assert row.signal_date==pd.Timestamp('2021-12-31')
    dec=panel.query("month == '2021-12'").iloc[0]
    assert dec.exit_date>row.signal_date
    p,_=m.walkforward(panel,'2022-01','2022-02',m.CANDIDATES['small_100'])
    assert (p.latest_training_exit<=p.signal_date).all()


def test_return_uses_open():
    op,cl,y=sample();p=m.make_panel(op,cl,y)
    r=p.query("month == '2022-01' and ticker == 'XLE'").iloc[0]
    expected=op.at[r.exit_date,'XLE']/op.at[r.entry_date,'XLE']-1
    assert np.isclose(expected,r.realised)
    assert r.label==int(expected>r.spy_return)


def test_cost_and_rebalance():
    op,cl,y=sample();p=m.make_panel(op,cl,y)
    ch={mo:('XLE','XLF','XLB') for mo in ('2022-01','2022-02','2022-03')}
    a=m.simulate(p,ch,bps=7.5)
    b=m.simulate(p,ch,bps=0)
    assert np.isclose(a.cost.iloc[0],.00075)
    assert a.nav.iloc[-1]<b.nav.iloc[-1]
    spy=m.simulate(p,{mo:('SPY',) for mo in ch},spy=True)
    assert spy.turnover.tolist()==[1.,0.,0.]


def test_full_final_calendar():
    op,cl,y=sample();p=m.make_panel(op,cl,y)
    pred,_=m.walkforward(p,'2022-01','2026-09',m.CANDIDATES['small_100'])
    assert m.audit(p,pred)
    ch=m.select(p,pred,'forest')
    assert len(ch)==57
    assert len(m.simulate(p,ch))==57


def test_forecasts_cannot_see_unfinished_outcomes():
    op,cl,y=sample(); panel=m.make_panel(op,cl,y)
    kwargs=m.CANDIDATES['small_100']
    original,_=m.walkforward(panel,'2022-01','2022-01',kwargs)
    altered=panel.copy()
    # The December 2021 outcome is unknown on the January 2022 signal date.
    # The entire 2022+ final period is also still unknown.
    mask=altered['month']>='2021-12'
    altered.loc[mask,'label']=1-altered.loc[mask,'label']
    changed,_=m.walkforward(altered,'2022-01','2022-01',kwargs)
    np.testing.assert_array_equal(original.probability, changed.probability)


def test_costs_apply_to_actual_rebalancing_turnover():
    op,cl,y=sample(); panel=m.make_panel(op,cl,y)
    held={str(mo):('XLB','XLE','XLF') for mo in pd.period_range('2022-01','2022-06',freq='M')}
    result=m.simulate(panel,held,bps=7.5)
    assert result.turnover.iloc[0]==1
    assert (result.turnover.iloc[1:]>0).all(), 'Rebalancing costs apply even if names unchanged'
    np.testing.assert_allclose(result.cost,result.turnover*7.5/10000)
    assert m.simulate(panel,held,bps=0).cost.eq(0).all()


def test_bootstrap_is_reproducible_and_zero_has_zero_interval():
    x=np.zeros(57)
    a=m.block_bootstrap_ci(x,block_months=6,draws=200,seed=24)
    b=m.block_bootstrap_ci(x,block_months=6,draws=200,seed=24)
    assert a==b
    assert a['ci_low']==0 and a['ci_high']==0
    r=m.block_bootstrap_ci(np.arange(57,dtype=float)/1000,draws=200)
    assert r['ci_low']<=r['mean_monthly_difference']<=r['ci_high']


def test_top3_lift_respects_each_months_cross_section():
    months=['2022-01','2022-02']
    rows=[]
    for mo in months:
        for i,ticker in enumerate(m.SECTORS):
            rows.append(dict(month=mo,ticker=ticker,probability=float(9-i)/10,label=int(i<3)))
    pred=pd.DataFrame(rows)
    result=m.top3_monthly_lift(pred)
    np.testing.assert_allclose(result.to_numpy(),[2/3,2/3])


def test_max_drawdown_includes_initial_capital():
    results=pd.DataFrame({'return_net':[-.2,.1], 'nav':[.8,.88],
         'riskfree':[0,0], 'turnover':[1.,0.]},index=['2022-01','2022-02'])
    result=m.metrics(results)
    assert np.isclose(result['max_monthly_drawdown'],-.2)


def test_cache_manifest_rejects_tampering(tmp_path):
    import json,hashlib
    op,cl,y=sample()
    paths=[tmp_path/'adjusted_open.csv',tmp_path/'adjusted_close.csv',tmp_path/'treasury_yield.csv']
    op.to_csv(paths[0]);cl.to_csv(paths[1]);y.to_frame('annual_yield_pct').to_csv(paths[2])
    (tmp_path/'manifest.json').write_text(json.dumps({
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}))
    _,_,_=m.load_prices(tmp_path)
    with paths[0].open('a') as fp:fp.write('\nTAMPERED')
    import pytest
    with pytest.raises(RuntimeError,match='integrity'):
        m.load_prices(tmp_path)
