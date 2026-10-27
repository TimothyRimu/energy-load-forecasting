"""Day-ahead hourly load forecast for a Portuguese energy cooperative (Loureiro dataset).
Target: mean kWh per reporting meter per hour. Walk-forward (expanding window, monthly folds)."""
import pandas as pd, numpy as np, json, sys
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
D = sys.argv[1]
e = pd.read_csv(D+'/loureiro_energy.csv', parse_dates=['Time']).set_index('Time')
w = pd.read_csv(D+'/weather_aveiro_final.csv', parse_dates=['Time']).set_index('Time')
act = e.notna().sum(axis=1)
y15 = e.mean(axis=1).where(act >= 100)            # mean per reporting meter; drop rows with <100 meters
y = y15.resample('h').mean()                       # hourly
W = w[['Avg_Temp','Avg_Rel_Humidity','Avg_Wind_Speed','Total_Global_Rad','Quantity_Precip']].resample('h').mean()
df = pd.DataFrame({'y': y}).join(W)
# reporting window: stable meter coverage (Oct-2022 .. May-2023)
df = df['2022-10-01':'2023-05-31'].copy()
full = df.index
df = df.reindex(pd.date_range(full.min(), full.max(), freq='h'))
# features available at forecast time (information up to 00:00 of the day before target hour => lag>=24h)
f = pd.DataFrame(index=df.index)
for L in (24, 48, 168): f[f'lag{L}'] = df.y.shift(L)
f['mean_prev24'] = df.y.shift(24).rolling(24, min_periods=12).mean()
f['hour'] = df.index.hour; f['dow'] = df.index.dayofweek
f['sin_h'] = np.sin(2*np.pi*f.hour/24); f['cos_h'] = np.cos(2*np.pi*f.hour/24)
f['weekend'] = (f.dow >= 5).astype(int)
for c in W.columns: f[c] = df[c]
ok = df.y.notna() & f[['lag24','lag168']].notna().all(axis=1)
X, Y = f[ok], df.y[ok]
def metrics(t, p):
    m = ~np.isnan(p); t, p = t[m], p[m]
    return dict(MAE=float(np.mean(abs(t-p))), RMSE=float(np.sqrt(np.mean((t-p)**2))),
                MAPE=float(np.mean(abs(t-p)/np.maximum(t,1e-3))*100), n=int(m.sum()))
months = pd.period_range('2022-12', '2023-05', freq='M')
res = {k: [] for k in ('yesterday','last_week','ridge','hgb')}; ys=[]; idx=[]; P={k:[] for k in res}
for m in months:
    te = X.index.to_period('M') == m
    tr = X.index < m.start_time
    if tr.sum() < 500 or te.sum()==0: continue
    Xt, Yt, Xe, Ye = X[tr], Y[tr], X[te], Y[te]
    P['yesterday'].append(Xe.lag24); P['last_week'].append(Xe.lag168)
    r = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2,3,12))).fit(Xt.fillna(Xt.median()), Yt)
    P['ridge'].append(pd.Series(r.predict(Xe.fillna(Xt.median())), Xe.index))
    g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=5, random_state=0).fit(Xt, Yt)
    P['hgb'].append(pd.Series(g.predict(Xe), Xe.index))
    ys.append(Ye)
yt = pd.concat(ys)
out = {}
for k in P:
    p = pd.concat(P[k]).reindex(yt.index)
    out[k] = metrics(yt.values, p.values)
    if k=='hgb': hg=p
# per-month for best model vs baseline
pm = {}
for m in months:
    s = yt.index.to_period('M')==m
    if s.sum()==0: continue
    pm[str(m)] = {k: round(metrics(yt[s].values, pd.concat(P[k]).reindex(yt.index)[s].values)['MAE'],4) for k in P}
out['per_month_MAE'] = pm
out['target_mean']=float(yt.mean())
json.dump(out, open('results.json','w'), indent=1)
yt.to_frame('actual').assign(hgb=hg, yesterday=pd.concat(P['yesterday']).reindex(yt.index), last_week=pd.concat(P['last_week']).reindex(yt.index)).to_csv('predictions.csv')
print(json.dumps(out, indent=1))
