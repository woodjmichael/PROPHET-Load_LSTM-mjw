PROPHET_LSTM_ENV = True

from numpy.random import randint
import pandas as pd
from pandas import Timestamp as ts
from pandas import Timedelta as td
from math import nan

if not PROPHET_LSTM_ENV:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import plotly.graph_objects as go
    import plotly.io as pio
    from tabulate import tabulate
    
#project_path = r'/home/mjw/Code/PROPHET-Load_LSTM-mjw/'
#project_path = r'/home/mjw/Code/load-forecast/'
project_path = r'C:\Users\Admin\Code\load-forecast/'

limit_days = list(range(7))
prior_days = 7
dx = 0.02
resamp='15min'
verbose=False
calc_errors=True

"""
Impianto 4
"""   
    
filepath = project_path + 'data/Impianto_4_clean.csv'
load_col = 'Potenza'

t_test_begin = pd.Timestamp('2019-9-13 0:00')

persist_filepath =   project_path + r'forecasts/impianto_4/PROPHET-Load_LSTM/u144_ud36_nb96_d0.1/test_forecasts.csv'
forecast_filepath2 = project_path + r'forecasts/impianto_4/PROPHET-Load_LSTM/u144_ud36_nb96_d0.1/test_forecasts.csv'
forecast_filepath3 = None #project_path + r'forecasts/impianto_4/PROPHET-Load_LSTM/u24_ud36_nb432_d0/test_forecasts.csv'
forecast_filepath4 = None #project_path + r'forecasts/impianto_4/PROPHET-Load_LSTM/u432_ud24_nb288_d0/test_forecasts.csv'

col_t = 'timestamp_forecast'
col_t_update = 'timestamp_forecast_update'
col_pred = 'predicted_activepower_ev_1'
col_pers = 'persist'
col_load = 'power'
forecast_length_h = 36
    
"""
JPL
""" 

# t_test_begin = pd.Timestamp('2019-12-26 0:00')

# filepath = project_path + r'data/all_JPL_v5.csv'
# load_col = 'power'

# forecast_filepath = project_path + r'forecasts/jpl_ev/bayes-lstm-persist/u24-96_d0.1_n48_f4/test_forecasts_fill-persist.csv'
# forecast_length_h = 36

# col_t = 'timestamp'
# col_t_update = 'timestamp_update'
# col_pred = 'Pred'
# col_pers = 'Persist'
# col_load = 'Load'

def dxrange(start,stop,step):
    return [round(x*step,3) for x in range(int(start/step), int(stop/step))]


def get_ppd():
    return {'1h': 24, '15min': 96}[resamp]


def minmax_normalize(df):
    return (df - df.min()) / (df.max() - df.min())


def plot_weekly(df, freq=None, limit_days=False, figsize=(8, 5)):
    pd.options.plotting.backend = "matplotlib"

    freq = df.index.freqstr if df.index.freqstr is not None else freq
    days = 7 if limit_days == False else limit_days

    ppd = {'1H': 24, 'H': 24, '15MIN': 96}[freq.upper()]
    ppw = ppd * days

    if df.index.weekday[0] != 0:
        df = df[ppd * (days - df.index.weekday[0]):]
    df = df[:ppw * (len(df) // ppw)]

    pd.DataFrame(df.Load.values.reshape(-1, ppw).T).plot(alpha=0.05, legend=False, figsize=figsize)


def calc_mae(y: pd.Series = None, yhat: pd.Series = None, err: pd.Series = None, round=3):
    if err is None:
        return (yhat - y).abs().mean().round(round)
    else:
        return err.abs().mean().round(round)


def calc_skill(y: pd.Series, ybench: pd.Series, yhat: pd.Series):
    mae_forecast = calc_mae(y, yhat, round=9)
    mae_benchmark = calc_mae(y, ybench, round=9)
    if mae_benchmark < 0.001:
        print('Benchmark MAE is probably artifically low or all values are zeros (night)')
        return nan
    return round(1 - mae_forecast / mae_benchmark, 3)


def calculate_daily_skills(df_pred, err_ybench_col='ErrorPersist', err_yhat_col='ErrorPred', update_hours=None,
                           round_digits=3):
    df = df_pred.copy(deep=True)

    if update_hours != None:
        print('/// Experimental having update hours other than 0:00')

    for t in df.timestamp_update.unique():
        if (update_hours is None) or (t.hour in update_hours):
            daily_skill = 1 - df.loc[df.timestamp_update == t, err_yhat_col].abs().mean() \
                          / df.loc[df.timestamp_update == t, err_ybench_col].abs().mean()

            df.loc[df.timestamp_update == t, 'SkillDaily'] = daily_skill

    return round(len(df.loc[df.SkillDaily > 0]) / len(df), round_digits)


def resample_forecast(lstm, freq='1h', forecast_length=24):
    lstm = lstm.copy()
    lstm_1h = pd.DataFrame()
    for t_up in lstm.timestamp_update.unique():
        lstm_up = lstm.loc[lstm.timestamp_update == t_up].copy()
        lstm_up.set_index('timestamp', inplace=True)
        lstm_up = lstm_up[['Pred', 'Load', 'Persist']].copy()
        # print(t,f'Days in lstm _up = {len(lstm_up.index.dayofyear.unique())}') # DEBUG
        if len(lstm_up.index.dayofyear.unique()) == 2:  # 2 different days in forecast
            day = lstm_up.index.dayofyear.unique()[0]
            lstm_day1 = lstm_up.loc[lstm_up.index.dayofyear == day].resample(freq).mean()
            day = lstm_up.index.dayofyear.unique()[1]
            lstm_day2 = lstm_up.loc[lstm_up.index.dayofyear == day].resample(freq).mean()
            lstm_up = pd.concat((lstm_day1, lstm_day2))
        elif len(lstm_up.index.dayofyear.unique()) == 3:  # 3 different days in forecast
            day = lstm_up.index.dayofyear.unique()[0]
            lstm_day1 = lstm_up.loc[lstm_up.index.dayofyear == day].resample(freq).mean()
            day = lstm_up.index.dayofyear.unique()[1]
            lstm_day2 = lstm_up.loc[lstm_up.index.dayofyear == day].resample(freq).mean()
            day = lstm_up.index.dayofyear.unique()[2]
            lstm_day3 = lstm_up.loc[lstm_up.index.dayofyear == day].resample(freq).mean()
            lstm_up = pd.concat((lstm_day1, lstm_day2, lstm_day3))
        else:  # only 1 day in forecast
            lstm_up = lstm_up.resample('1h').mean()
        lstm_up = lstm_up.reset_index()
        lstm_up.insert(0, 'timestamp_update', t_up)
        len(lstm_up) == forecast_length
        lstm_1h = pd.concat((lstm_1h, lstm_up), ignore_index=True)
    return lstm_1h


def calc_mean_of_distribution(X): 
    csum = 0
    X = X.fillna(0)
    for i, x in zip(X.index, X.values):
        csum = csum + i * x
    try: 
        return csum / (X.sum())
    except:
        return nan


def create_distribution(pred, col_name, dx, plot_i=None):    
    ppd = get_ppd()
    idx = dxrange(-1, 1, dx) # [-1,-1+dx,-1+2*dx,..1-dx]
    err_dist = pd.DataFrame()
    for i in range(ppd):
        y = pred.loc[(pred.timestamp.dt.hour * 4 + pred.timestamp.dt.minute // 15) == i, col_name]
        X = []
        for x in idx:
            X.append(y[(y >= x) & (y < (x + dx))].count() / len(y))
        X = pd.DataFrame(X, index=idx)
        err_dist[f'i{i}'] = X

    if plot_i: err_dist[f'i{plot_i}'].plot()
    
    if verbose:
        print('Err dist shape', err_dist.shape)

    return err_dist

def calc_results(df, dfM=None):
    res = {

        'nMAE Persist %': calc_mae(err=df.ErrorPersist) * 100,
        'nMAE Pred %': calc_mae(err=df.ErrorPred) * 100,
        'nMAE Bayes %': calc_mae(err=df.ErrorBayes) * 100,
    }
    if dfM is not None:
        res.update({'nMAE Bayes Multi %': calc_mae(err=dfM.ErrorBayes) * 100})

    res.update({
        'Skill Pred %': calc_skill(df.Load, df.Persist, df.Pred) * 100,
        'Skill Bayes %': calc_skill(df.Load, df.Persist, df.Bayes) * 100})
    if dfM is not None:
        res.update({'Skill Bayes Multi %': calc_skill(dfM.Load, dfM.Persist, dfM.Bayes) * 100})

    res.update({
        'Positive Daily Skill Pred %': calculate_daily_skills(df, 'ErrorPersist', 'ErrorPred') * 100,
        'Positive Daily Skill Bayes %': calculate_daily_skills(df, 'ErrorPersist', 'ErrorBayes') * 100})
    if dfM is not None:
        res.update(
            {'Positive Daily Skill Bayes Multi %': calculate_daily_skills(dfM, 'ErrorPersist', 'ErrorBayes') * 100})

    return res


# # Priors

def read_historical_measurements(filepath,
                                 load_col,
                                 t_test_begin,
                                 resamp='15min',
                                 limit_days=list(range(7)),
                                 prior_days=7,
                                 plot=False):
    
    meas = pd.read_csv(filepath,
                       index_col=0,
                       parse_dates=True,
                       comment='#').resample(resamp).mean()

    meas = meas.rename(columns={load_col: 'Load'})

    meas = meas[['Load']]

    meas_max = meas.Load.max()
    meas = meas / meas_max

    meas = meas[meas.index.weekday.isin(limit_days)]

    if plot: plot_weekly(meas, limit_days=len(limit_days))

    meas = meas[:t_test_begin][:-1]  # only make priors on training set values

    return meas, meas_max


# ## Build from measurements

def build_priors(meas, prior_days, limit_days, dx, verbose=False):
    ppd = get_ppd()
    
    # begin on a monday
    meas_mon = meas[ppd * (len(limit_days) - meas.index.weekday[0]):].copy() if meas.index.weekday[
                                                                                    0] != 0 else meas.copy()

    # integer number of days
    meas_vals = meas_mon[:(prior_days * ppd) * (len(meas_mon) // (prior_days * ppd))].values

    # matrix
    meas_mat = meas_vals.reshape(-1, prior_days * ppd)

    meas_mat = pd.DataFrame(meas_mat)

    if verbose: print('Meas mat shape', meas_mat.shape)
    
    idx = dxrange(0,1,dx) # [0,dx,2*dx,..1-dx]

    priors = pd.DataFrame()

    if ppd == 24:
        for h in range(prior_days * ppd):
            y = meas_mat[h]
            X = []
            for x in idx:
                X.append(y[(y >= x) & (y < (x + dx))].count() / len(y))
            X = pd.DataFrame({f'h{h}': X}, index=idx)
            priors = pd.concat((priors, X), axis=1)
    elif ppd == 96:
        for i in range(prior_days * ppd):
            y = meas_mat[i]
            X = []
            for x in idx:
                if len(y) in [0, nan]:
                    print(f'len(y) is {len(y)}')
                X.append(y[(y >= x) & (y < (x + dx))].count() / len(y))
            X = pd.DataFrame({f'i{i}': X}, index=idx)
            priors = pd.concat((priors, X), axis=1)

    if verbose: print('Priors shape', priors.shape)

    return priors, meas_mat


# ### Bagging

def bag_priors(meas_mat, dx, prior_days):
    ppd = get_ppd()
    
    n_bags = 500
    bag_size = 5

    bags_of_new_bags = []
    for hw in range(meas_mat.shape[1]):  # hour of week
        new_bags = []
        for j in range(n_bags):
            idx_bag = randint(meas_mat.shape[0], size=bag_size)
            meas_1h = meas_mat[hw]
            new_bag = meas_1h.iloc[idx_bag].mean()

            new_bags.append(new_bag)
        bags_of_new_bags.append(new_bags)

    bags_of_new_bags = pd.DataFrame(bags_of_new_bags).T

    meas_mat_bagged = pd.concat((meas_mat, bags_of_new_bags), axis=0, ignore_index=True)

    idx = dxrange(0,1,dx)  # [0,dx,2*dx,..1-dx]

    priors_bagged = pd.DataFrame()

    for h in range(prior_days * ppd):
        y = meas_mat_bagged[h]
        X = []
        for x in idx:
            X.append(y[(y >= x) & (y < (x + dx))].count() / len(y))
        X = pd.DataFrame({f'i{h}': X}, index=idx)
        priors_bagged = pd.concat((priors_bagged, X), axis=1)

    # d,h=0,7
    # priors_bagged[f'h{(d*ppd+h)}'].plot()

    # priors = priors_bagged.copy(deep=True)

    return priors_bagged


# ### Quantiles
# For plots

def calc_quantiles(meas_mat, prior_days, plot=False):
    ppd = get_ppd()
    prior_qts = pd.DataFrame({'p99': [], 'p95': [], 'p75': [], 'Mean': [], 'p50': [], 'p25': [], 'p5': [], 'p1': []})
    for i in range(prior_days * ppd):
        prior_qts.loc[len(prior_qts)] = [meas_mat.iloc[:, i].quantile(0.99),
                                         meas_mat.iloc[:, i].quantile(0.95),
                                         meas_mat.iloc[:, i].quantile(0.75),
                                         meas_mat.iloc[:, i].mean(),
                                         meas_mat.iloc[:, i].quantile(0.50),
                                         meas_mat.iloc[:, i].quantile(0.25),
                                         meas_mat.iloc[:, i].quantile(0.05),
                                         meas_mat.iloc[:, i].quantile(0.01), ]

    if not PROPHET_LSTM_ENV:
        if plot:
            plt.figure(figsize=(10, 6))
            plt.fill_between(prior_qts.index, prior_qts['p99'], prior_qts['p1'], color=cm.viridis(0.9), alpha=0.4,
                             label='p1-p99')
            plt.fill_between(prior_qts.index, prior_qts['p95'], prior_qts['p5'], color=cm.viridis(0.5), alpha=0.4,
                             label='p5-p95')
            plt.fill_between(prior_qts.index, prior_qts['p75'], prior_qts['p25'], color=cm.viridis(0.3), alpha=0.6,
                             label='p25-p75')
            plt.plot(prior_qts.index, prior_qts['p50'], color=cm.viridis(0.1), label='p50')
            plt.plot(prior_qts.index, prior_qts['Mean'], color='black', linestyle='--', label='Mean', alpha=0.7)

            plt.title('Intervals of Priors')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            plt.show()

    return prior_qts


# # Likelihoods


def read_forecasts(filepath,
                   forecast_length_h,
                   meas_max,
                   col_t='timestamp',
                   col_t_update='timestamp_update',
                   col_pred='Pred',
                   col_pers='Persist',
                   col_load='Load',
                   resamp='15min'):
    # filepath = r'forecasts/jpl_ev/bayes-lstm-persist/u24-96_d0.1_n48_f4'
    # forecast_length_h = 36

    pred = pd.read_csv(filepath,
                       # filepath+r'all_forecasts_nout144_1h.csv',
                       # index_col=0,
                       usecols=[col_t_update, col_t, col_pred, col_pers, col_load],
                       parse_dates=[col_t_update, col_t])

    pred = pred.rename(columns={col_t: 'timestamp',
                                col_t_update: 'timestamp_update',
                                col_pred: 'Pred',
                                col_pers: 'Persist',
                                col_load: 'Load'})

    if pred.timestamp.diff().mode()[0] < td(resamp):
        pred = resample_forecast(pred, forecast_length=forecast_length_h)
        pred.to_csv(filepath + r'/test_forecasts_fill-persist_1h.csv')

    if pred.timestamp.dt.tz is not None:
        pred.timestamp = pred.timestamp.dt.tz_convert(None)
        pred.timestamp_update = pred.timestamp_update.dt.tz_convert(None)

    t_first_monday = pred.loc[pred.timestamp_update.dt.dayofweek == 0, 'timestamp_update'].iloc[0]
    pred = pred.loc[pred.timestamp_update.values >= t_first_monday, :]

    #pred = pred.rename(columns={'Pred': 'LSTM'})

    pred['Load'] = pred.Load / meas_max
    pred['Persist'] = pred.Persist / meas_max
    pred['Pred'] = pred.Pred / meas_max

    pred['ErrorPred'] = pred.Load - pred.Pred
    pred['ErrorPersist'] = pred.Load - pred.Persist

    return pred


def plot_bayes(load_val, prior_h, pred_val1, likelihood1_h, pred_val2=None, likelihood2_h=None, bayes_val=None,
               bayes=None):
    if not PROPHET_LSTM_ENV:
        plt.figure(figsize=(10, 6))

        plt.fill_between(prior_h.index, [0] * len(prior_h), prior_h, color='orange', label='Prior PDF', alpha=0.4,
                         linewidth=3)
        plt.fill_between(likelihood1_h.index, [0] * len(likelihood1_h), likelihood1_h, color='blue',
                         label='Forecast 1 PDF', alpha=0.2, linewidth=3)
        if likelihood2_h is not None:
            plt.fill_between(likelihood2_h.index, [0] * len(likelihood2_h), likelihood2_h, color='red',
                             label='Forecast 2 PDF', alpha=0.2, linewidth=3)
        plt.fill_between(bayes.index, [0] * len(bayes), bayes, color='lightgreen', label='Bayes PDF', alpha=0.4,
                         linewidth=3)

        plt.plot([pred_val1], [0], 'blue', label='Forecast 1', marker='x')
        if pred_val2 is not None:
            plt.plot([pred_val2], [0], 'red', label='Forecast 2', marker='x')
        plt.plot([bayes_val], [0], 'green', label='Bayes', marker='x')

        plt.plot([load_val], [0], 'black', label='Load', marker='.')

        plt.legend(loc='upper left')
        plt.xlabel('Load [p.u.]')
        plt.xlim(load_val - 0.25, load_val + 0.25)
        plt.ylabel('Probability')
        plt.title(
            f'Bayes Inference')  # (t_now={t_now.strftime("%Y-%m-%d %H:%M")}, t_predict={t_predict.strftime("%H:%M")})')
        plt.show()
        

def plot_bayes_3(load_val,
               prior_h,
               pred_val1,
               likelihood1_h,
               pred_val2=None,
               likelihood2_h=None,
               pred_val3=None,
               likelihood3_h=None,               
               bayes_val=None,
               bayes=None):
    if not PROPHET_LSTM_ENV:
        plt.figure(figsize=(10, 6))

        plt.fill_between(prior_h.index, [0] * len(prior_h), prior_h, color='orange', label='Prior PDF', alpha=0.4,
                         linewidth=3)
        plt.fill_between(likelihood1_h.index, [0] * len(likelihood1_h), likelihood1_h, color='blue',
                         label='Forecast 1 PDF', alpha=0.2, linewidth=3)
        if likelihood2_h is not None:
            plt.fill_between(likelihood2_h.index, [0] * len(likelihood2_h), likelihood2_h, color='red',
                             label='Forecast 2 PDF', alpha=0.2, linewidth=3)
        if likelihood3_h is not None:
            plt.fill_between(likelihood3_h.index, [0] * len(likelihood3_h), likelihood3_h, color='cyan',
                             label='Forecast 3 PDF', alpha=0.2, linewidth=3)    
        plt.fill_between(bayes.index, [0] * len(bayes), bayes, color='lightgreen', label='Bayes PDF', alpha=0.4,
                         linewidth=3)

        plt.plot([pred_val1], [0], 'blue', label='Forecast 1', marker='x')
        if pred_val2 is not None:
            plt.plot([pred_val2], [0], 'red', label='Forecast 2', marker='x')
        if pred_val3 is not None:
            plt.plot([pred_val3], [0], 'cyan', label='Forecast 3', marker='x')            
            
        plt.plot([bayes_val], [0], 'green', label='Bayes', marker='x')

        plt.plot([load_val], [0], 'black', label='Load', marker='.')

        plt.legend(loc='upper left')
        plt.xlabel('Load [p.u.]')
        plt.xlim(load_val - 0.25, load_val + 0.25)
        plt.ylabel('Probability')
        plt.title(
            f'Bayes Inference')  # (t_now={t_now.strftime("%Y-%m-%d %H:%M")}, t_predict={t_predict.strftime("%H:%M")})')
        plt.show()    
           
        
def plot_bayes_4(load_val,
               prior_h,
               pred_val1,
               likelihood1_h,
               pred_val2=None,
               likelihood2_h=None,
               pred_val3=None,
               likelihood3_h=None,               
               pred_val4=None,
               likelihood4_h=None,               
               bayes_val=None,
               bayes=None):
    if not PROPHET_LSTM_ENV:
        plt.figure(figsize=(10, 6))

        plt.fill_between(prior_h.index, [0] * len(prior_h), prior_h, color='orange', label='Prior PDF', alpha=0.4,
                         linewidth=3)
        plt.fill_between(likelihood1_h.index, [0] * len(likelihood1_h), likelihood1_h, color='blue',
                         label='Forecast 1 PDF', alpha=0.2, linewidth=3)
        if likelihood2_h is not None:
            plt.fill_between(likelihood2_h.index, [0] * len(likelihood2_h), likelihood2_h, color='red',
                             label='Forecast 2 PDF', alpha=0.2, linewidth=3)
        if likelihood3_h is not None:
            plt.fill_between(likelihood3_h.index, [0] * len(likelihood3_h), likelihood3_h, color='cyan',
                             label='Forecast 3 PDF', alpha=0.2, linewidth=3)    
        if likelihood4_h is not None:
            plt.fill_between(likelihood4_h.index, [0] * len(likelihood4_h), likelihood4_h, color='gray',
                             label='Forecast 4 PDF', alpha=0.2, linewidth=3)    
        plt.fill_between(bayes.index, [0] * len(bayes), bayes, color='lightgreen', label='Bayes PDF', alpha=0.4,
                         linewidth=3)

        plt.plot([pred_val1], [0], 'blue', label='Forecast 1', marker='x')
        
        if pred_val2 is not None:
            plt.plot([pred_val2], [0], 'red', label='Forecast 2', marker='x')
        
        if pred_val3 is not None:
            plt.plot([pred_val3], [0], 'cyan', label='Forecast 3', marker='x')            
        
        if pred_val4 is not None:
            plt.plot([pred_val4], [0], 'cyan', label='Forecast 4', marker='x')            
            
        plt.plot([bayes_val], [0], 'green', label='Bayes', marker='x')

        plt.plot([load_val], [0], 'black', label='Load', marker='.')

        plt.legend(loc='upper left')
        plt.xlabel('Load [p.u.]')
        plt.xlim(load_val - 0.25, load_val + 0.25)
        plt.ylabel('Probability')
        plt.title(
            f'Bayes Inference')  # (t_now={t_now.strftime("%Y-%m-%d %H:%M")}, t_predict={t_predict.strftime("%H:%M")})')
        plt.show()                


def calc_posterior(t, priors, prior_days, ppd, pred_val1, likelihood1, pred_val2=None, likelihood2=None,
                   plot_with_true_val=None):
    dx = round( likelihood1.index.to_series().diff().mode()[0], 3)
    dx_inv = 1 / dx

    if ppd == 24:
        h = f'h{t.hour}'
        if prior_days > 1:
            hw = f'h{t.weekday() * 24 + t.hour}'
        else:
            hw = f'h{t.hour}'
        likelihood1_h = likelihood1[h].copy()
        prior_h = priors[hw].copy()
    else:
        qh = f'i{t.hour * 4 + t.minute // 15}'
        if prior_days > 1:
            qhw = f'i{t.weekday() * 96 + t.hour * 4 + t.minute // 15}'
        else:
            qhw = f'i{t.hour * 4 + t.minute // 15}'
        likelihood1_h = likelihood1[qh].copy()
        prior_h = priors[qhw].copy()

    if pred_val1 == nan:
        print('Prediction columns begins with NaN')
        return nan

    # pred_val is added to index (0.0, 0.01, 0.02 etc) to shift the distribution
    # new shifted index must be rounded to nearest bin
    # therefore round pred_val to nearest bin, then to nearest 0.01 (to remove floating point errors)
    pred_val1 = round(round(pred_val1 * dx_inv) * dx, 2)  # faster than: round(pred_val/dx)*dx
    likelihood1_h.index = [round(x+pred_val1,2) for x in likelihood1_h.index] # round(likelihood1_h.index + pred_val1, 2)
    likelihood1_h = likelihood1_h[(likelihood1_h.index >= 0) & (likelihood1_h.index <= 1)]
    posterior = likelihood1_h * prior_h / ((likelihood1_h * prior_h).sum())  # should divide by dx, but then bayes>>1

    if pred_val2 is not None:
        if pred_val2 == nan:
            print('Prediction columns begins with NaN')
            pred_val2 = None
        else:
            pred_val2 = round(round(pred_val2 * dx_inv) * dx, 2)
            likelihood2_h = likelihood2[h].copy() if ppd == 24 else likelihood2[qh].copy()
            likelihood2_h.index = [round(x+pred_val2,2) for x in likelihood2_h.index] # round(likelihood2_h.index + pred_val2, 2)
            likelihood2_h = likelihood2_h[(likelihood2_h.index >= 0) & (likelihood2_h.index <= 1)]
            posterior = likelihood1_h * likelihood2_h * prior_h / ((likelihood1_h * likelihood1_h * prior_h).sum())

    posterior_val = calc_mean_of_distribution(posterior)

    if plot_with_true_val is not None:
        if pred_val2 is None:
            plot_bayes(plot_with_true_val, prior_h, pred_val1, likelihood1_h, bayes_val=posterior_val, bayes=posterior)
        else:
            plot_bayes(plot_with_true_val, prior_h, pred_val1, likelihood1_h, pred_val2, likelihood2_h, posterior_val,
                       posterior)

    return posterior_val

def calc_posterior_3(t,
                     priors,
                     prior_days,
                     ppd,
                     pred_val1,
                     likelihood1, 
                     pred_val2=None, 
                     likelihood2=None,
                     pred_val3=None,
                     likelihood3=None,
                     plot_with_true_val=None):
    dx = round( likelihood1.index.to_series().diff().mode()[0], 3)
    dx_inv = 1 / dx

    if ppd == 24:
        h = f'h{t.hour}'
        if prior_days > 1:
            hw = f'h{t.weekday() * 24 + t.hour}'
        else:
            hw = f'h{t.hour}'
        likelihood1_h = likelihood1[h].copy()
        prior_h = priors[hw].copy()
    else:
        qh = f'i{t.hour * 4 + t.minute // 15}'
        if prior_days > 1:
            qhw = f'i{t.weekday() * 96 + t.hour * 4 + t.minute // 15}'
        else:
            qhw = f'i{t.hour * 4 + t.minute // 15}'
        likelihood1_h = likelihood1[qh].copy()
        prior_h = priors[qhw].copy()

    if pred_val1 == nan:
        print('Prediction columns begins with NaN')
        return nan


    # pred_val is added to index (0.0, 0.01, 0.02 etc) to shift the distribution
    # new shifted index must be rounded to nearest bin
    # therefore round pred_val to nearest bin, then to nearest 0.01 (to remove floating point errors)
    pred_val1 = round(round(pred_val1 * dx_inv) * dx, 2)  # faster than: round(pred_val/dx)*dx
    likelihood1_h.index = [round(x+pred_val1,2) for x in likelihood1_h.index] # round(likelihood1_h.index + pred_val1, 2)
    likelihood1_h = likelihood1_h[(likelihood1_h.index >= 0) & (likelihood1_h.index <= 1)]
    posterior = likelihood1_h * prior_h / ((likelihood1_h * prior_h).sum())  # should divide by dx, but then bayes>>1
    if pred_val2 is not None:
        if pred_val2 == nan:
            print('Prediction columns begins with NaN')
            pred_val2 = None
        else:
            pred_val2 = round(round(pred_val2 * dx_inv) * dx, 2)
            likelihood2_h = likelihood2[h].copy() if ppd == 24 else likelihood2[qh].copy()
            likelihood2_h.index = [round(x+pred_val2,2) for x in likelihood2_h.index] # round(likelihood2_h.index + pred_val2, 2)
            likelihood2_h = likelihood2_h[(likelihood2_h.index >= 0) & (likelihood2_h.index <= 1)]
            posterior = likelihood1_h * likelihood2_h * prior_h / ((likelihood1_h * likelihood1_h * prior_h).sum())
    if pred_val3 is not None:
            if pred_val3==nan:
                print('Prediction columns begins with NaN')
                pred_val3 = None
            else:
                pred_val3 = round(round(pred_val3 * dx_inv) * dx, 2)
                likelihood3_h = likelihood3[h].copy() if ppd == 24 else likelihood3[qh].copy()
                likelihood3_h.index = [round(x+pred_val3,2) for x in likelihood3_h.index] # round(likelihood2_h.index + pred_val2, 2)
                likelihood3_h = likelihood3_h[(likelihood3_h.index >= 0) & (likelihood3_h.index <= 1)]
                posterior = likelihood1_h * likelihood2_h * likelihood3_h * prior_h \
                    / ((likelihood1_h * likelihood1_h * likelihood3_h * prior_h).sum())            

    posterior_val = calc_mean_of_distribution(posterior)

    if plot_with_true_val is not None:
        if pred_val2 is None:
            plot_bayes(plot_with_true_val, prior_h, pred_val1, likelihood1_h, bayes_val=posterior_val, bayes=posterior)
        elif pred_val3 is None:
            plot_bayes(plot_with_true_val, prior_h, pred_val1, likelihood1_h, pred_val2, likelihood2_h, posterior_val,
                       posterior)
        else:
            plot_bayes_3(plot_with_true_val,
                         prior_h,
                         pred_val1, likelihood1_h, 
                         pred_val2, likelihood2_h,
                         pred_val3, likelihood3_h,
                         posterior_val,
                         posterior)

    return posterior_val


def calc_posterior_4(t,
                     priors,
                     prior_days,
                     pred_val1,
                     likelihood1, 
                     pred_val2=None, 
                     likelihood2=None,
                     pred_val3=None,
                     likelihood3=None,
                     pred_val4=None,
                     likelihood4=None,
                     plot_with_true_val=None):
    ppd = get_ppd()
    dx = round( likelihood1.index.to_series().diff().mode()[0], 3)
    dx_inv = 1 / dx

    if ppd == 24:
        h = f'h{t.hour}'
        if prior_days > 1:
            hw = f'h{t.weekday() * 24 + t.hour}'
        else:
            hw = f'h{t.hour}'
        likelihood1_h = likelihood1[h].copy()
        prior_h = priors[hw].copy()
    else:
        qh = f'i{t.hour * 4 + t.minute // 15}'
        if prior_days > 1:
            qhw = f'i{t.weekday() * 96 + t.hour * 4 + t.minute // 15}'
        else:
            qhw = f'i{t.hour * 4 + t.minute // 15}'
        likelihood1_h = likelihood1[qh].copy()
        prior_h = priors[qhw].copy()

    if pred_val1 == nan:
        print('Prediction columns begins with NaN')
        return nan


    # pred_val is added to index (0.0, 0.01, 0.02 etc) to shift the distribution
    # new shifted index must be rounded to nearest bin
    # therefore round pred_val to nearest bin, then to nearest 0.01 (to remove floating point errors)
    pred_val1 = round(round(pred_val1 * dx_inv) * dx, 2)  # faster than: round(pred_val/dx)*dx
    likelihood1_h.index = [round(x+pred_val1,2) for x in likelihood1_h.index] # round(likelihood1_h.index + pred_val1, 2)
    likelihood1_h = likelihood1_h[(likelihood1_h.index >= 0) & (likelihood1_h.index <= 1)]
    posterior = likelihood1_h * prior_h / ((likelihood1_h * prior_h).sum())  # should divide by dx, but then bayes>>1
    if pred_val2 is not None:
        if pred_val2 == nan:
            print('Prediction columns begins with NaN')
            pred_val2 = None
        else:
            pred_val2 = round(round(pred_val2 * dx_inv) * dx, 2)
            likelihood2_h = likelihood2[h].copy() if ppd == 24 else likelihood2[qh].copy()
            likelihood2_h.index = [round(x+pred_val2,2) for x in likelihood2_h.index] # round(likelihood2_h.index + pred_val2, 2)
            likelihood2_h = likelihood2_h[(likelihood2_h.index >= 0) & (likelihood2_h.index <= 1)]
            if (pred_val3 is None) and (pred_val4 is None):
                posterior = likelihood1_h * likelihood2_h * prior_h / ((likelihood1_h * likelihood2_h * prior_h).sum())
    if pred_val3 is not None:
            if pred_val3==nan:
                print('Prediction columns begins with NaN')
                pred_val3 = None
            else:
                pred_val3 = round(round(pred_val3 * dx_inv) * dx, 2)
                likelihood3_h = likelihood3[h].copy() if ppd == 24 else likelihood3[qh].copy()
                likelihood3_h.index = [round(x+pred_val3,2) for x in likelihood3_h.index] # round(likelihood2_h.index + pred_val2, 2)
                likelihood3_h = likelihood3_h[(likelihood3_h.index >= 0) & (likelihood3_h.index <= 1)]
                if pred_val4 is None:
                    posterior = likelihood1_h * likelihood2_h * likelihood3_h * prior_h \
                        / ((likelihood1_h * likelihood2_h * likelihood3_h * prior_h).sum())            
    if pred_val4 is not None:
            if pred_val4==nan:
                print('Prediction columns begins with NaN')
                pred_val4 = None
            else:
                pred_val4 = round(round(pred_val4 * dx_inv) * dx, 2)
                likelihood4_h = likelihood4[h].copy() if ppd == 24 else likelihood4[qh].copy()
                likelihood4_h.index = [round(x+pred_val4,2) for x in likelihood4_h.index] # round(likelihood2_h.index + pred_val2, 2)
                likelihood4_h = likelihood4_h[(likelihood4_h.index >= 0) & (likelihood4_h.index <= 1)]
                posterior = likelihood1_h * likelihood2_h * likelihood3_h * likelihood4_h * prior_h \
                    / ((likelihood1_h * likelihood2_h * likelihood3_h * likelihood4_h * prior_h).sum())                                

    posterior_val = calc_mean_of_distribution(posterior)

    if plot_with_true_val is not None:
        if pred_val2 is None:
            plot_bayes(plot_with_true_val, prior_h, pred_val1, likelihood1_h, bayes_val=posterior_val, bayes=posterior)
        elif pred_val3 is None:
            plot_bayes(plot_with_true_val, prior_h, pred_val1, likelihood1_h, pred_val2, likelihood2_h, posterior_val, posterior)
        elif pred_val4 is None:
            plot_bayes_3(plot_with_true_val,
                         prior_h,
                         pred_val1, likelihood1_h, 
                         pred_val2, likelihood2_h,
                         pred_val3, likelihood3_h,
                         posterior_val,
                         posterior)
        else:
            plot_bayes_4(plot_with_true_val,
                         prior_h,
                         pred_val1, likelihood1_h, 
                         pred_val2, likelihood2_h,
                         pred_val3, likelihood3_h,
                         pred_val4, likelihood4_h,
                         posterior_val,
                         posterior)

    return posterior_val


def bayes(all_pred: pd.DataFrame,
               priors,
               prior_days,
               pred_col1,
               likelihood1,
               pred_col2=None,
               likelihood2=None,
               plot:dict=None,
               verbose=False,
               calc_errors=False):
    """ Perform bayes for every `timestamp_update` and `timestamp` in a forecast dataframe
    """
    deltaT = all_pred.timestamp.diff().value_counts().index[0].seconds / 3600  # hours
    ppd = int(24 / deltaT)

    pred_val2 = None

    df = all_pred.copy(deep=True)
    df_bayes = pd.DataFrame({'Bayes': [nan] * df.shape[0]}, index=range(df.shape[0]))
    if calc_errors: df_bayes['ErrorBayes'] = [nan] * df.shape[0]

    i = 0
    for t_now in df.timestamp_update.unique():
        df_now = df.loc[df.timestamp_update == t_now, :].dropna()
        df_now.set_index('timestamp', inplace=True)

        if len(df_now) == 0:
            print('No predictions for this t_now')
            break

        if verbose: print('\nt_now: ', t_now, '+ (i): ', end='')

        for t_predict, pred_val1 in zip(df_now.index, df_now[pred_col1]):

            if verbose: print(f'{(t_predict - t_now).seconds // 900} ', end=' ')
            
            load_val = df_now.loc[t_predict,'Load'] if calc_errors else None
            
            plot_with_true_val = None
            if plot and plot['timestamp_update'] == t_now and plot['timestamp'] == t_predict:
                plot_with_true_val = load_val

            if pred_col2 is not None: pred_val2 = df_now.loc[t_predict, pred_col2]

            bayes_val = calc_posterior(t=t_predict,
                                       priors=priors,
                                       prior_days=prior_days,
                                       ppd=ppd,
                                       pred_val1=pred_val1,
                                       likelihood1=likelihood1,
                                       pred_val2=pred_val2,
                                       likelihood2=likelihood2,
                                       plot_with_true_val=plot_with_true_val, )

            df_bayes.iloc[i, :] = [bayes_val, bayes_val-load_val] if calc_errors else [bayes_val]

            i += 1

    df_bayes.index = df.index
    df = pd.concat((df, df_bayes), axis=1)

    return df

def bayes_3(   all_pred: pd.DataFrame,
               all_pred2: pd.DataFrame=None,
               priors=None,
               prior_days=None,
               pred_col1=None,
               likelihood1=None,
               pred_col2=None,
               likelihood2=None,
               pred_col3=None,
               likelihood3=None,
               plot:dict=None,
               verbose=False,
               calc_errors=False):
    """ Perform bayes for every `timestamp_update` and `timestamp` in a forecast dataframe
    """
    deltaT = all_pred.timestamp.diff().value_counts().index[0].seconds / 3600  # hours
    ppd = int(24 / deltaT)

    pred_val2,pred_val3 = None,None

    df = all_pred.copy(deep=True)
    df2 = all_pred2.copy(deep=True) if all_pred2 is not None else None
    df_bayes = pd.DataFrame({'Bayes': [nan] * df.shape[0]}, index=range(df.shape[0]))
    if calc_errors: df_bayes['ErrorBayes'] = [nan] * df.shape[0]

    i = 0
    for t_now in df.timestamp_update.unique():
        df_now = df.loc[df.timestamp_update == t_now, :].dropna() 
        if df2 is not None:
            df_now2 = df2.loc[df2.timestamp_update == t_now, :].dropna()
            df_now2.set_index('timestamp', inplace=True)
        
        df_now.set_index('timestamp', inplace=True)

        if (len(df_now) == 0) or (df2 is not None and len(df_now2) == 0):
            print('No predictions for this t_now')
            break
        
        if verbose: print('\nt_now: ', t_now, '+ (i): ', end='')

        for t_predict, pred_val1 in zip(df_now.index, df_now[pred_col1]):

            if verbose: print(f'{(t_predict - t_now).seconds // 900} ', end=' ')
            
            load_val = df_now.loc[t_predict,'Load'] if calc_errors else None
            
            plot_with_true_val = None
            if plot and plot['timestamp_update'] == t_now and plot['timestamp'] == t_predict:
                plot_with_true_val = load_val

            if pred_col2 is not None: pred_val2 = df_now.loc[t_predict, pred_col2]
            
            if pred_col3 is not None: pred_val3 = df_now2.loc[t_predict, pred_col3]

            bayes_val = calc_posterior_3(t=t_predict,
                                       priors=priors,
                                       prior_days=prior_days,
                                       ppd=ppd,
                                       pred_val1=pred_val1,
                                       likelihood1=likelihood1,
                                       pred_val2=pred_val2,
                                       likelihood2=likelihood2,
                                       pred_val3=pred_val3,
                                       likelihood3=likelihood3,
                                       plot_with_true_val=plot_with_true_val, )

            df_bayes.iloc[i, :] = [bayes_val, bayes_val-load_val] if calc_errors else [bayes_val]

            i += 1

    df_bayes.index = df.index
    df = pd.concat((df, df_bayes), axis=1)

    return df

def bayes_4(   all_pred1: pd.DataFrame,
               all_pred2: pd.DataFrame=None,
               all_pred3: pd.DataFrame=None,
               all_pred4: pd.DataFrame=None,
               priors=None,
               prior_days=None,
               pred_col1=None,
               likelihood1=None,
               pred_col2=None,
               likelihood2=None,
               pred_col3=None,
               likelihood3=None,
               pred_col4=None,
               likelihood4=None,
               plot:dict=None,
               verbose=False,
               calc_errors=False):
    """ Perform bayes for every `timestamp_update` and `timestamp` in a forecast dataframe
    """
    deltaT = all_pred1.timestamp.diff().value_counts().index[0].seconds / 3600  # hours
    ppd = int(24 / deltaT)

    pred_val2,pred_val3,pred_val4 = None,None,None

    df1 = all_pred1.copy(deep=True)
    df2 = all_pred2.copy(deep=True) if all_pred2 is not None else None
    df3 = all_pred3.copy(deep=True) if all_pred3 is not None else None
    df4 = all_pred4.copy(deep=True) if all_pred4 is not None else None
    
    df_bayes = pd.DataFrame({'Bayes': [nan] * df1.shape[0]}, index=range(df1.shape[0]))
    if calc_errors: df_bayes['ErrorBayes'] = [nan] * df1.shape[0]

    i = 0
    for t_now in df1.timestamp_update.unique():
        df_now1 = df1.loc[df1.timestamp_update == t_now, :].dropna() 
        
        if df2 is not None:
            df_now2 = df2.loc[df2.timestamp_update == t_now, :].dropna()
            df_now2.set_index('timestamp', inplace=True)
            
        if df3 is not None:
            df_now3 = df3.loc[df3.timestamp_update == t_now, :].dropna()
            df_now3.set_index('timestamp', inplace=True)
            
        if df4 is not None:
            df_now4 = df4.loc[df4.timestamp_update == t_now, :].dropna()
            df_now4.set_index('timestamp', inplace=True)            
        
        df_now1.set_index('timestamp', inplace=True)

        if any([(len(df_now1) == 0),
                (df2 is not None and len(df_now2) == 0),
                (df3 is not None and len(df_now3) == 0),
                (df4 is not None and len(df_now4) == 0)]):
            print('No predictions for this t_now')
            break
        
        if verbose: print('\nt_now: ', t_now, '+ (i): ', end='')

        for t_predict, pred_val1 in zip(df_now1.index, df_now1[pred_col1]):

            if verbose: print(f'{(t_predict - t_now).seconds // 900} ', end=' ')
            
            load_val = df_now1.loc[t_predict,'Load'] if calc_errors else None
            
            
            if plot and plot['timestamp_update'] == t_now and plot['timestamp'] == t_predict:
                plot_with_true_val = load_val
            else:
                plot_with_true_val = None

            if pred_col2 is not None: pred_val2 = df_now2.loc[t_predict, pred_col2]
            
            if pred_col3 is not None: pred_val3 = df_now3.loc[t_predict, pred_col3]
            
            if pred_col4 is not None: pred_val4 = df_now4.loc[t_predict, pred_col4]

            bayes_val = calc_posterior_4(t=t_predict,
                                       priors=priors,
                                       prior_days=prior_days,
                                       pred_val1=pred_val1,
                                       likelihood1=likelihood1,
                                       pred_val2=pred_val2,
                                       likelihood2=likelihood2,
                                       pred_val3=pred_val3,
                                       likelihood3=likelihood3,
                                       pred_val4=pred_val4,
                                       likelihood4=likelihood4,
                                       plot_with_true_val=plot_with_true_val, )

            df_bayes.iloc[i, :] = [bayes_val, bayes_val-load_val] if calc_errors else [bayes_val]

            i += 1

    df_bayes.index = df1.index

    return pd.concat((df1, df_bayes), axis=1)

# def bayes_test(all_pred: pd.DataFrame,
#                priors,
#                prior_days,
#                pred_col1,
#                likelihood1,
#                pred_col2=None,
#                likelihood2=None,
#                plot=None,
#                verbose=False,
#                calc_errors=False):
#     """ Perform bayes for every `timestamp_update` and `timestamp` in a forecast dataframe
#     """
#     deltaT = all_pred.timestamp.diff().value_counts().index[0].seconds / 3600  # hours
#     ppd = int(24 / deltaT)

#     pred_val2 = None

#     df = all_pred.copy(deep=True)
    
#     df_bayes = pd.DataFrame({'Bayes': [nan] * df.shape[0]}, index=range(df.shape[0]))
#     if calc_errors:
#         df_bayes['ErrorBayes'] = [nan] * df.shape[0]

#     i = 0
#     for t_now in df.timestamp_update.unique():
#         df_now = df.loc[df.timestamp_update == t_now, :].dropna()
#         df_now.set_index('timestamp', inplace=True)

#         if len(df_now) == 0:
#             print('No predictions for this t_now')
#             break

#         if verbose: print('\nt_now: ', t_now, '+ (i): ', end='')

#         for t_predict, pred_val1 in zip(df_now.index, df_now[pred_col1]):
            
#             if calc_errors:
#                 load_val = df_now.loc[t_predict,'Load'] 

#             if verbose: print(f'{(t_predict - t_now).seconds // 900} ', end=' ')

#             if pred_col2 is not None:
#                 pred_val2 = df_now.loc[t_predict, pred_col2]

#             plot_with_true_val = None
#             if plot and plot['timestamp_update'] == t_now and plot['timestamp'] == t_predict:
#                 plot_with_true_val = load_val

#             bayes_val = calc_posterior(t=t_predict,
#                                        priors=priors,
#                                        prior_days=prior_days,
#                                        ppd=ppd,
#                                        pred_val1=pred_val1,
#                                        likelihood1=likelihood1,
#                                        pred_val2=pred_val2,
#                                        likelihood2=likelihood2,
#                                        plot_with_true_val=plot_with_true_val, )

#             if calc_errors:
#                 error_bayes = bayes_val - load_val
#                 df_bayes.iloc[i, :] = [bayes_val, error_bayes]
#             else:
#                 df_bayes.iloc[i, :] = [bayes_val]

#             i += 1

#     df_bayes.index = df.index
#     df = pd.concat((df, df_bayes), axis=1)

#     return df

def read_measurements_forecasts():
    
    meas,meas_max = read_historical_measurements(filepath=filepath,
                                                    load_col=load_col,
                                                    t_test_begin=t_test_begin,
                                                    resamp=resamp,
                                                    limit_days=limit_days,
                                                    prior_days=prior_days)
    
    pred1 = read_forecasts(  persist_filepath,
                            forecast_length_h,
                            meas_max,
                            col_t,
                            col_t_update,
                            col_pred,
                            col_pers,
                            col_load,
                            resamp)
    
    pred2 = None if forecast_filepath2 is None else read_forecasts( forecast_filepath2,
                                                                    forecast_length_h,
                                                                    meas_max,
                                                                    col_t,
                                                                    col_t_update,
                                                                    col_pred,
                                                                    col_pers,
                                                                    col_load,
                                                                    resamp)     
    
    pred3 = None if forecast_filepath3 is None else read_forecasts( forecast_filepath3,
                                                                    forecast_length_h,
                                                                    meas_max,
                                                                    col_t,
                                                                    col_t_update,
                                                                    col_pred,
                                                                    col_pers,
                                                                    col_load,
                                                                    resamp)    
    
    pred4 = None if forecast_filepath4 is None else read_forecasts( forecast_filepath4,
                                                                    forecast_length_h,
                                                                    meas_max,
                                                                    col_t,
                                                                    col_t_update,
                                                                    col_pred,
                                                                    col_pers,
                                                                    col_load,
                                                                    resamp)    
    
    pred_list = pred1, pred2, pred3, pred4
    
    return meas, meas_max, pred_list



def train(meas,pred_list,plot=False):
    
    pred_persist,pred2,pred3,pred4 = pred_list
    
    """
    priors
    """

    priors,meas_mat = build_priors( meas=meas,
                                    prior_days=prior_days,
                                    limit_days=limit_days,
                                    dx=dx,
                                    verbose=verbose)
    
    """
    likelihoods
    """
    plot_i = 28 if plot is not None else None
    err_dist_persist = create_distribution(     pred_persist,
                                                'ErrorPersist',
                                                dx=dx,
                                                plot_i=plot_i,)

    err_dist2 = None if pred2 is None else create_distribution( pred2, 
                                                                'ErrorPred',
                                                                dx=dx,
                                                                plot_i=plot_i,)
    
    err_dist3 = None if pred3 is None else create_distribution( pred3, 
                                                                'ErrorPred',
                                                                dx=dx,
                                                                plot_i=plot_i,)
    
    err_dist4 = None if pred4 is None else create_distribution( pred4, 
                                                                'ErrorPred',
                                                                dx=dx,
                                                                plot_i=plot_i,)
    
    err_dist_list = [err_dist_persist,err_dist2,err_dist3,err_dist4]

    return pred_list, err_dist_list, priors, meas_mat

def predict(pred_list,err_dist_list,priors,meas_max,limit=None,calc_errors=False):
    """
    all the bayes
    """
    pred1,pred2,pred3,pred4 = pred_list
    err_dist1,err_dist2,err_dist3,err_dist4 = err_dist_list
    
    times = pred1.timestamp_update.unique()[:None] # weird hack that seems to help reduce nans
    
    pred_bayes = bayes_4(   all_pred1=pred1.loc[pred1.timestamp_update.isin(times), :], 
                            all_pred2=None if pred2 is None else pred2.loc[pred2.timestamp_update.isin(times), :], 
                            all_pred3=None if pred3 is None else pred3.loc[pred3.timestamp_update.isin(times), :], 
                            all_pred4=None if pred4 is None else pred4.loc[pred4.timestamp_update.isin(times), :], 
                            priors=priors,
                            prior_days=prior_days,
                            pred_col1='Persist',
                            likelihood1=err_dist1,
                            pred_col2='Pred' if pred2 is not None else None,
                            likelihood2=err_dist2,
                            pred_col3='Pred' if pred3 is not None else None,
                            likelihood3=err_dist3,
                            pred_col4='Pred' if pred4 is not None else None,
                            likelihood4=err_dist4,
                            #plot={'timestamp_update':t, 'timestamp':t+td(hours=7)},
                            verbose=verbose,
                            calc_errors=calc_errors,)

    if pred_bayes.isna().sum().sum() > 0:
        if verbose:
            print(f'\n\n///// NaN values found: {pred_bayes.isna().sum().sum()}')
        pred_bayes = pred_bayes.ffill().bfill()

    if calc_errors:
        results_all = calc_results(pred_bayes)

        if verbose:
            print("\nResults:")
            for key, value in results_all.items():
                print(f"{key:<35} {value:>6.2f}")
    
    # check if any perfect persist error timesteps
    if verbose:
        assert pred_bayes.loc[pred_bayes.ErrorPersist.abs() == 0.0].shape[0] == 0
    
    #pred_bayes = pred_bayes.loc[pred_bayes.ErrorPersist.abs() > 0.0]
    
    # rescale to original magnitude
    numeric_cols = [x for x in pred_bayes.columns if x not in ['timestamp','timestamp_update']]
    pred_bayes[numeric_cols] = pred_bayes[numeric_cols] * meas_max
    
    return pred_bayes


if __name__ == '__main__': 
    
    """
    Read measurements and forecasts
    """
    
    meas, meas_max, pred_list = read_measurements_forecasts()
    
    """
    Train
    """
    
    pred_list, err_dist_list, priors,meas_mat = train(meas,pred_list)
    
    """
    Look at quantiles
    """
    
    #prior_qts = calc_quantiles(meas_mat, prior_days, plot=False)
    

    """
    Single Estimation
    """
    
    # [err_dist_pers,err_dist_lstm,err_dist_lstm2,err_dist_lstm3] = err_dist_list
    # t = pred.timestamp_update.unique()[0] + td(hours=7)
    # df_now = pred.loc[pred.timestamp_update==t,:].copy().dropna().set_index('timestamp')
    # df_now2 = pred2.loc[pred2.timestamp_update==t,:].copy().dropna().set_index('timestamp')
    # df_now3 = pred3.loc[pred3.timestamp_update==t,:].copy().dropna().set_index('timestamp')
    # posterior_val = calc_posterior_4( t=t,    
    #                                     priors=priors,
    #                                     prior_days=prior_days,
    #                                     pred_val1=df_now.loc[t, 'Persist'] ,
    #                                     likelihood1=err_dist_pers,
    #                                     pred_val2=df_now.loc[t, 'Pred'],
    #                                     likelihood2=err_dist_lstm,
    #                                     pred_val3=df_now2.loc[t, 'Pred'],
    #                                     likelihood3=err_dist_lstm2,
    #                                     #plot_with_true_val=df_now.loc[t, 'Load'],
    #                                     )    


    """
    Predict
    """
    
    # predict starting from t_test_begin
    pred_list = [pred[pred.timestamp_update>=t_test_begin] for pred in pred_list]
    
    pred_bayes = predict(   pred_list,
                            err_dist_list,
                            priors,
                            meas_max,
                            limit=None,
                            calc_errors=calc_errors)
    
    