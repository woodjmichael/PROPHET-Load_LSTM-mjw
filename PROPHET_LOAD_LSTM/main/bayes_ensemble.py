# Ideas
# - Need a better way to not calc skill when persist MAE = 0
# - Calculate prior in time-deterministic manor
#   - Maybe calculate all `meas_mat` but only *use* rows from past to make `priors` for each `t`
# - Forecast weekends
# 
# Online workflow
# - "Train" bayes on all past meas
#   - Prior: from all past meas
#   - Forecast errors: from all past meas
# - "Test" bayes with new online forecast values
#   - Get new forecast point
#   - Shift likelihood
#   - Calculate posterior and get bayes estimate
# 
# To do
# - Dont forget to uncomment line in `create_distribution()` to divide `n_bins` by 2    

PLOTTING = False

from numpy.random import randint
import pandas as pd
from pandas import Timestamp as ts
from pandas import Timedelta as td
from math import nan

if PLOTTING:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import plotly.graph_objects as go
    import plotly.io as pio
    from tabulate import tabulate


def dxrange(start,stop,step):
    return [round(x*step,3) for x in range(int(start/step), int(stop/step))]


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


def create_distribution(pred, col_name, dx, ppd=24, plot_i=None):    
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

    ppd = {'1h': 24, '15min': 96}[resamp]
    ppw = ppd * len(limit_days)

    meas = meas[meas.index.weekday.isin(limit_days)]

    if plot: plot_weekly(meas, limit_days=len(limit_days))

    meas = meas[:t_test_begin][:-1]  # only make priors on training set values

    return meas, meas_max, ppw, ppd


# ## Build from measurements

def build_priors(meas, ppd, prior_days, limit_days, dx, verbose=False):
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

def bag_priors(meas_mat, dx, prior_days, ppd):
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

def calc_quantiles(meas_mat, prior_days, ppd, plot=False):
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

    if PLOTTING:
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
    # filepath = r'models/jpl_ev/bayes-lstm-persist/u24-96_d0.1_n48_f4'
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
    if PLOTTING:
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


def estimate(pred_new,verbose=False):
    
    #
    # Impianto 4 (PROPHET-Load_LSTM)
    #
    
    #project_path = r'/home/mjw/Code/PROPHET-Load_LSTM-mjw/'
    #project_path = r'/home/mjw/Code/load-forecast/'
    #project_path = r'C:/Users/Admin/Code/load-forecast/'
    project_path = r'C:/Users/Admin/Code/PROPHET-Load_LSTM/'

    limit_days = list(range(7))
    prior_days = 7
    dx = 0.02
    resamp='15min'   
        
    filepath = project_path + 'data/Impianto_4_clean.csv'

    load_col = 'Potenza'
    t_test_begin = pd.Timestamp('2019-9-13 0:00')

    forecast_filepath = project_path + r'data/output/impianto_4b/test_forecasts.csv'

    col_t = 'timestamp_forecast'
    col_t_update = 'timestamp_forecast_update'
    col_pred = 'predicted_activepower_ev_1'
    col_pers = 'persist'
    col_load = 'power'
    forecast_length_h = 36

    col_t_new = col_t
    col_t_update_new = col_t_update
    col_pred_new = col_pred
    col_pers_new = col_pers    

    meas, meas_max, _, ppd = read_historical_measurements(  filepath=filepath,
                                                            load_col=load_col,
                                                            t_test_begin=t_test_begin,
                                                            resamp=resamp,
                                                            limit_days=limit_days,
                                                            prior_days=prior_days)    

    priors, _ = build_priors(   meas=meas,
                                ppd=ppd,
                                prior_days=prior_days,
                                limit_days=limit_days,
                                dx=dx,
                                verbose=verbose)  

    pred = read_forecasts(forecast_filepath,
                          forecast_length_h,
                          meas_max,
                          col_t,
                          col_t_update,
                          col_pred,
                          col_pers,
                          col_load,
                          resamp)

    err_dist_pers = create_distribution(    pred,
                                            'ErrorPersist',
                                            ppd=ppd,
                                            dx=dx, )

    err_dist_lstm = create_distribution(    pred,
                                            'ErrorPred',
                                            ppd=ppd,
                                            dx=dx, )
    
    pred_new = pred_new.rename( columns={col_t_new: 'timestamp',
                                col_t_update_new: 'timestamp_update',
                                col_pred_new: 'Pred',
                                col_pers_new: 'Persist',})
    
    times = pred_new.timestamp_update.unique() # weird hack that seems to help reduce nans
    pred_bayes = bayes( pred_new.loc[pred_new.timestamp_update.isin(times), :], 
                        priors,
                        prior_days,
                        pred_col1='Pred',
                        likelihood1=err_dist_lstm,
                        pred_col2='Persist',
                        likelihood2=err_dist_pers,
                        verbose=verbose,)

    if pred_bayes.isna().sum().sum() > 0:
        print(f'\n\n///// NaN values found: {pred_bayes.isna().sum().sum()}')
        pred_bayes = pred_bayes.ffill().bfill()    
    
    pred_bayes = pred_bayes.rename( columns={'timestamp':col_t_new,
                                    'timestamp_update':col_t_update_new,
                                    'Pred':col_pred_new,
                                    'Persist':col_pers_new,})

    if pred_bayes.isna().sum().sum() > 0:
        pred_bayes = pred_bayes.ffill().bfill()  
    
    return pred_bayes


if __name__ == '__main__':
    """#project_path = r'/home/mjw/Code/PROPHET-Load_LSTM-mjw/'
    #project_path = r'/home/mjw/Code/load-forecast/'
    project_path = r'C:/Users/Admin/Code/load-forecast/'

    limit_days = list(range(7))
    prior_days = 7
    dx = 0.02
    resamp='15min'

    #
    # Impianto 4
    #   
        
    filepath = project_path + 'data/Impianto_4_clean.csv'

    load_col = 'Potenza'
    t_test_begin = pd.Timestamp('2019-9-13 0:00')

    forecast_filepath = project_path + r'models/impianto_4/PROPHET-Load_LSTM/u432_ud24_nb288_d0/test_forecasts.csv'

    col_t = 'timestamp_forecast'
    col_t_update = 'timestamp_forecast_update'
    col_pred = 'predicted_activepower_ev_1'
    col_pers = 'persist'
    col_load = 'power'
    forecast_length_h = 36

    col_t_new = col_t
    col_t_update_new = col_t_update
    col_pred_new = col_pred
    col_pers_new = col_pers"""
        
    #
    # JPL
    #
    
    """
    t_test_begin = pd.Timestamp('2019-12-26 0:00')

    filepath = project_path + r'data/all_JPL_v5.csv'
    load_col = 'power'

    forecast_filepath = project_path + r'models/jpl_ev/bayes-lstm-persist/u24-96_d0.1_n48_f4/test_forecasts_fill-persist.csv'
    forecast_length_h = 36

    col_t = 'timestamp'
    col_t_update = 'timestamp_update'
    col_pred = 'Pred'
    col_pers = 'Persist'
    col_load = 'Load'

    col_t_new = 'timestamp'
    col_t_update_new = 'timestamp_update'
    col_pred_new = 'Pred'
    col_pers_new = 'Persist'

    # col_t_new = 'timestamp_forecast'
    # col_t_update_new = 'timestamp_forecast_update'
    # col_pred_new = 'predicted_activepower_ev_1'
    # col_pers_new = 'persist'    
    """
    
    meas, meas_max, ppw, ppd = read_historical_measurements(filepath=filepath,
                                                            load_col=load_col,
                                                            t_test_begin=t_test_begin,
                                                            resamp=resamp,
                                                            limit_days=limit_days,
                                                            prior_days=prior_days)

    priors, meas_mat = build_priors(meas=meas,
                                    ppd=ppd,
                                    prior_days=prior_days,
                                    limit_days=limit_days,
                                    dx=dx,
                                    verbose=True)

    prior_qts = calc_quantiles(meas_mat, prior_days, ppd, plot=False)

    pred = read_forecasts(forecast_filepath,
                          forecast_length_h,
                          meas_max,
                          col_t,
                          col_t_update,
                          col_pred,
                          col_pers,
                          col_load,
                          resamp)

    err_dist_pers = create_distribution(pred,
                                           'ErrorPersist',
                                           ppd=ppd,
                                           dx=dx, )

    err_dist_lstm = create_distribution(pred,
                                           'ErrorPred',
                                           ppd=ppd,
                                           dx=dx, )

    # one bayes estimation
    t = pred.timestamp_update.iloc[0] + td(hours=7)
    df_now = pred.loc[pred.timestamp_update == t, :].copy().dropna().set_index('timestamp')
    posterior_val = calc_posterior(t=t,
                                   priors=priors,
                                   prior_days=prior_days,
                                   ppd=ppd,
                                   pred_val1=df_now.loc[t, 'Pred'],
                                   likelihood1=err_dist_lstm,
                                   # pred_val2=df_now.loc[t, 'Persist'],
                                   # likelihood2=err_dist_pers,
                                   # plot_with_true_val=df_now.loc[t, 'Load'],
                                   )

    # bayes estimations with multiple forecasters
    t = pred.timestamp_update.iloc[0]
    bayes(  pred.loc[pred.timestamp_update == t, :],
            priors,
            prior_days,
            pred_col1='Pred',
            likelihood1=err_dist_lstm,
            pred_col2='Persist',
            likelihood2=err_dist_pers,
            # plot={'timestamp_update':t, 'timestamp':t+td(hours=7)},
            verbose=True,
            calc_errors=True,)
    
    # all the bayes
    times = pred.timestamp_update.unique() # weird hack that seems to help reduce nans
    pred_bayes = bayes( pred.loc[pred.timestamp_update.isin(times), :], 
                        priors,
                        prior_days,
                        pred_col1='Pred',
                        likelihood1=err_dist_lstm,
                        pred_col2='Persist',
                        likelihood2=err_dist_pers,
                        #plot={'timestamp_update':t, 'timestamp':t+td(hours=7)},
                        verbose=True,
                        calc_errors=True,)

    if pred_bayes.isna().sum().sum() > 0:
        print(f'\n\n///// NaN values found: {pred_bayes.isna().sum().sum()}')
        pred_bayes = pred_bayes.ffill().bfill()

    pred_bayes = pred_bayes.copy(deep=True)
    pred_bayes = pred_bayes.loc[pred_bayes.ErrorPersist.abs() > 0.0]

    results_all = calc_results(pred_bayes)

    print("\nResults:")
    for key, value in results_all.items():
        print(f"{key:<35} {value:>6.2f}")