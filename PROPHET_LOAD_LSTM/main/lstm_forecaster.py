MICROGRID_PC = False
SQL = False
BAYES = False

import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pickle import load
from tensorflow import keras
import tensorflow as tf

from PROPHET_LOAD_LSTM.util import util
from PROPHET_LOAD_LSTM.util.data_util import prepare_data_test
from PROPHET_LOAD_LSTM.util.model_util import attention, Custom_Loss_Prices

if BAYES:
    from PROPHET_LOAD_LSTM.main import bayes_ensemble as bayes
    print('Pre-training Bayes method, this will take 10-60 seconds..')
    meas, meas_max, pred_list = bayes.read_measurements_forecasts()
    pred_list, err_dist_list, priors, _ = bayes.train(meas,pred_list)
    print('.. done pre-training bayes')  


if MICROGRID_PC:
    from PROPHET_DB import mysql
if not MICROGRID_PC:
    import matplotlib.pyplot as plt
    pd.options.plotting.backend='matplotlib'
 

WEEKDAY_LOOKUP={'0':'Mon','1':'Tue','2':'Wed','3':'Thu','4':'Fri','5':'Sat','6':'Sun'}
   

# import daiquiri
# import sklearn.preprocessing

# from PROPHET_DB.setup_logger import setup_logger

# Logger definition
# LOGGER = daiquiri.getLogger(__name__)


def main(argv=None,t_now=None,plots=False,saveplots=False,units=None,n_back=None,n_dense=None,dropout=None,df=None,model=None):
    # read configuration file
    args = util.parse_arguments(argv)
    config = util.read_config(args.config)
    data_opt, model_opt = util.read_options(config=config)
    
    model_opt['LSTM_num_hidden_units'] = model_opt['LSTM_num_hidden_units'] if units is None else units
    data_opt['n_back'] = data_opt['n_back'] if n_back is None else n_back
    model_opt['Dense_input_dim'] = model_opt['Dense_input_dim'] if n_dense is None else n_dense
    model_opt['Dropout_rate'] = model_opt['Dropout_rate'] if dropout is None else dropout

    # logger_config = {'mailhost': (config["logger"]["mailhost"], config.getint("logger", "port")),
    #                  'fromaddr': config["logger"]["fromaddr"],
    #                  'toaddr': config["logger"]["toaddrs"],
    #                  'subject': config["logger"]["subject"],
    #                  'credentials': (config["logger"]["username"], config["logger"]["password"]),
    #                  'mail_handler': config.getboolean("logger", "mail_handler"),
    #                  'backup_count': config.getint("logger", "backup_count")}

    if df is None:
        if MICROGRID_PC and SQL:
            talktoSQL = mysql.MySQLConnector(database=config["mysql"]["database"],
                                            host=config["mysql"]["host"],
                                            port=config["mysql"]["port"],
                                            user=config["mysql"]["user"],
                                            password=config["mysql"]["password"])

            days_back = 10
            # ev_query = util.create_query(talktoSQL, config["sql_table"]["ev_power_table"], config["sql_table"]["time_column"])
            ev_query_test = util.create_query_test(talktoSQL, config["sql_table"]["ev_power_table"],
                                                config["sql_table"]["ev_power_field"],
                                                config["sql_table"]["time_column"], days_back)
            df = talktoSQL.read_query(ev_query_test, {config["sql_table"]["time_column"]})
            df[config["sql_table"]["time_column"]] = pd.to_datetime(df[config["sql_table"]["time_column"]],
                                                                    format='%Y-%m-%d %H:%M:%S%z')  # .dt.tz_convert('America/Los_Angeles')
            df.rename(columns={config["sql_table"]["ev_power_field"]: 'power'}, inplace=True)
            df.set_index(config["sql_table"]["time_column"], inplace=True)
        
        else:
            df = pd.read_csv(data_opt['data_path'],
                            index_col=0,
                            parse_dates=True,
                            comment='#')
            df.rename(columns={config["data_opt"]["out_col"]: 'power'}, inplace=True)
            df.index = df.index.tz_localize('UTC')

        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.dayofweek
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df = df[data_opt['columns']]

    #
    # lstm forecast
    #

    ## definizione tempi inizio
    #ToDo: attenzione, i dati sono diversi dall'ev in termini di freq e risoluzione temporale

    if t_now is None:
        t_now = datetime.utcnow()
    else:
        t_now += timedelta(seconds=1) # failure if h,m,s are all 0

    inizio_ts = pd.to_datetime(t_now).tz_localize('UTC')-timedelta(hours=(data_opt['n_back']/4+0.25))  # 7 per aggiustare ora LA, modificare
    fine_ts = pd.to_datetime(t_now).tz_localize('UTC')

    mask_ts = np.logical_and(df.index > inizio_ts, df.index < fine_ts)
    test = df.loc[mask_ts].copy()

    scaler_X = load(open(model_opt["model_path"] / Path("scaler_in.pkl"), 'rb'))
    scaler_y =  load(open(model_opt["model_path"] / Path("scaler_out.pkl"), 'rb'))

    test_X, test_y= prepare_data_test(test, scaler_X, data_opt)
    #test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))
    if model is None:
        model = keras.models.load_model(model_opt["model_path"] / Path('model.h5'),
                                        custom_objects={"attention": attention,
                                                        'Custom_Loss_Prices':Custom_Loss_Prices,
                                                        })
    y_hat_sc = model.predict(test_X)

    y_hat_sc = y_hat_sc[:, :, 0]#.reshape((test_X.shape[0], test_X.shape[1]))
    y_hat = scaler_y.inverse_transform(y_hat_sc)[-1,:]

    y_hat[np.isnan(y_hat)] = 0
    y_hat[y_hat < 0] = 0

    idx_y = pd.date_range(start=test_y.index[-1],
                                                    periods=data_opt['n_timesteps'],
                                                    freq=str(data_opt['freq'])+'min')
    forecast_dict = {'timestamp_utc': idx_y,
                     'timestamp_forecast_update': t_now,
                     'predicted_activepower_ev_1': y_hat,
                     }

    forecast_df = pd.DataFrame.from_dict(forecast_dict)

    #
    # persistence forecast
    #

    t_begin = pd.to_datetime(t_now).floor(str(data_opt['freq'])+'min').tz_localize('UTC') - timedelta(days=7)
    idx_x = pd.date_range(t_begin,
                        periods=int(config['data_opt']['n_timesteps']),
                        freq=str(data_opt['freq'])+'min')

    
    #y_hat = df['power'].loc[idx_x].values.flatten()
    y_hat = df.loc[idx_x,data_opt['target_col'][0]].values.flatten()
    #idx_y = [t+pd.Timedelta(days=7) for t in idx_x]

    forecast_df['persist'] = y_hat
    
    #
    # bayes ensemble/filter
    #

    if BAYES:
        forecast_df['timestamp_forecast_update'] = forecast_df['timestamp_forecast_update'].dt.floor('min')
        forecast_df['timestamp_utc'] = forecast_df.timestamp_utc.dt.tz_localize(None)
        forecast_df.rename(columns={'timestamp_utc':'timestamp',
                                    'timestamp_forecast_update':'timestamp_update',
                                    'predicted_activepower_ev_1':'Pred',
                                    'persist':'Persist'}, inplace=True)
        
        pred_list = [forecast_df,forecast_df,None,None]
        
        pred_bayes = bayes.predict(pred_list,err_dist_list,priors,meas_max,limit=None)
        
        pred_bayes.rename(columns={'timestamp':'timestamp_utc',
                                    'timestamp_update':'timestamp_forecast_update',
                                    'Bayes':'predicted_activepower_ev_1',
                                    'Persist':'persist'}, inplace=True)
        
        pred_bayes.timestamp_utc = pred_bayes.timestamp_utc.dt.tz_localize('UTC')
        pred_bayes.timestamp_forecast_update = pred_bayes.timestamp_forecast_update + timedelta(minutes=1)
        
        forecast_df = pred_bayes.copy()

    #
    # output
    #

    if MICROGRID_PC and SQL:
        talktoSQL.write(forecast_df, config['sql_table']['ev_forecast_table'])
        #talktoSQL.write(forecast_df, 'db_lstm_prediction_test')
    else:
        forecast_df.index = forecast_df['timestamp_utc']
        forecast_df.drop(columns=['timestamp_utc'], inplace=True)
        #forecast_df.to_csv(data_opt['out_dir']/Path('forecast_df.csv'))
    
        t_begin = pd.to_datetime(t_now).floor(str(data_opt['freq'])+'min').tz_localize('UTC')# - timedelta(days=7)
        #t_end = pd.to_datetime(t_now).floor(str(data_opt['freq'])+'min').tz_localize('UTC') + timedelta(days=3)
        idx = pd.date_range(t_begin,periods=data_opt['n_timesteps'],freq=str(data_opt['freq'])+'min')
        
        #df = df.loc[idx,'power'].to_frame()
        df = df.loc[idx,data_opt['target_col']]
        
        # mae_persist = (df.power - forecast_df.persist).dropna().abs().mean()
        # mae_lstm =    (df.power - forecast_df.predicted_activepower_ev_1).dropna().abs().mean()
        # skill = 1 - mae_lstm/mae_persist
        
        forecast_df = pd.concat((forecast_df.drop(columns=['timestamp_forecast_update']),df),axis=1)
        
        # if plots:
        #     weekday = WEEKDAY_LOOKUP[str(forecast_df.index[0].weekday())]
        #     title = f'Skill={100*skill:.0f}%, Weekday={weekday}'
        #     forecast_df.plot(title=title)
        #     if saveplots:
        #         ymd = f'{t_now.year}-{t_now.month}-{t_now.day}'
        #         plt.savefig(f'img/{ymd}.png')
        #     else:
        #         plt.show()
            
        # forecast_df.timestamp_forecast_update = t_begin.tz_convert(None)

        # return mae_persist, mae_lstm, skill, forecast_df
    
        return None,None,None,forecast_df


if __name__ == "__main__":
    main()