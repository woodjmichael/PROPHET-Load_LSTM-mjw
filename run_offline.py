import os
from math import nan
from datetime import datetime as dt
from datetime import timedelta as td
from random import shuffle
import pandas as pd
from PROPHET_LOAD_LSTM.main import lstm_forecaster_training, lstm_forecaster
from PROPHET_LOAD_LSTM.util import util

#ini_path = "/home/mjw/Code/PROPHET-Load_LSTM-mjw/lstm_forecaster.ini"
ini_path = r"C:\Users\Admin\Code\PROPHET-Load_LSTM\lstm_forecaster.ini"

config = util.read_config(ini_path)
out_dir = config.get("data_opt", "out_dir")

#
# Train
#

lstm_forecaster_training.main([ini_path])
 
#
# Predict
#

"""mae_persist, mae_lstm, skill, forecast = lstm_forecaster.main([ini_path],dt(2019,12,13,0,0),plots=True)"""


#
# Test
#

forecasts = pd.DataFrame()

for t in pd.date_range('2019-9-13 0:00','2019-12-28 0:00',freq='1h'):
    print('timestamp_update:',t)
    
    _, _, _, new_forecast = lstm_forecaster.main([ini_path],t,plots=False)
    
    new_forecast.index = new_forecast.index.tz_convert(None)
    new_forecast.reset_index(inplace=True)
    new_forecast.rename(columns={'index':'timestamp_forecast'},inplace=True)
    new_forecast.insert(0,'timestamp_forecast_update',t)

    forecasts = pd.concat([forecasts,new_forecast],axis=0,ignore_index=True)    

forecasts.to_csv(out_dir+'test_forecasts.csv')

#
# Hyper parameter search
#
"""
# check for existing hp search
already_searched_space = []
if os.path.exists(out_dir + 'hp_search.csv'):
    hp = pd.read_csv(out_dir + 'hp_search.csv',index_col=0)
    for units,n_back,n_dense,dropout in hp[['units','n_back','n_dense','dropout']].values:
        already_searched_space.append((int(units),int(n_back),int(n_dense),dropout))

# create search space
search_space = []
for units in [24*x for x in [1,2,3,4,5,6,8,10,12,14,16,18,20]]:
    for n_back in [24*x for x in [1,2,3,4,5,6,8,10,12,14,16,18,20]]:
        for n_dense in [24,36]: #[x for x in [12,24,48,96,144]]:
            for dropout in [0,0.1]:
                if (units,n_back,n_dense,dropout) not in already_searched_space:
                    search_space.append((units,n_back,n_dense,dropout))         
shuffle(search_space)

# for each model
results = pd.DataFrame(columns=['units','n_back','n_dense','dropout','vloss','test_mae_pers','test_mae_pred','test_skill'])
for units,n_back,n_dense,dropout in search_space:
    
    # train
    print(f'\n\n\n ///// units={units} n_back={n_back} n_dense={n_dense} dropout={dropout}/////\n')
    try:
        vloss = lstm_forecaster_training.main([ini_path],units,n_back,n_dense,dropout)
    except:
        print('//////////////////// FAIL /////////////////////')
        vloss=nan
        
    # test
    forecasts = pd.DataFrame()
    for t in pd.date_range('2019-9-13 0:00','2019-12-28 0:00',freq='4h'):
        print('timestamp_update:',t)
        _, _, _, new_forecast = lstm_forecaster.main([ini_path],t,units=units,n_back=n_back,n_dense=n_dense,dropout=dropout)
        new_forecast.index = new_forecast.index.tz_convert(None)
        new_forecast.reset_index(inplace=True)
        new_forecast.rename(columns={'index':'timestamp_forecast'},inplace=True)
        new_forecast.insert(0,'timestamp_forecast_update',t)
        forecasts = pd.concat([forecasts,new_forecast],axis=0,ignore_index=True)    
        
    test_mae_pers = forecasts.loc[(forecasts.persist-forecasts.power)!=0,['persist','power']].diff(axis=1).power.abs().mean()
    test_mae_pred = forecasts.loc[(forecasts.persist-forecasts.power)!=0,['predicted_activepower_ev_1','power']].diff(axis=1).power.abs().mean()        
    test_skill = 1 - test_mae_pred / test_mae_pers
        
    # output
    results.loc[len(results)] = {'units':units,'n_back':n_back,'n_dense':n_dense,'dropout':dropout,'vloss':vloss,'test_mae_pers':test_mae_pers,'test_mae_pred':test_mae_pred,'test_skill':test_skill}
    results.to_csv(out_dir+'hp_search.csv')"""
