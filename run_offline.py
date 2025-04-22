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

"""lstm_forecaster_training.main([ini_path])"""

#
# Predict
#

#mae_persist, mae_lstm, skill, forecast = lstm_forecaster.main([ini_path],dt(2019,12,13,0,0),plots=True)

#
# Hyper parameter search
#
search_space = []
results = pd.DataFrame({'units':[],'n_back':[],'n_dense':[],'dropout':[],'vloss':[]})
for units in [24*x for x in [1,2,3,4,5,8,10,12,14,16,18,20]]:
    for n_back in [24*x for x in [1,2,3,4,5,8,10,12,14,16,18,20]]:
        for n_dense in [24]:
            for dropout in [0,0.1,0.2]:
                search_space.append((units,n_back,n_dense,dropout))           
shuffle(search_space)
for u,nb,nd,d in search_space:
    print(f'\n\n\n ///// units={u} n_back={nb} n_dense={nd} dropout={d}/////\n')
    try:
        vloss = lstm_forecaster_training.main([ini_path],u,nb,nd,d)
    except:
        print('//////////////////// FAIL /////////////////////')
        vloss=pd.NA
    results.loc[len(results)] = [u,nb,nd,d,vloss]
    results.to_csv(out_dir+'hp_search.csv')
    
#
# Test
#

"""forecasts = pd.DataFrame()

for t in pd.date_range('2019-9-13 0:00','2019-12-28 0:00',freq='1h'):
    print('timestamp_update:',t)
    
    _, _, _, new_forecast = lstm_forecaster.main([ini_path],t,plots=False)
    
    new_forecast.index = new_forecast.index.tz_convert(None)
    new_forecast.reset_index(inplace=True)
    new_forecast.rename(columns={'index':'timestamp_forecast'},inplace=True)
    new_forecast.insert(0,'timestamp_forecast_update',t)

    forecasts = pd.concat([forecasts,new_forecast],axis=0,ignore_index=True)    

forecasts.to_csv(out_dir+'test_forecasts.csv')"""