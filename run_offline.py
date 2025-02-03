from datetime import datetime as dt
from datetime import timedelta as td
from random import shuffle
import pandas as pd
from PROPHET_LOAD_LSTM.main import lstm_forecaster_training, lstm_forecaster

ini_path = "/home/mjw/Code/PROPHET-Load_LSTM-mjw/lstm_forecaster.ini"
#ini_path = r"C:/Users/woodj/OneDrive - Politecnico di Milano/Code/PROPHET-Load_LSTM/lstm_forecaster.ini"

#
# Train
#

lstm_forecaster_training.main([ini_path])

#
# Predict
#

#mae_persist, mae_lstm, skill, forecast = lstm_forecaster.main([ini_path],dt(2019,12,13,0,0),plots=True)

#
# Random search
#
"""search_space = []
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
        vloss = lstm_forecaster_training.main([ini_path],u,nb,nd,d)M
    except:
        print('//////////////////// FAIL /////////////////////')
        vloss=pd.NA
    results.loc[len(results)] = [u,nb,nd,d,vloss]
    results.to_csv(ini_path[:-19]+'data/output/random_search.csv')"""
    
#
# Test DAILY UPDATE
#
mae = pd.DataFrame({'t':[],'persist':[],'lstm':[],'skill':[]})
forecasts = pd.DataFrame({'timestamp_forecast_update':[],
                          'predicted_activepower_ev_1':[],
                          'persist':[]})
#for t in pd.date_range(dt(2018,10,15),dt(2019,12,1),freq='24h'):
for t in pd.date_range(dt(2019,12,13),dt(2020,1,16),freq='24h'):
    t=pd.to_datetime(t)
    #_, _, _, forecast = lstm_forecaster.main([ini_path],t,plots=False)
    mae_persist, mae_lstm, skill, forecast = lstm_forecaster.main([ini_path],t,plots=False)
    forecast.index = forecast.index.tz_convert(None)
    
    forecast = forecast[:96] # only day-ahead
    
    forecasts = pd.concat([forecasts,forecast],axis=0)
    forecasts.to_csv(ini_path[:-19]+'forecasts.csv')
    mae.loc[len(mae)] = [t,mae_persist,mae_lstm,skill]
    mae.to_csv(ini_path[:-19]+'errors.csv')
    
#forecast = forecast[['predicted_activepower_ev1','persist']]
#forecast.colummns = ['LSTM kW','Persist kW']
#forecasts.to_csv(ini_path[:-19]+'forecasts.csv')

print('Total skill',1-(mae.lstm.mean()/mae.persist.mean()))

