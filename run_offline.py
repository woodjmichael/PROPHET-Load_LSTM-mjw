import os
from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd
from PROPHET_LOAD_LSTM.main import lstm_forecaster_training, lstm_forecaster

# def multi_test(t = datetime(2020,1,1,0,0,0),t_final=datetime(2020,1,1,1,0,0)):
#     r = pd.DataFrame({'t':[],'rmse persist':[],'rmse lstm':[],'skill':[]})
#     results_path = os.path.dirname(ini_path)+'/data/test_results.csv'
#     forecast_df = pd.DataFrame({'timestamp_forecast_update':[],'predicted_activepower_ev_1':[],'persist':[]})
#     while t < t_final:
#         rmse_persist, rmse_lstm, skill, _forecast_df = lstm_forecaster.main([ini_path],pd.to_datetime(t),plots=False)
#         r.loc[len(r)] = [t,rmse_persist,rmse_lstm,skill]
#         forecast_df = pd.concat([forecast_df,_forecast_df],axis=0)
#         r.to_csv(results_path)
#         t += timedelta(hours=1)
#     forecast_df.to_csv(ini_path[:-19]+'/data/output/all_forecast_df.csv')

#ini_path = "/home/mjw/Code/PROPHET-Load_LSTM/lstm_forecaster.ini"
ini_path = "C:/Users/Admin/Code/PROPHET-Load_LSTM/lstm_forecaster.ini"

#lstm_forecaster_training.main([ini_path])
#lstm_forecaster_training.grid_search([ini_path])


#mae_persist, mae_lstm, skill, forecast = lstm_forecaster.main([ini_path],dt(2020,1,1,0,0),plots=False)

mae = pd.DataFrame({'t':[],'persist':[],'lstm':[],'skill':[]})
forecasts = pd.DataFrame({'timestamp_forecast_update':[],'predicted_activepower_ev_1':[],'persist':[]})
for t in pd.date_range(dt(2019,12,5,0,0,0),dt(2020,1,15,0,0),freq='15min'):
    t=pd.to_datetime(t)
    #_, _, _, forecast = lstm_forecaster.main([ini_path],t,plots=False)
    mae_persist, mae_lstm, skill, forecast = lstm_forecaster.main([ini_path],t,plots=False)
    forecast.index = forecast.index.tz_convert(None)
    forecasts = pd.concat([forecasts,forecast],axis=0)
    forecasts.to_csv(ini_path[:-19]+'data/output/forecasts.csv')
    mae.loc[len(mae)] = [t,mae_persist,mae_lstm,skill]
    mae.to_csv(ini_path[:-19]+'data/output/errors.csv')
    pass