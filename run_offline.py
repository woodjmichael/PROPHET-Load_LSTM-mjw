from datetime import datetime,timedelta
import pandas as pd
from PROPHET_LOAD_LSTM.main import lstm_forecaster_training, lstm_forecaster


#ini_path = "/home/mjw/Code/PROPHET-Load_LS 
# 
# 
# TM-231011/lstm_forecaster.ini"
ini_path = "C:/Users/Admin/Code/PROPHET-Load_LSTM/lstm_forecaster.ini"

#lstm_forecaster_training.main([ini_path])
#lstm_forecaster_training.grid_search([ini_path])
#lstm_forecaster.main([ini_path])

t = datetime(2023,9,1,0,1,0)
r = pd.DataFrame({'t':[],'rmse persist':[],'rmse lstm':[],'skill':[]})
while t < datetime(2023,12,29,0,0,0):
    rmse_persist, rmse_lstm, skill = lstm_forecaster.main([ini_path],pd.to_datetime(t),plots=False)
    r.loc[len(r)] = [t,rmse_persist,rmse_lstm,skill]
    r.to_csv('C:/Users/Admin/Code/PROPHET-Load_LSTM/data/results.csv')
    t += timedelta(hours=1)