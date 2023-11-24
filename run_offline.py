import os
from datetime import datetime,timedelta
import pandas as pd
from PROPHET_LOAD_LSTM.main import lstm_forecaster_training, lstm_forecaster

def multi_test(t = datetime(2023,10,1,0,15,0)):
    r = pd.DataFrame({'t':[],'rmse persist':[],'rmse lstm':[],'skill':[]})
    results_path = os.path.dirname(ini_path)+'/data/test_results.csv'
    while t < datetime(2023,10,31,0,0,0):
        rmse_persist, rmse_lstm, skill = lstm_forecaster.main([ini_path],pd.to_datetime(t),plots=False)
        r.loc[len(r)] = [t,rmse_persist,rmse_lstm,skill]
        r.to_csv(results_path)
        t += timedelta(hours=1)

ini_path = "/home/mjw/Code/PROPHET-Load_LSTM/lstm_forecaster.ini"
#ini_path = "C:/Users/Admin/Code/PROPHET-Load_LSTM/lstm_forecaster.ini"

lstm_forecaster_training.main([ini_path])
#lstm_forecaster_training.grid_search([ini_path])

#lstm_forecaster.main([ini_path],datetime(2019,12,17,0,1,0))

#multi_test()

