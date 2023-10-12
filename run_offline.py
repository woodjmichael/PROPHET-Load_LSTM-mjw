from PROPHET_LOAD_LSTM.main import lstm_forecaster_training, lstm_forecaster

ini_path = "/home/mjw/Code/PROPHET-Load_LSTM-231011/lstm_forecaster.ini"
#ini_path = "C:/Users/Admin/Code/PROPHET-Load_LSTM-231011/lstm_forecaster.ini"

#lstm_forecaster_training.main([ini_path])
#lstm_forecaster_training.grid_search([ini_path])
lstm_forecaster.main([ini_path])