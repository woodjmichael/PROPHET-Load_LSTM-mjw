MICROGRID_PC = False

import pandas as pd
from datetime import datetime
from pickle import dump
from PROPHET_LOAD_LSTM.util import util
from PROPHET_LOAD_LSTM.util.data_util import prepare_data
from PROPHET_LOAD_LSTM.util.model_util import create_model_attention

import argparse
import configparser
from datetime import datetime, timedelta
from pathlib import Path
from random import shuffle
import shutil

if MICROGRID_PC:
    from PROPHET_DB import mysql

# import daiquiri
# import sklearn.preprocessing

# from PROPHET_DB.setup_logger import setup_logger

# Logger definition
# LOGGER = daiquiri.getLogger(__name__)

def main(argv=None,units=None,n_back=None,dropout=None):
    
    # read configuration file
    args = util.parse_arguments(argv)
    config = util.read_config(args.config)
    data_opt, model_opt = util.read_options(config=config)
    
    # tired of manually copying .ini to PROPHET-EMS
    try:
        ems_ini_path = Path(argv[0]).parent.parent / Path('PROPHET-EMS/ini/lstm_forecaster.ini')
        shutil.copyfile(argv[0],ems_ini_path)    
    except:
        print('Error copying .ini file to PROPHET-EMS')
    
    model_opt['LSTM_num_hidden_units'] = model_opt['LSTM_num_hidden_units'] if units is None else units
    data_opt['n_back'] = data_opt['n_back'] if n_back is None else n_back
    model_opt['Dropout_rate'] = model_opt['Dropout_rate'] if dropout is None else dropout
    
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

    ## definizione tempi inizio
    now = datetime.utcnow()
    fine_tr = pd.to_datetime(now).tz_localize('UTC')  ## METTERE A POSTO MA SECONDARIO

    mask_tr = (df.index < fine_tr)
    train = df.loc[mask_tr]

    #### addestra il modello
    #train_X, train_y, scaler_X, scaler_y = prepare_data(train, data_opt)
    train_X, train_y, _, _, scaler_X, scaler_y = prepare_data(train, data_opt)
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    dump(scaler_X, open(model_opt["model_path"] / Path("scaler_in.pkl"), 'wb'))
    dump(scaler_y, open(model_opt["model_path"] / Path("scaler_out.pkl"), 'wb'))

    model, history = create_model_attention(model_opt, train_X, train_y)
    model.save(model_opt['model_path'] / Path("model.h5"))

    val_loss = min(history.history['val_loss'])
    
    return val_loss
        

if __name__ == "__main__":
    main()
