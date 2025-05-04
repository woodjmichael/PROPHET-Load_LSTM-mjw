import argparse
import configparser
import sys
from datetime import datetime, timedelta
from pathlib import Path

import daiquiri


# Logger definition
LOGGER = daiquiri.getLogger(__name__)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="EMS predictive optimizer")
    parser.add_argument("config", help="configuration file")
    return parser.parse_args(argv)


def read_config(config_file):
    config = configparser.ConfigParser()
    config.sections()
    config_file_fullname = Path(config_file)
    if config_file_fullname.exists():
        config.read(config_file)
    else:
        LOGGER.error(f"Not possible to access scheduler configuration file")
        sys.exit(1)
    return config

def read_options(ini_path=None,config=None):
    config = read_config(ini_path) if config is None else config
    data_opt = {
        'n_back': config.getint("data_opt", "n_back"),  # 4*24*7
        'n_timesteps': config.getint("data_opt", "n_timesteps"),  # 4*4
        'lag': config.getint("data_opt", "lag"),
        'tr_per': config.getfloat("data_opt", "tr_per"),
        'out_col': config.get("data_opt", "out_col").split(','),
        'target_col': config.get("data_opt", "target_col").split(','),
        'features': config.get("data_opt", "features").split(','),
        'freq': config.getint("data_opt", "freq"),
        #'tr_days_step': config.getint("data_opt", "tr_days_step"), # unused so removed
        'data_path': Path(config.get("data_opt", "data_path")),
        'out_dir': Path(config.get("data_opt", "out_dir")),
        #'data_train_path':Path(config.get("data_opt", "data_path")) / Path(config.get('data_opt', 'train_csv')),
    }
    if data_opt['features'] == ['']:
        #data_opt['columns'] = ['power'] #data_opt['out_col']
        data_opt['columns'] = data_opt['target_col']
    else:
        #data_opt['columns'] = data_opt['features'] + ['power'] #data_opt['out_col']
        data_opt['columns'] = data_opt['features'] + data_opt['target_col'] #data_opt['out_col']
    data_opt['n_features'] = len(data_opt['columns'])

    model_opt = {'Dense_input_dim': config.getint("model_opt", "Dense_input_dim"),
                 'LSTM_num_hidden_units': config.getint("model_opt", "LSTM_num_hidden_units"), #list(map(int, config.get("model_opt", "LSTM_num_hidden_units").split(','))),
                 'LSTM_layers': config.getint("model_opt", "LSTM_layers"),
                 'metrics': config.get("model_opt", "metrics"),
                 'optimizer': config.get("model_opt", "optimizer"),
                 'patience': config.getint("model_opt", "patience"),
                 'epochs': config.getint("model_opt", "epochs"),
                 'validation_split': config.getfloat("model_opt", "validation_split"),
                 'model_path': Path(config.get("model_opt", "model_path")),
                 'Dropout_rate': config.getfloat("model_opt", "Dropout_rate"),
                 'input_dim': (data_opt['n_back'], data_opt['n_features']),
                 'dense_out': data_opt['n_timesteps']
                 } 
    
    return data_opt, model_opt 


def create_query(talktoSQL, table_mysql, column_mysql, time_column):
    query = "SELECT {0}, {1} FROM {2}.{3} " \
            "WHERE({0} <=now())".format(time_column, column_mysql, talktoSQL._database, table_mysql)
    return query


def create_query_test(talktoSQL, table_mysql, column_mysql, time_column, days):
    query = "SELECT {0}, {1} FROM {2}.{3} " \
            "WHERE({0} <= NOW() AND {0} >= DATE_SUB(NOW(), INTERVAL '{4}' DAY))".format(time_column, column_mysql, talktoSQL._database, table_mysql, days)
    return query