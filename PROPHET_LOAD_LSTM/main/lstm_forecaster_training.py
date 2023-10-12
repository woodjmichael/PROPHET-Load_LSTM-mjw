MICROGRID_PC = False

import pandas as pd
from datetime import datetime
from pickle import dump
from PROPHET_LOAD_LSTM.util import util
from PROPHET_LOAD_LSTM.util.data_util import prepare_data_train
from PROPHET_LOAD_LSTM.util.model_util import create_model_attention

import argparse
import configparser
from datetime import datetime, timedelta
from pathlib import Path

if MICROGRID_PC:
    from PROPHET_DB import mysql

# import daiquiri
# import sklearn.preprocessing

# from PROPHET_DB.setup_logger import setup_logger

# Logger definition
# LOGGER = daiquiri.getLogger(__name__)


def main(argv=None):
    # read configuration file
    args = util.parse_arguments(argv)

    config = util.read_config(args.config)
    data_opt = {
        'n_back': config.getint("data_opt", "n_back"),  # 4*24*7
        'n_timesteps': config.getint("data_opt", "n_timesteps"),  # 4*4
        'lag': config.getint("data_opt", "lag"),
        'tr_per': config.getfloat("data_opt", "tr_per"),
        'out_col': config.get("data_opt", "out_col").split(','),
        'features': config.get("data_opt", "features").split(','),
        'freq': config.getint("data_opt", "freq"),
        'tr_days_step': config.getint("data_opt", "tr_days_step"),
        'data_path': Path(config.get("data_opt", "data_path")),
        'data_train_path':Path(config.get("data_opt", "data_path")) / Path(config.get('data_opt', 'train_csv')),
    }
    if data_opt['features'] == ['']:
        data_opt['columns'] = data_opt['out_col']
    else:
        data_opt['columns'] = data_opt['features'] + data_opt['out_col']
    data_opt['n_features'] = len(data_opt['columns'])

    model_opt = {'Dense_input_dim': config.getint("model_opt", "Dense_input_dim"),
                 'LSTM_num_hidden_units': config.getint("model_opt", "LSTM_num_hidden_units"), #list(map(int, config.get("model_opt", "LSTM_num_hidden_units").split(','))),
                 'LSTM_layers': config.getint("model_opt", "LSTM_layers"),
                 'metrics': config.get("model_opt", "metrics"), 'optimizer': config.get("model_opt", "optimizer"),
                 'patience': config.getint("model_opt", "patience"),
                 'epochs': config.getint("model_opt", "epochs"),
                 'validation_split': config.getfloat("model_opt", "validation_split"),
                 'model_path': Path(config.get("model_opt", "model_path")),
                 'Dropout_rate': config.getfloat("model_opt", "Dropout_rate"),
                 'input_dim': (data_opt['n_back'], data_opt['n_features']), 'dense_out': data_opt['n_timesteps']
                 }
    # logger_config = {'mailhost': (config["logger"]["mailhost"], config.getint("logger", "port")),
    #                  'fromaddr': config["logger"]["fromaddr"],
    #                  'toaddr': config["logger"]["toaddrs"],
    #                  'subject': config["logger"]["subject"],
    #                  'credentials': (config["logger"]["username"], config["logger"]["password"]),
    #                  'mail_handler': config.getboolean("logger", "mail_handler"),
    #                  'backup_count': config.getint("logger", "backup_count")}

    #
    # talktoSQL = mysql.MySQLConnector(database=config["mysql"]["database"],
    #                                  host=config["mysql"]["host"],
    #                                  port=config["mysql"]["port"],
    #                                  user=config["mysql"]["user"],
    #                                  password=config["mysql"]["password"])

    # (talktoSQL, table_mysql, column_mysql, time_column)
    # ev_query = util.create_query(talktoSQL, config["tables"]["measures"], config["columns"]["measures"], config["tables"]["time_column"])
    # ev_query = ev_query + 'order by TimeCol desc limit 1000'
    # df = talktoSQL.read_query(ev_query, {config["tables"]["time_column"]})
    # df[config["tables"]["time_column"]] = pd.to_datetime(df[config["tables"]["time_column"]], format='%Y-%m-%d %H:%M:%S%z')#.dt.tz_convert('America/Los_Angeles')
    #
    # df.rename(columns={config["columns"]["measures"]: 'power'}, inplace=True)
    # df.set_index(config["tables"]["time_column"], inplace=True)

    #df.to_csv('C:\\Users\\PROPHET\\Desktop\\Mike\\test.csv')

    #df = pd.read_csv("C:\\Users\\PROPHET\\Documents\\Mike\\train_JPL_5_mjw.csv", sep=",")
    #df[config["tables"]["time_column"]] = pd.to_datetime(df['timestamp_utc'], format='%d/%m/%Y %H:%M')
    #df.set_index(config["tables"]["time_column"], inplace=True)
    
    df = pd.read_csv(data_opt['data_train_path'],
                     index_col=0,
                     parse_dates=True)
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
    train_X, train_y, scaler_X, scaler_y = prepare_data_train(train, data_opt)
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    dump(scaler_X, open(model_opt["model_path"] / Path("scaler_in.pkl"), 'wb'))
    dump(scaler_y, open(model_opt["model_path"] / Path("scaler_out.pkl"), 'wb'))

    model, history = create_model_attention(model_opt, train_X, train_y)
    model.save(model_opt['model_path'] / Path("model.h5"))


def grid_search(argv=None):
    for units in [24*x for x in range(20,100)]:
        print(f'\n\n\n ///// units = {units} /////\n')
        # read configuration file
        args = util.parse_arguments(argv)

        config = util.read_config(args.config)
        data_opt = {
            'n_back': config.getint("data_opt", "n_back"),  # 4*24*7
            'n_timesteps': config.getint("data_opt", "n_timesteps"),  # 4*4
            'lag': config.getint("data_opt", "lag"),
            'tr_per': config.getfloat("data_opt", "tr_per"),
            'out_col': config.get("data_opt", "out_col").split(','),
            'features': config.get("data_opt", "features").split(','),
            'freq': config.getint("data_opt", "freq"),
            'tr_days_step': config.getint("data_opt", "tr_days_step"),
            'data_path': Path(config.get("data_opt", "data_path")),
            'data_train_path':Path(config.get("data_opt", "data_path")) / Path(config.get('data_opt', 'train_csv')),
        }
        if data_opt['features'] == ['']:
            data_opt['columns'] = data_opt['out_col']
        else:
            data_opt['columns'] = data_opt['features'] + data_opt['out_col']
        data_opt['n_features'] = len(data_opt['columns'])

        model_opt = {'Dense_input_dim': config.getint("model_opt", "Dense_input_dim"),
                    'LSTM_num_hidden_units': units,
                    'LSTM_layers': config.getint("model_opt", "LSTM_layers"),
                    'metrics': config.get("model_opt", "metrics"), 'optimizer': config.get("model_opt", "optimizer"),
                    'patience': config.getint("model_opt", "patience"),
                    'epochs': config.getint("model_opt", "epochs"),
                    'validation_split': config.getfloat("model_opt", "validation_split"),
                    'model_path': Path(config.get("model_opt", "model_path")),
                    'Dropout_rate': config.getfloat("model_opt", "Dropout_rate"),
                    'input_dim': (data_opt['n_back'], data_opt['n_features']), 'dense_out': data_opt['n_timesteps']
                    }
        # logger_config = {'mailhost': (config["logger"]["mailhost"], config.getint("logger", "port")),
        #                  'fromaddr': config["logger"]["fromaddr"],
        #                  'toaddr': config["logger"]["toaddrs"],
        #                  'subject': config["logger"]["subject"],
        #                  'credentials': (config["logger"]["username"], config["logger"]["password"]),
        #                  'mail_handler': config.getboolean("logger", "mail_handler"),
        #                  'backup_count': config.getint("logger", "backup_count")}

        #
        # talktoSQL = mysql.MySQLConnector(database=config["mysql"]["database"],
        #                                  host=config["mysql"]["host"],
        #                                  port=config["mysql"]["port"],
        #                                  user=config["mysql"]["user"],
        #                                  password=config["mysql"]["password"])

        # (talktoSQL, table_mysql, column_mysql, time_column)
        # ev_query = util.create_query(talktoSQL, config["tables"]["measures"], config["columns"]["measures"], config["tables"]["time_column"])
        # ev_query = ev_query + 'order by TimeCol desc limit 1000'
        # df = talktoSQL.read_query(ev_query, {config["tables"]["time_column"]})
        # df[config["tables"]["time_column"]] = pd.to_datetime(df[config["tables"]["time_column"]], format='%Y-%m-%d %H:%M:%S%z')#.dt.tz_convert('America/Los_Angeles')
        #
        # df.rename(columns={config["columns"]["measures"]: 'power'}, inplace=True)
        # df.set_index(config["tables"]["time_column"], inplace=True)

        #df.to_csv('C:\\Users\\PROPHET\\Desktop\\Mike\\test.csv')

        #df = pd.read_csv("C:\\Users\\PROPHET\\Documents\\Mike\\train_JPL_5_mjw.csv", sep=",")
        #df[config["tables"]["time_column"]] = pd.to_datetime(df['timestamp_utc'], format='%d/%m/%Y %H:%M')
        #df.set_index(config["tables"]["time_column"], inplace=True)
        
        df = pd.read_csv(data_opt['data_train_path'],
                        index_col=0,
                        parse_dates=True)
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
        train_X, train_y, scaler_X, scaler_y = prepare_data_train(train, data_opt)
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

        dump(scaler_X, open(model_opt["model_path"] / Path("scaler_in.pkl"), 'wb'))
        dump(scaler_y, open(model_opt["model_path"] / Path("scaler_out.pkl"), 'wb'))

        model, history = create_model_attention(model_opt, train_X, train_y)
        #model.save(model_opt['model_path'] / Path("model.h5"))

        val_loss = min(history.history['val_loss'])
        model.save(model_opt['model_path'] / Path(f"model_units{units}_vloss{val_loss:.4f}.h5"))
    

if __name__ == "__main__":
    main()