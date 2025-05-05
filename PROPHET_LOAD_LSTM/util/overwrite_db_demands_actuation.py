import sys
import atexit
import pandas
import sqlalchemy
from pathlib import Path
import configparser

def read_config(config_file):
    config = configparser.ConfigParser()
    config.sections()
    config_file_fullname = Path(config_file)
    if config_file_fullname.exists():
        config.read(config_file)
    else:
        sys.exit(1)
    return config

def populate_opts(config):
    data_opt = {
        'n_back': config.getint("data_opt", "n_back"),  # 4*24*7
        'n_timesteps': config.getint("data_opt", "n_timesteps"),  # 4*4
        'lag': config.getint("data_opt", "lag"),
        'tr_per': config.getfloat("data_opt", "tr_per"),
        'out_col': config.get("data_opt", "out_col").split(','),
        'features': config.get("data_opt", "features").split(','),
        'freq': config.getint("data_opt", "freq"),
        #'tr_days_step': config.getint("data_opt", "tr_days_step"), # unused so removed
        'data_path': Path(config.get("data_opt", "data_path")),
        'data_train_path':Path(config.get("data_opt", "data_train_path")),
        'out_dir': Path(config.get("data_opt", "out_dir")),
        'db_demands_actuation_csv': Path(config.get("data_opt", "db_demands_actuation_csv")),
        
    }
    if data_opt['features'] == ['']:
        data_opt['columns'] = ['power'] #data_opt['out_col']
    else:
        data_opt['columns'] = data_opt['features'] + ['power'] #data_opt['out_col']
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


class MySQLConnector:
    def __init__(self, database='test', host='localhost', port=3306, user='root', password='test'):
        self._user = user
        self._password = password
        self._database = database
        self._host = host
        self._port = port
        self._engine = sqlalchemy.create_engine(
            f"mysql+pymysql://{self._user}:{self._password}@{self._host}:{self._port}/{self._database}",
            connect_args={'connect_timeout': 10})
        atexit.register(lambda x: x.finalize(), self)

    def read_query(self, query, date_columns, **kwargs):    
        with self._engine.connect() as connection:
            df = pandas.read_sql_query(sql=query, con=connection, parse_dates=date_columns)

        if 'no_utc_timezone' in kwargs.keys():
            if kwargs['no_utc_timezone']:
                pass
            else:
                for col in date_columns:
                    df[col] = df[col].dt.tz_localize("UTC")
        else:
            for col in date_columns:
                df[col] = df[col].dt.tz_localize("UTC")

        return df

    def read_table(self, table, date_columns):
        with self._engine.connect() as connection:
            df = pandas.read_sql_table(table_name=table, con=connection, parse_dates=date_columns)

        for col in date_columns:
            df[col] = df[col].dt.tz_localize("UTC")

        return df

    def write(self, df, table, if_exists='append'):
        with self._engine.connect() as connection:
            df.to_sql(name=table, con=connection, if_exists=if_exists, index=False)

    def write_query(self, query):
        with self._engine.connect() as connection:
            connection.execute(query)

    def clear(self, table):
        with self._engine.connect() as connection:
            connection.execute("TRUNCATE table " + table)

    def finalize(self):
        self._engine.dispose()

def create_query(talktoSQL, table_mysql, column_mysql=None, time_column=None):
    if column_mysql is None and time_column is None:
        query = f'SELECT * FROM {talktoSQL._database}.{table_mysql}'
    else:
        query = f'SELECT {time_column}, {column_mysql} FROM {talktoSQL._database}.{table_mysql}'
    #query = query + 'WHERE({0} <=now())'
    return query        


if __name__ == '__main__':

    ini_path = r"D:\GitHub\PROPHET-Load_LSTM\lstm_forecaster.ini"

    config = read_config(ini_path)
    data_opt, model_opt = populate_opts(config)

    talktoSQL = MySQLConnector( database=config["mysql"]["database"],
                                host=config["mysql"]["host"],
                                port=config["mysql"]["port"],
                                user=config["mysql"]["user"],
                                password=config["mysql"]["password"])
    
    """
    First retrieve the old
    """

    db_demands_actuation_query = create_query(  talktoSQL,
                                                config["sql_table"]["ev_power_table"],
                                                #config["sql_table"]["ev_power_field"],
                                                #config["sql_table"]["time_column"]
                                                )
    
    df = talktoSQL.read_query(db_demands_actuation_query, {config["sql_table"]["time_column"]})

    df.to_csv('data/db_demands_actuation_old.csv')


    """
    Now go find that csv and change the columns as you want
    make sure its comma-delimited and the datetime formats are yyyy-mm-dd HH:MM:SS
    """


    """
    Then send the new
    """

    df = pandas.read_csv(data_opt['db_demands_actuation_csv'])

    print(f"The csv at {data_opt['db_demands_actuation_csv']} is imported as:\n",df)

    response = input('Okay to overwrite the data in db_demands_actuation? (Y continue): ')

    if response.capitalize() == 'Y':
        print('Writing..')
        talktoSQL.write(df, 'db_demands_actuation', if_exists='replace')
        print('Done (you should also confirm in MySQL Workbench)')
    else:
        print('No? Rats. Nothing written.')

    talktoSQL.finalize()