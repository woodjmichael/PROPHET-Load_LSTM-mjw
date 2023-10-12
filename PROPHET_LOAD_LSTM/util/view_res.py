import pandas as pd
from PROPHET_LOAD_LSTM.util import util
from PROPHET_DB import mysql
import matplotlib.pyplot as plt

def main(argv=None):
    # read configuration file
    args = util.parse_arguments(argv)

    config = util.read_config(args.config)

    #
    talktoSQL = mysql.MySQLConnector(database=config["mysql"]["database"],
                                     host=config["mysql"]["host"],
                                     port=config["mysql"]["port"],
                                     user=config["mysql"]["user"],
                                     password=config["mysql"]["password"])

    load_query_meas = 'SELECT timestamp_utc, measured_activepower_load_1 FROM q1012.db_demands_actuation'
    df_meas = talktoSQL.read_query(load_query_meas, {config["sql_table"]["time_column"]})

    df_meas[config["sql_table"]["time_column"]] = pd.to_datetime(df_meas[config["sql_table"]["time_column"]],
                                                            format='%Y-%m-%d %H:%M:%S%z')


    load_query_pred = 'SELECT * FROM q1012.db_lstm_prediction;'
    df_pred = talktoSQL.read_query(load_query_pred, {config["sql_table"]["time_column"]})
    df_pred[config["sql_table"]["time_column"]] = pd.to_datetime(df_pred[config["sql_table"]["time_column"]],
                                                            format='%Y-%m-%d %H:%M:%S%z')

    frames = [df_meas, df_pred]

    b = pd.merge(df_meas, df_pred, on='timestamp_utc')
    c = b.drop_duplicates(subset=['timestamp_utc'], keep='last')
    plt.figure()
    plt.plot(c['timestamp_utc'], c['measured_activepower_load_1'], label='misu')
    plt.plot(c['timestamp_utc'], c['predicted_activepower_ev_1'], label='pre')
    plt.legend()
    plt.show()

    a = 1

if __name__ == "__main__":
    main()
