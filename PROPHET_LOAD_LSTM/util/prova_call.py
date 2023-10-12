import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pickle import load
from tensorflow import keras

from PROPHET_EV.util import util
from PROPHET_EV.util.data_util import prepare_data_test
from PROPHET_EV.util.model_util import attention

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
    talktoSQL = mysql.MySQLConnector(database=config["mysql"]["database"],
                                     host=config["mysql"]["host"],
                                     port=config["mysql"]["port"],
                                     user=config["mysql"]["user"],
                                     password=config["mysql"]["password"])

    query = "SELECT * FROM q1012.db_demands_actuation;"
    df = talktoSQL.read_query(query, {config["sql_table"]["time_column"]})
    a = 1


if __name__ == "__main__":
    main()