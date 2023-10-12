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


def create_query(talktoSQL, table_mysql, column_mysql, time_column):
    query = "SELECT {0}, {1} FROM {2}.{3} " \
            "WHERE({0} <=now())".format(time_column, column_mysql, talktoSQL._database, table_mysql)
    return query


def create_query_test(talktoSQL, table_mysql, column_mysql, time_column, days):
    query = "SELECT {0}, {1} FROM {2}.{3} " \
            "WHERE({0} <= NOW() AND {0} >= DATE_SUB(NOW(), INTERVAL '{4}' DAY))".format(time_column, column_mysql, talktoSQL._database, table_mysql, days)
    return query