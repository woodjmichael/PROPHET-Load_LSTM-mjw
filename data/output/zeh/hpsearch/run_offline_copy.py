import os
from pathlib import Path
from math import nan
from datetime import datetime as dt
from datetime import timedelta as td
from random import shuffle

from shutil import copyfile

import numpy as np 
import pandas as pd
 
from PROPHET_LOAD_LSTM.main import lstm_forecaster_training, lstm_forecaster
from PROPHET_LOAD_LSTM.util import util

from tensorflow import keras
from PROPHET_LOAD_LSTM.util.model_util import attention, Custom_Loss_Prices

from PROPHET_LOAD_LSTM.main import bayes_ensemble as bayes

tic = dt.now()

"""
Offline options
"""

#ini_path = "/home/mjw/Code/PROPHET-Load_LSTM-mjw/lstm_forecaster.ini"
ini_path = r"C:\Users\Admin\Code\PROPHET-Load_LSTM\lstm_forecaster.ini"

TRAIN =     0
PRELOAD =   0
TEST =      0
HP_SEARCH = True

TEST_OUTPUT_FILENAME = 'test_forecasts.csv' #'train_test_forecasts.csv'
#TEST_BEGIN = '2019-9-13 0:00' # impianto 4
#TEST_END = '2019-12-27 23:00' # impianto 4
TEST_BEGIN = '2019-8-6' # zeh
TEST_END = '2019-12-25' # zeh
TEST_FREQ = '1h'# '24h' for peaks, else '1h'

HP_CONTINUE_PREVIOUS_SEARCH =   1 # picks up where last search left off
HP_COPY_PREVIOUS_SPACE =        0 # use existing search space but build new models
HP_SHUFFLE =                    True
HP_RESULTS_FILENAME = 'hp_search_patience1b.csv'
HP_PREV_RESULTS_FILENAME = 'hp_search_patience1.csv' #Path('hp_search_patience1_customLF.csv')

LIMIT = None # int or None


"""
Options

"""

data_opt, model_opt = util.read_options(ini_path)

if not os.path.exists(data_opt['out_dir']):
    os.makedirs(data_opt['out_dir'])
if not os.path.exists(model_opt['model_path']):
    os.makedirs(model_opt['model_path'])

def preload_data():
    df = pd.read_csv(   data_opt['data_path'],
                        index_col=0,
                        parse_dates=True,
                        comment='#')
    df.rename(columns={data_opt['out_col'][0]: 'power'}, inplace=True)
    df.index = df.index.tz_localize('UTC')
    
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.dayofweek
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute

    df = df[data_opt['columns']].copy()
    
    model = keras.models.load_model(model_opt["model_path"] / Path('model.h5'),
                                    custom_objects={"attention": attention,
                                                    'Custom_Loss_Prices':Custom_Loss_Prices,
                                                    })
    
    return df, model

def test_range(df,model):
    filename = TEST_OUTPUT_FILENAME
    if os.path.exists(data_opt['out_dir']/Path(filename)):
        t=pd.Timestamp.now()
        filename = filename.split('.')[0] + f'_{t.year-2000}{t.month}{t.day}{t.hour}{t.minute}.csv'        
    
    forecasts = pd.DataFrame()
    for t in pd.date_range(TEST_BEGIN,TEST_END,freq=TEST_FREQ)[:LIMIT]:
    #for t in pd.date_range('2017-1-8 0:00','2019-12-28 0:00',freq='1h')[:limit]:
        print('test timestamp_update:',t)
        _, _, _, new_forecast = lstm_forecaster.main([ini_path],t,df=df,model=model)
        new_forecast.index = new_forecast.index.tz_convert(None)
        new_forecast.reset_index(inplace=True)
        new_forecast.rename(columns={'index':'timestamp_forecast'},inplace=True)
        new_forecast.insert(0,'timestamp_forecast_update',t)
        forecasts = pd.concat([forecasts,new_forecast],axis=0,ignore_index=True)    
    forecasts.to_csv(data_opt['out_dir'] / Path(filename))    

def hyper_parameter_search():
    # setup files
    results_filepath = data_opt['out_dir'] / Path(HP_RESULTS_FILENAME)
    prev_results_filepath = data_opt['out_dir'] / Path(HP_PREV_RESULTS_FILENAME)
    if os.path.exists(results_filepath) and not HP_CONTINUE_PREVIOUS_SEARCH:
        input('Warning you may be overwriting results (enter to continue):')
    copyfile('lstm_forecaster.ini',data_opt['out_dir']/Path('lstm_forecaster_copy.ini'))
    copyfile('run_offline.py',data_opt['out_dir']/Path('run_offline_copy.py'))

    # check for existing hp search
    if (HP_CONTINUE_PREVIOUS_SEARCH or HP_COPY_PREVIOUS_SPACE) \
            and os.path.exists(prev_results_filepath):
        results = pd.read_csv(prev_results_filepath,index_col=0)
        prev_search_space = []
        for units,n_back,dropout in results[['units','n_back','dropout']].values:
            prev_search_space.append((int(units),int(n_back),dropout))
            
    if HP_CONTINUE_PREVIOUS_SEARCH:
        input(f'\nContinuting the search at {prev_results_filepath} ok? (enter to continue):')
        results = pd.read_csv(prev_results_filepath,index_col=0)
    else:
        #results = pd.DataFrame(columns=['units','n_back','dropout','vloss','mae_pers','mae_pred','frac_pos_skill','frac_0_mae_pers','skill','sec_elaps'])
        results = pd.DataFrame(columns=['units','n_back','dropout','vloss','sec_elaps'])

    # create search space
    if HP_COPY_PREVIOUS_SPACE and not HP_CONTINUE_PREVIOUS_SEARCH:
        input(f'\nCopying previous searched space at {prev_results_filepath} ok? (enter to continue):')
        search_space = prev_search_space.copy()
    else:    
        search_space = []
        for units in [int(24*x) for x in [0.5,1,2,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32,36]]:
            for n_back in [int(24*x) for x in [0.5,1,2,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32,36]]:
                for dropout in [0,0.1,0.2]:
                    if HP_CONTINUE_PREVIOUS_SEARCH:
                        if (units,n_back,dropout) not in prev_search_space:
                            search_space.append((units,n_back,dropout))         
                    else:    
                        search_space.append((units,n_back,dropout))     
    if HP_SHUFFLE:
        shuffle(search_space)

    # for each model in search space
    for units,n_back,dropout in search_space[:LIMIT]:
        
        print(f'\n\n\n ///// HP TEST: units={units} n_back={n_back} dropout={dropout}/////\n')
        tic2 = pd.Timestamp.now()
        try:
            # train
            vloss = lstm_forecaster_training.main([ini_path],units,n_back,dropout)
            
            # test
            
            # model = keras.models.load_model(model_opt["model_path"] / Path('model.h5'),
            #             custom_objects={"attention": attention,
            #                             #'Custom_Loss_Prices':Custom_Loss_Prices,
            #                             })
            
            # forecasts = pd.DataFrame()
            # for t in pd.date_range(TEST_BEGIN,TEST_END,freq=TEST_FREQ):
            #     print('timestamp_update:',t)
            #     _, _, _, new_forecast = lstm_forecaster.main([ini_path],
            #                                                 t,
            #                                                 units=units,
            #                                                 n_back=n_back,
            #                                                 dropout=dropout,
            #                                                 df=df,
            #                                                 model=model)
            #     new_forecast.index = new_forecast.index.tz_convert(None)
            #     new_forecast.reset_index(inplace=True)
            #     new_forecast.rename(columns={'index':'timestamp_forecast'},inplace=True)
            #     new_forecast.insert(0,'timestamp_forecast_update',t)
            #     forecasts = pd.concat([forecasts,new_forecast],axis=0,ignore_index=True)        
            
            # # errors
            # test_mae_pers = (forecasts.persist - forecasts.power).dropna().abs().mean()
            # test_mae_pred = (forecasts.predicted_activepower_ev_1 - forecasts.power).dropna().abs().mean()
            # test_skill = 1 - test_mae_pred / test_mae_pers
            # skills = 1 - (forecasts.predicted_activepower_ev_1 - forecasts.power).abs() / (forecasts.persist - forecasts.power).abs()
            # frac_pos_skills = skills[skills>0].shape[0] / skills.shape[0]
            # frac_0_mae_pers = len(forecasts.loc[(forecasts.persist-forecasts.power)==0,:]) / len(forecasts)                
            
        except:
            print('//////////////////// TRAINING FAIL /////////////////////')
            #vloss,test_mae_pers,test_mae_pred,test_skill,frac_pos_skills,frac_0_mae_pers=nan,nan,nan,nan,nan,nan
            vloss=nan
            
        # output
        results.loc[len(results)] = {   'units':units,
                                        'n_back':n_back,
                                        'dropout':dropout,
                                        'vloss':vloss,
                                        #'mae_pers':test_mae_pers,
                                        #'mae_pred':test_mae_pred,                                        
                                        #'frac_pos_skill':frac_pos_skills,
                                        #'frac_0_mae_pers':frac_0_mae_pers,
                                        #'skill':test_skill,
                                        'sec_elaps':(pd.Timestamp.now()-tic2).seconds}
        results.sort_values('vloss',ascending=False).to_csv(results_filepath)

"""
Train
"""

if TRAIN:
    lstm_forecaster_training.main([ini_path])


"""
Pre-Load
"""

if PRELOAD:
    df,model = preload_data()
else:
    df = None
    model = None    

"""
Test
"""

if TEST: 
    test_range(df,model)


"""
Hyper parameter search
"""

if HP_SEARCH:        
    hyper_parameter_search()
    
print('Ding! Fries are done. Elapsed time:',dt.now()-tic)    