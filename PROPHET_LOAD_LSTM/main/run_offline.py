import sys
import time
from datetime import datetime
import pandas as pd
import lstm_forecaster

idx_test = pd.date_range('2023-9-8 0:00','2023-10-5',freq='1h')[:1]

for t in idx_test:
    print(t)
    lstm_forecaster.main(sys.argv[1:],output='sql-test',t_fake=t)

"""while(1):
    now = datetime.utcnow()
    if now.minute == 0: #if now.hour == 0 and now.minute == 15:
        lstm_forecaster.main(sys.argv[1:],output='sql-test')
        time.sleep(55*60) #time.sleep(int(23.75*60*60))"""