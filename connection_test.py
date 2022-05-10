import time
import numpy as np
import pandas as pd

from plantsim.plantsim import Plantsim
from plantsim.pandas_table import PandasTable

model_path = r'C:\Users\jastr\PycharmProjects\Discrete-Simulation-Reinforcement-Learning\PlantSim_1\model.spp'
p = Plantsim(version='16.1', license_type='Educational', path_context='.Modelle.Modell', model=model_path, socket=None, visible=False)


print('Begin simulation...')
p.start_simulation()

time.sleep(3)

res_table_mapping = PandasTable(p, 'Results')
res_pd = res_table_mapping.table

print(res_pd, '\n\n', 'Set custom simulation time:')
res_pd.at[0, "SimTime"] = 10.0
print(res_pd)

print('Stop simulation')
p.quit()
