from sklearn.model_selection import train_test_split
import numpy as np
from data_processing import read_edinburgh_data 

_, ed_agent_ids = read_edinburgh_data()
ed_train_agent_ids, ed_test_agent_ids = train_test_split(ed_agent_ids, test_size=0.2)

np.save('Edinburgh_train_agent_ids', ed_train_agent_ids)
np.save('Edinburgh_test_agent_ids', ed_test_agent_ids)