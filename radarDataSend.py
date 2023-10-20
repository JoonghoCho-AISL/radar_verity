import os
from MobiusAPI import http_post_get
import json
import numpy as np

# disk_path = 'counting_disk'
# cylinder_path = 'counting_cylinder'

def sample_data(path):
    path_list = sorted(os.listdir(path))
    data = None
    for i in path_list:
        temp_path = os.path.join(path, i)
        temp_data = np.reshape(np.load(temp_path)[0], (1, -1, 1))
        if data is None:
            data = temp_data
        else:
            data = np.concatenate((data, temp_data), axis=0)
    return data

# disk_data = sample_data(disk_path)
# cylinder_data = sample_data(cylinder_path)

# sample_data = np.concatenate((disk_data, cylinder_data), axis=0)
URI = '/Mobius/radarEye/sensor1/rawdata'
AE_ID = 'radarEye1'

# try:
#     for i in range(sample_data.shape[0]):
#         input("Press Enter to Measure Radar Data")
#         send_data = {'data_1': np.expand_dims(sample_data[i], axis=0).tolist()}
#         http_post_get.mobius_post(URI, AE_ID, AE_ID, send_data)
# except KeyboardInterrupt:
#     print('End')

if __name__ == '__main__':
    data_path = input('Input Data Path: ')
    path = os.path.join('test_data', data_path)
    data = sample_data(path)
    print(data.shape)
    for i in range(data.shape[0]):
        input("Press Enter to Measure Radar Data")
        send_data = {'input_1' :  np.expand_dims(data[i], axis=0).tolist()}
        http_post_get.mobius_post(URI, AE_ID, AE_ID, send_data) 