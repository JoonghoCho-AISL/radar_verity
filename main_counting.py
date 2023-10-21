from tkinter import S
import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
# import time
from tqdm import tqdm
import json
import os
import argparse
from MobiusAPI import http_post_get
import pickle as pkl
from scipy.ndimage import gaussian_filter1d
from models import Resnet, basemodel, Resnet_pool
import matplotlib.pyplot as plt
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def save_data_plot(data):
    plt.imshow(data, aspect='auto', cmap='jet', interpolation='none')
    # plt.imshow(np.abs(profile), aspect='auto', cmap='jet', interpolation='none')
    plt.colorbar(label="dB")
    plt.xlabel("Range Bin")
    plt.ylabel("Pulse Number")
    plt.title(f"Range Profiles: data")

    plt.tight_layout()
    # plt.show()
    plt.savefig('./data.png')


# MAX = 20655.93

class radar_raspi():
    def __init__(self,
                    ip_address='127.0.0.1',
                    start_point = 16,
                    num_points = 32,
                    hwaas = 8,
                    sweep_per_frame = 10,
                    frame_rate = 10
                    ):
        self.client = a121.Client(ip_address = ip_address)

        self.client.connect()
        #start_distance = start_point * 2.5mm
        # start_point = 160 # 400mm
        #end_distance = start_point * 2.5mm + num_points * 2.5mm 
        # num_points = (end_dis - start_point * 2.5) / 2.5 + 1
        # num_points = 301 # 1,150mm

        sensor_config = a121.SensorConfig(
            subsweeps=[
                a121.SubsweepConfig(
                    start_point = start_point,
                    step_length = 1,
                    num_points = num_points,
                    profile = a121.Profile.PROFILE_1,
                    hwaas = hwaas,
                ),
            ],
            sweeps_per_frame = sweep_per_frame,
            frame_rate = frame_rate,
        )

        self.client.setup_session(sensor_config)
        self.client.start_session()
    
    def read_data(self):
        raw_data = self.client.get_next()
        frame = raw_data.frame
        data = np.expand_dims(np.abs(np.mean(frame, axis=0)), axis = 0)
        return data
    
    def read_mean_var(self):
        raw_data = self.client.get_next()
        frame = raw_data.frame
        mean = np.mean(frame, axis = 0)
        var = np.var(frame, axis = 0)
        data = np.expand_dims(np.abs(np.concatenate((mean, var), axis = 0)), axis = 0 )
        return data
        
    def disconnect(self):
        self.client.disconnect()

def preprocessing(arr):
    amp = np.abs(arr)
    phs = np.angle(arr)
    data = np.array([amp, phs])

def main():
    parser = argparse.ArgumentParser(description = 'select ip, road name, save or post')
    # parser.add_argument('-i', '--ip', action = 'store', default = '192.168.222.144')
    parser.add_argument('-c', '--counting', action = 'store', default = None)
    parser.add_argument('-s', '--save', action = 'store_true')
    parser.add_argument('-n', '--number', action = 'store', default = '20')
    parser.add_argument('-p', '--predict', action = 'store_true')
    parser.add_argument('-m', '--model', action = 'store', default = None)
    parser.add_argument('-a', '--pca', action = 'store_true')
    parser.add_argument('-i', '--init', action = 'store_true')
    args = parser.parse_args()

    # ip = args.ip
    counting = args.counting
    save = args.save
    number = args.number
    predict = args.predict
    model_name = args.model
    PCA = args.pca
    initial = args.init

    initial_data_path = './initial_data.json'
    # with open(initial_data_path, "r") as json_file:
    #     initial_data = np.array(json.load(json_file))
    
    #start_distance = start_point * 2.5mm
    start_point = 160 # 400mm
    #end_distance = start_point * 2.5mm + num_points * 2.5mm 
    # num_points = (end_dis - start_point * 2.5) / 2.5 + 1
    num_points = 301 # 1,150mm
    # MAX = 122299391.16999999
    
    client = radar_raspi(start_point = start_point, num_points = num_points)
    print('clinet_start')
    # print(os.getcwd())
    cur_path = os.getcwd()
    scaler_path = os.path.join(cur_path, "preprocessed/scaler.pkl")
    scaler = pkl.load(open(scaler_path, "rb"))

    pca_path = os.path.join(cur_path, "preprocessed/pca.pkl")
    pca = pkl.load(open(pca_path, "rb"))

    minmax_path = os.path.join(cur_path, "preprocessed/minmax.pkl")
    minmax = pkl.load(open(minmax_path, "rb"))
    
    if save:
        # folder_path = './counting_cylinder/'
        folder_path = './counting_disk/'
        # folder_path = './counting_cylinder_demo/'

        file_path = folder_path + counting
        temp = list()
        for i in tqdm(range(int(number))):
            data = client.read_data()
            temp.append(data)
        save_data = np.array(temp)
        np.save(file_path, save_data)

    elif predict:
        mm = MinMaxScaler()
        import tensorflow as tf
        start = time.time()
        path = './model_weights/'
        model_path = os.path.join(path, model_name) + '.h5'
        # path = './model/'
        # model_path = os.path.join(path, model_name)
        dummy_data = client.read_data()
        dummy_data = np.expand_dims(client.read_data(), axis = 2)
        filter_in_list = [1, 32]
        filter_out_list = [32, 64]
        kernel_size = 3
        # predict_model = tf.keras.models.load_model(model_path)
        # predict_model = Resnet_pool(filter_in = filter_in_list, filters = filter_out_list, kernel_size = kernel_size, out_nums=17)
        predict_model = Resnet_pool(filter_in = filter_in_list, filters = filter_out_list, kernel_size = kernel_size, out_nums=6)
        predict_model.build(input_shape = (None, dummy_data.shape[1], dummy_data.shape[2]))
        
        # predict_model = basemodel(17)
               # predict_model.build(input_shape = (None, dummy_data.shape[1]))

        # predict_model.summary()
        predict_model.load_weights(model_path)
        end = time.time()
        print("Model Load Time : {:.4f}".format(end-start))
        # model = tf.model.load(model_path)
        # while True:
        featured_data = list()
        
        for i in range(10):
            data = client.read_data()
            featured_data.append(data)
        featured_data = np.array(featured_data)
        mean_data = np.mean(featured_data, axis = 0)
        # mean_data = np.subtract(mean_data, initial_data)
        data = mm.fit_transform(mean_data)
        # data = client.read_data()
        # mean_data = data

        print('data shape : ', data.shape)

        save_data_plot(data)
        data = np.expand_dims(data, axis = 2)
        start = time.time()
        inference = predict_model.predict(data)
        end = time.time()
        print("Model Inference Time : {:.4f}".format(end-start))
        # print(inference.shape)
        print(inference)
        print(np.argmax(inference))

    elif initial:
        data_list = list()
        for i in range(10):
            data = client.read_data()
            data_list.append(data)
        save_data = np.mean(np.array(data_list), axis = 0).tolist()
        # save_data = np.squeeze(save_data, axis = 0)
        with open(initial_data_path, "w") as json_file:
            json.dump(save_data, json_file)
        print('Save Initial Data')
        

    else:
        # pca = pkl.load(open('pca.pkl', 'rb'))
        mm = MinMaxScaler()
        with open(initial_data_path, "r") as json_file:
            initial_data = np.array(json.load(json_file))
        initial_data = initial_data.reshape(-1, 1)
        # print(initial_data.shape)
        while True:
            input('Press Enter to Measure')
            data = client.read_data()
            data = data.reshape(-1, 1)
            data = np.subtract(data, initial_data)
            # processed_data = data/MAX
            # processed_data = pca.transform(processed_data)
            processed_data = mm.fit_transform(data)
            processed_data = np.expand_dims(gaussian_filter1d(processed_data, sigma = 0.5), axis = 0)
            send_data = {
                'data_1' : processed_data.tolist()
            }
            send_data = json.dumps(send_data)
            URI = '/Mobius/radarEye1/sensor1/rawData'
            AE_ID = 'radarEye1'
            http_post_get.mobius_post(URI, AE_ID, AE_ID, send_data)
            print('pub')

    client.disconnect()

if __name__ == '__main__':
    main()