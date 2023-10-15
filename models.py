import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow import keras

class basemodel(Model):
    def __init__(self, num_classes = 5):
        super(basemodel, self).__init__()
        # self.nl1 = layers.experimental.preprocessing.Normalization(axis = -1)
        # self.nl1 = layers.Normalization(axis = -1)
        self.fc1 = layers.Dense(units = 8, activation = 'relu')
        self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(units = 16, activation = 'relu')
        self.fc3 = layers.Dense(units = 32, activation = 'relu')
        self.fc4 = layers.Dense(units = 64, activation = 'relu')
        self.out = layers.Dense(units = num_classes, activation = 'softmax')
    
    def call(self, x, training = False, mask = None):
        # out = self.nl1(x)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.out(out)
        return out

class basemodel_regression(Model):
    def __init__(self):
        super(basemodel_regression, self).__init__()
        # self.nl1 = layers.experimental.preprocessing.Normalization(axis = -1)
        # self.nl1 = layers.Normalization(axis = -1)
        self.fc1 = layers.Dense(units = 8, activation = keras.layers.LeakyReLU(alpha=0.01))
        self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(units = 16, activation = keras.layers.LeakyReLU(alpha=0.01))
        self.fc3 = layers.Dense(units = 32, activation = keras.layers.LeakyReLU(alpha=0.01))
        self.fc4 = layers.Dense(units = 64, activation = keras.layers.LeakyReLU(alpha=0.01))
        self.out = layers.Dense(units = 1)
    
    def call(self, x, training = False, mask = None):
        # out = self.nl1(x)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.out(out)
        return out

class non_norm(Model):
    def __init__(self, num_classes = 5):
        super(non_norm, self).__init__()
        # self.nl1 = layers.Normalization(axis = -1)
        self.fc1 = layers.Dense(units = 8, activation = 'relu')
        self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(units = 16, activation = 'relu')
        self.fc3 = layers.Dense(units = 32, activation = 'relu')
        self.out = layers.Dense(units = num_classes, activation = 'softmax')
    
    def call(self, x, training = False, mask = None):
        # out = self.nl1(x)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.out(out)
        return out

class acconeer(Model):
    def __init__(self):
        super(acconeer, self).__init__()
        self.fc1 = layers.Dense(units = 24, activation = 'relu')
        self.fc2 = layers.Dense(units = 12, activation = 'relu')
        self.out = layers.Dense(units = 5, activation = 'softmax')
    def call(self, x , training = False, mask = None):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.out(out)
        return out

def create_model():
    model = tf.keras.models.Sequential([
        layers.Normalization(axis = -1),
        layers.Dense(units = 10, activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dense(units=5, activation = 'relu'),
        layers.Dense(units = 1, activation = 'relu')
        ])
    return model


class ResidualUnit(tf.keras.Model):
    def __init__(self, filter_in, filter_out, kernel_size):
        super(ResidualUnit, self).__init__()

        self.bn1 = keras.layers.BatchNormalization()
        self.conv1 = keras.layers.Conv1D(filter_out, kernel_size, padding='same')
        self.bn2 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv1D(filter_out, kernel_size, padding='same')

        if filter_in == filter_out:
            self.identity = lambda x: x
        else:
            self.identity = keras.layers.Conv1D(filter_out, 1, padding = 'same')
        
    def call(self, x, training=False, mask = None):
        h = self.bn1(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv1(h)

        h = self.bn2(h, training=training)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        return self.identity(x) + h

class ResnetLayer(tf.keras.Model):
    def __init__(self, filter_in, filters, kernel_size):
        super(ResnetLayer, self).__init__()
        self.sequence = list()

        for f_in, f_out in zip([filter_in] + list(filters), filters):
            self.sequence.append(ResidualUnit(f_in, f_out, kernel_size))

    def call(self, x, training=False , mask=None):
        for unit in self.sequence:
            x = unit(x, training=training)
        return x
    
class Resnet(tf.keras.Model):
    def __init__(self, filter_in, filters, kernel_size, out_nums=17):
        super(Resnet, self).__init__()
        # self.nl = layers.Normalization(axis = -1)
        self.resnet_layer = ResnetLayer(filter_in, filters, kernel_size)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units = 128, activation = 'relu')
        self.fc2 = layers.Dense(units = 64, activation = 'relu')
        self.fc3 = layers.Dense(units = 32, activation = 'relu')
        self.fc4 = layers.Dense(units = out_nums, activation = 'softmax')
    
    def call(self, x, training=False, mask=None):
        # x = self.nl(x)
        x = self.resnet_layer(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    
class Resnet_pool(tf.keras.Model):
    def __init__(self, filter_in, filters, kernel_size, out_nums=17):
        super(Resnet_pool, self).__init__()
        # self.nl = layers.Normalization(axis = -1)
        self.resnet_layer = ResnetLayer(filter_in, filters, kernel_size)
        self.maxpool = layers.MaxPool1D(pool_size=2, strides=2, padding='valid')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units = 128, activation = 'relu')
        self.fc2 = layers.Dense(units = 64, activation = 'relu')
        self.fc3 = layers.Dense(units = 32, activation = 'relu')
        self.fc4 = layers.Dense(units = out_nums, activation = 'softmax')
    
    def call(self, x, training=False, mask=None):
        # x = self.nl(x)
        x = self.resnet_layer(x, training=training)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class Resnet_regression(tf.keras.Model):
    def __init__(self, filter_in, filters, kernel_size, out_nums=17):
        super(Resnet_regression, self).__init__()
        self.nl = layers.Normalization(axis = -1)
        self.resnet_layer = ResnetLayer(filter_in, filters, kernel_size)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(units = 128, activation = 'relu')
        self.fc2 = layers.Dense(units = 64, activation = 'relu')
        self.fc3 = layers.Dense(units = 32, activation = 'relu')
        self.fc4 = layers.Dense(units = 1)
    
    def call(self, x, training=False, mask=None):
        x = self.nl(x)
        x = self.resnet_layer(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
    

def build_resnet(filter_in_list, filter_out_list, kernel_size, out_nums=17):
    # 입력 레이어
    input_layer = layers.Input(shape=(301, 1))  # 입력 데이터의 shape에 따라 수정해야 합니다.

    # Normalization 레이어
    # nl = layers.BatchNormalization(axis=-1)(input_layer)

    # Resnet 레이어 생성
    # x = nl
    x = input_layer
    for filter_in, filter_out in zip(filter_in_list, filter_out_list):
        x = ResidualUnit(filter_in, filter_out, kernel_size)(x)

    # Flatten 레이어
    x = layers.Flatten()(x)

    # Fully Connected 레이어
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Dense(units=32, activation='relu')(x)
    
    # 출력 레이어
    output_layer = layers.Dense(units=out_nums, activation='softmax')(x)

    # Functional 모델 생성
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    return model