#!/usr/bin/env python
# coding: utf-8

# 划分训练、测试集函数

import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import random
from tensorflow import keras
from scipy.ndimage.interpolation import zoom
from random import shuffle
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers,Model
from tensorflow.keras.layers import add, AveragePooling3D, Input,Conv3D,concatenate,MaxPooling3D, UpSampling3D, Activation, BatchNormalization,Dropout,Flatten,Dense,GlobalAveragePooling3D
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adagrad,Adadelta
K.set_image_data_format('channels_first')

# 给定分类类别函数
def get_task(stage='task3'):
    if stage == 'task3':
        Class_num = 3
    else:
        Class_num = 2
    return Class_num

# 生成数据集
# 根据病人ID划分训练、测试集
# 给定test_id, Null_id 和 train_id
# 注意data_file为data文件夹的前缀，以/
def get_par_lab(datflag_file, data_file, stage, train_id=False, test_id=[], Null_id=[]):
    print('Current task stage: '+str(stage))
    # 首先读取结节文件
    datflag_0 = pd.read_csv(datflag_file)
    datflag_0['subset'] = None
    datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in test_id, axis=1),'subset'] = 'test'
    # 若指定train_id，则划分train数据集，否则除Null_id外全划为train
    if train_id:
        datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in train_id, axis=1),'subset'] = 'train'
    else:
        datflag_0.loc[datflag_0['subset'].isnull(),'subset'] = 'train'
    # 排除个别数据集
    datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in Null_id, axis=1),'subset'] = None
    
    ## 根据任务阶段给定x，y
    if stage=='task1':
        # task1选取全部数据
        datflag = pd.DataFrame(datflag_0)
        # 良性为0，其余为1
        datflag.loc[:, 'label'] = 1
        datflag.loc[datflag['flag']=='良性','label'] = 0
    elif stage=='task2':
        # task2排除良性数据
        datflag = pd.DataFrame(datflag_0.loc[datflag_0['flag']!='良性',])
        # 浸润前病变为0，其余为1
        datflag.loc[:, 'label'] = 1
        datflag.loc[datflag['flag']=='浸润前病变','label'] = 0
    elif stage=='task3':
        # task3仅保留grade1-3数据
        datflag = pd.DataFrame(datflag_0.loc[datflag_0.apply(lambda x: x['flag'] in ['grade1','grade2','grade3'], axis=1),])
        # grade1 = 0，grade2 = 1, grade3 = 2
        datflag.loc[:, 'label'] = 0
        datflag.loc[datflag['flag']=='grade2','label'] = 1
        datflag.loc[datflag['flag']=='grade3','label'] = 2
    ## 生成训练集、测试集，根据csv数据的index对应回存储的数据
    datflag_train = datflag.loc[datflag['subset']=='train',]
    train_index = list(datflag_train.index)
    #train_flag = list(datflag_train['label'])
    datflag_test = datflag.loc[datflag['subset']=='test',]
    test_index = list(datflag_test.index)
    ## 根据train_index生成训练集路径
    train_file_list=[]
    train_flag_list=[]
    agm_flist=glob.glob(data_file+'data/agm/*')  
    for f_name in agm_flist:
        # 提取当前文件名
        c_name = f_name.split('/')[-1]
        # 提取当前结节对应index
        c_index = int(c_name.split('_')[0])
        # 若对应index存在于训练集中，则纳入train_file_list，同时记录对应flag
        if c_index in train_index:
            train_file_list.append('agm/'+c_name)
            train_flag_list.append(datflag_train.loc[c_index,'label'])
    ## 根据test_index生成训练集路径
    test_file_list=[]
    test_flag_list=[]
    org_flist=glob.glob(data_file+'data/org/*')  
    for f_name in org_flist:
        # 提取当前文件名
        c_name = f_name.split('/')[-1]
        # 提取当前结节对应index
        c_index = int(c_name.split('.')[0])
        # 若对应index存在于训练集中，则纳入train_file_list，同时记录对应flag
        if c_index in test_index:
            test_file_list.append('org/'+c_name)
            test_flag_list.append(datflag_test.loc[c_index,'label'])
    partition = {'train': train_file_list, 'test': test_file_list}
    labels = dict(zip((train_file_list+test_file_list),(train_flag_list+test_flag_list)))
    return partition, labels


# def get_train_test_ID(datflag_file, data_file, stage, train_id=False, test_id=[], Null_id=[]):
#     print('Current task stage: '+str(stage))
#     # 首先读取结节文件
#     datflag_0 = pd.read_csv(datflag_file)
#     datflag_0['subset'] = None
#     datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in test_id, axis=1),'subset'] = 'test'
#     # 若指定train_id，则划分train数据集，否则除Null_id外全划为train
#     if train_id:
#         datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in train_id, axis=1),'subset'] = 'train'
#     else:
#         datflag_0.loc[datflag_0['subset'].isnull(),'subset'] = 'train'
#     # 排除个别数据集
#     datflag_0.loc[datflag_0.apply(lambda x: x['ID'] in Null_id, axis=1),'subset'] = None
    
#     ## 根据任务阶段给定x，y
#     if stage=='task1':
#         # task1选取全部数据
#         datflag = pd.DataFrame(datflag_0)
#         # 良性为0，其余为1
#         datflag.loc[:, 'label'] = 1
#         datflag.loc[datflag['flag']=='良性','label'] = 0
#     elif stage=='task2':
#         # task2排除良性数据
#         datflag = pd.DataFrame(datflag_0.loc[datflag_0['flag']!='良性',])
#         # 浸润前病变为0，其余为1
#         datflag.loc[:, 'label'] = 1
#         datflag.loc[datflag['flag']=='浸润前病变','label'] = 0
#     elif stage=='task3':
#         # task3仅保留grade1-3数据
#         datflag = pd.DataFrame(datflag_0.loc[datflag_0.apply(lambda x: x['flag'] in ['grade1','grade2','grade3'], axis=1),])
#         # grade1 = 0，grade2 = 1, grade3 = 2
#         datflag.loc[:, 'label'] = 0
#         datflag.loc[datflag['flag']=='grade2','label'] = 1
#         datflag.loc[datflag['flag']=='grade3','label'] = 2
#     ## 生成训练集、测试集，根据csv数据的index对应回存储的数据
#     datflag_train = datflag.loc[datflag['subset']=='train',]
#     train_ID = list(set(datflag_train['ID']))
#     #train_flag = list(datflag_train['label'])
#     datflag_test = datflag.loc[datflag['subset']=='test',]
#     test_ID = list(set(datflag_test['ID']))
#     return train_ID, test_ID

# 数据生成器函数

class DataGenerator(keras.utils.Sequence):
    # 加入参数data_file=''，作为data文件夹的前缀，以/结尾
    # 'Generates data for Keras'
    # 默认batchsize设为15，dim=80*80*80，n_channels=3，n_classes=2
    # volume: 结节的最大边长，目前为100（100*100*100）
    def __init__(self, list_IDs, labels, data_file, batch_size=15, dim=(80,80,80), n_channels=3,
                 n_classes=3, shuffle=True, Size=[100,80,60], volume=100):
        # 'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.Size = Size
        self.data_file = data_file
        self.volume = volume
        self.on_epoch_end()
 
    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
 
    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
 
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
 
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
 
        return X, y
 
    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __changeshape(self, imgs, new_shape):  # 插值改变数组形状，使得大小统一
        if len(imgs.shape)==3:   # 如果是3维的话：
            resize_factor = [a/b for a,b in zip(new_shape,imgs.shape)]
            imgnew = zoom(imgs, resize_factor, mode = 'nearest')   # 放缩，边缘使用最近邻，插值默认为三线性插值
            return imgnew
        else:
            raise ValueError('wrong shape')  # 本代码只能处理3维数据
    
    def __data_generation(self, list_IDs_temp):
        # 主要修改了data_generation的部分
        # 'Generates data containing batch_size samples' # X : (n_samples, n_channels, *dim)
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
        volume = self.volume
        Size_1 = self.Size[0]
        Size_2 = self.Size[1]
        Size_3 = self.Size[2]
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # 读取原始数据
            temp_x = np.load(self.data_file+'data/' + ID, mmap_mode='r' )
            # 3视角数据处理
            X[i,0] = self.__changeshape(temp_x[round(volume/2-Size_1/2):round(volume/2+Size_1/2),
                              round(volume/2-Size_1/2):round(volume/2+Size_1/2),
                              round(volume/2-Size_1/2):round(volume/2+Size_1/2)], new_shape=[Size_2,Size_2,Size_2])
            X[i,1] = self.__changeshape(temp_x[round(volume/2-Size_2/2):round(volume/2+Size_2/2),
                              round(volume/2-Size_2/2):round(volume/2+Size_2/2),
                              round(volume/2-Size_2/2):round(volume/2+Size_2/2)],
                              new_shape=[Size_2,Size_2,Size_2])
            X[i,2] = self.__changeshape(temp_x[round(volume/2-Size_3/2):round(volume/2+Size_3/2),
                              round(volume/2-Size_3/2):round(volume/2+Size_3/2),
                              round(volume/2-Size_3/2):round(volume/2+Size_3/2)], 
                              new_shape=[Size_2,Size_2,Size_2])
            # Store class
            y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def get_model(model_name,Input_size,weight_decay,dropout_rate,Class_num):
    if model_name == 1:
        return build_model_Incpt(Input_size,weight_decay,dropout_rate,Class_num)
    elif model_name == 2:
        return build_model_Vgg(Input_size,weight_decay,dropout_rate,Class_num)
    elif model_name == 3:
        return build_model_Res(Input_size,weight_decay,dropout_rate,Class_num)

def build_model_Incpt(Input_size,weight_decay,dropout_rate,Class_num):
    input_layer = Input([3,Input_size,Input_size,Input_size]) 
    x = input_layer

    #第一层:使用7x7x7的卷积核
    x = Conv3D(64,(7,7,7),strides=(2,2,2),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x) 
    x = BatchNormalization(axis=1)(x) 
    x = MaxPooling3D(pool_size=(3,3,3),strides=(2,2,2),padding='same')(x) 

    #第二层:使用3x3x3的卷积核
    x = Conv3D(192,(3,3,3),strides=(1,1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x) 
    x = BatchNormalization(axis=1)(x) 
    x = MaxPooling3D(pool_size=(3,3,3),strides=(2,2,2),padding='same')(x) 

    for i in range(9):

        branch1x1 = Conv3D(64,(1,1,1),strides=(1,1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x) 
        branch1x1 = BatchNormalization(axis=1)(branch1x1) 

        branch3x3 = Conv3D(96,(1,1,1),strides=(1,1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x) 
        branch3x3 = BatchNormalization(axis=1)(branch3x3) 
        branch3x3 = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(branch3x3) 
        branch3x3 = BatchNormalization(axis=1)(branch3x3) 

        branch5x5 = Conv3D(16,(1,1,1),strides=(1,1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x) 
        branch5x5 = BatchNormalization(axis=1)(branch5x5) 
        branch5x5 = Conv3D(32,(5,5,5),strides=(1,1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(branch5x5)
        branch5x5 = BatchNormalization(axis=1)(branch5x5) 

        branchpool = MaxPooling3D(pool_size=(3,3,3),strides=(1,1,1),padding='same')(x)
        branchpool = Conv3D(32,(1,1,1),strides=(1,1,1),padding='same',activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(branchpool)
        branchpool = BatchNormalization(axis=1)(branchpool) 
        x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=1)
        x = MaxPooling3D(pool_size=(3,3,3),strides=(2,2,2),padding='same')(x)

    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)
    x = Dense(Class_num,activation='softmax')(x) # 分为两类 # 注意不同的分类任务要修改这里的数字
    output_layer=x
    model_Inception=Model(input_layer,output_layer)
    return model_Inception    

def build_model_Vgg(Input_size,weight_decay,dropout_rate,Class_num):
    input_layer = Input([3,Input_size,Input_size,Input_size]) 
    x = input_layer
    # 1st Convolution Block
    x = BatchNormalization(axis=1)(x)
    x = Conv3D(64, [3, 3, 3], padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Conv3D(64, [3, 3, 3], padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = MaxPooling3D((2, 2, 2))(x)

    # 2nd Convolution Block
    x = BatchNormalization(axis=1)(x)
    x = Conv3D(128, [3, 3, 3], padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Conv3D(128, [3, 3, 3], padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = MaxPooling3D((2, 2, 2))(x)

    # 3rd Convolution Block
    x = BatchNormalization(axis=1)(x)
    x = Conv3D(256, [3, 3, 3], padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Conv3D(256, [3, 3, 3], padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Conv3D(256, [3, 3, 3], padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = MaxPooling3D((2, 2, 2))(x)

    # 4th Convolution Block
    x = BatchNormalization(axis=1)(x)
    x = Conv3D(512, [3, 3, 3], padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Conv3D(512, [3, 3, 3], padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Conv3D(512, [3, 3, 3], padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = MaxPooling3D((2, 2, 2))(x)

    # 5th Convolution Block
    x = BatchNormalization(axis=1)(x)
    x = Conv3D(512, [3, 3, 3], padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Conv3D(512, [3, 3, 3], padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Conv3D(512, [3, 3, 3], padding='same', activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = MaxPooling3D((2, 2, 2))(x)

    # FC Block
    x = Dropout(dropout_rate)(x)
    #x = Flatten()(x)
    x = GlobalAveragePooling3D()(x)
    x = Dense(Class_num, activation = "softmax")(x)
    output_layer = x
    model_vgg16 = Model(input_layer, output_layer)
    return model_vgg16   

def build_model_Res(Input_size,weight_decay,dropout_rate,Class_num):
    input_layer = Input([3,Input_size,Input_size,Input_size]) 
    x = BatchNormalization()(input_layer)
    #卷积层：使用64个7*7*7的卷积核，步长为2
    x = Conv3D(64, (7,7,7), padding='same', strides=(2,2,2), activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(input_layer)
    #最大值池化
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    # 把输入存到另外的变量中
    x0 = x

    #用循环搭建网络。
    #每个卷积核大小都为3*3*3，num_list存放每一步需要的卷积核个数。
    num_list=[64,64,128,128,256,256,512,512]
    for i in range(8):
        # 一个block
        # 判断，在某些步骤步长需要为2
        x = BatchNormalization(axis=1)(x)
        if (i==2|i==4|i==6):
            x = Conv3D(num_list[i], (3,3,3), padding='same', strides=(2,2,2), activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
        else: 
            x = Conv3D(num_list[i], (3,3,3), padding='same', strides=(1,1,1), activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x)
        # 一个卷积层加一个batch normalization
        x = BatchNormalization(axis=1)(x)
        # 卷积，但不进行激活
        x = Conv3D(num_list[i], (3,3,3), padding='same', strides=(1,1,1), activation=None,kernel_regularizer=regularizers.l2(weight_decay))(x)
         # 一个卷积层加一个batch normalization
        x = BatchNormalization(axis=1)(x)

        # 下面两步转换输入数据的通道数，用来让x0和x维数相同，可以进行加法计算
        # 同样在某些步骤步长需要为2
        x0 = BatchNormalization(axis=1)(x0)
        if (i==2|i==4|i==6):   
            x0 = Conv3D(num_list[i],(1,1,1),padding='same',strides=(2,2,2),activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x0)
        else:
            x0 = Conv3D(num_list[i],(1,1,1),padding='same',strides=(1,1,1),activation='relu',kernel_regularizer=regularizers.l2(weight_decay))(x0)
        x0 = BatchNormalization(axis=1)(x0)
        # add把输入的x和经过一个block之后输出的结果加在一起
        x = add([x,x0])
        # 求和之后的结果再做一次relu
        x = Activation('relu')(x)
        # 把输入存到一个另外的变量中
        x0 = x 
    # Average池化    
    x = AveragePooling3D(pool_size= 512, strides=(1, 1, 1), padding='same')(x)
    model = Model(inputs=input_layer,outputs=x)
    # 添加全连接层
    x = model.output
    #x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = GlobalAveragePooling3D()(x)
    #Dense
    predictions = Dense(Class_num,activation='softmax')(x)
    model_res = Model(inputs=model.input,outputs=predictions)
    return model_res   






