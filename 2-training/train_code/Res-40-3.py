#!/usr/bin/env python
# coding: utf-8

# In[1]:

from utils_2 import *


# In[2]:


# 给定测试集患者id----需手动
test_id = list(range(285,293))
test_id.extend(list(range(419,439)))
test_id.extend(list(range(1,84)))
test_id.extend(list(range(439,473)))
test_id.extend([259,294])
print(len(test_id))


# In[3]:


model_file_current='model-cut2/'  ## 存模型训练结果的路径
# 任务阶段
model_stage = 'task3'
# 使用模型：1为Inception，2为vgg，3为resnet
model_name  = 3


## 根据模型调整存储名
model_name_list = ['Inception', 'Vgg', 'Resnet']
# 目标视野
Size_list = [[80, 60, 40], [100, 80, 60], [60, 40, 20]]
## 根据模型调整视野大小
Size = Size_list[model_name-1]
## Inception可以用30，Vgg、Resnet用15
Batchsize_list = [30, 15, 15]
### 根据模型调整batchsize
Batchsize = Batchsize_list[model_name-1]
Shuffle = True

# 模型参数
dropout_rate = 0.5
weight_decay = 1e-4
learning_patience = 10
early_patience = 60
Epoch_num = 200


print('Stage: ', model_stage)
print('Model: ', model_name_list[model_name-1])
print('View: ',Size)
print('Shuffle:',Shuffle)

# In[4]:


# 分类类别
Class_num = get_task(model_stage)
# data文件所在位置前缀，默认为空，若非空，以/结尾
#data_file = '/course75/RealData/'
data_file = '/home/'
# csv文件所在位置
datflag_file = 'datflag636.csv'
# Parameters
params = {'dim': (Size[1],Size[1],Size[1]),
          'batch_size': Batchsize,
          'n_classes': Class_num,
          'n_channels': 3,
          'shuffle': Shuffle,
          'Size': [Size[0],Size[1],Size[2]],
          'data_file': data_file}
# 测试集batchsize需设置为1
params_val = {'dim': (Size[1],Size[1],Size[1]),
          'batch_size': 1,
          'n_classes': Class_num,
          'n_channels': 3,
          'shuffle': False,
          'Size': [Size[0],Size[1],Size[2]],
          'data_file': data_file}


# In[5]:


partition, labels=get_par_lab(datflag_file=datflag_file, data_file=data_file, stage = model_stage, test_id=test_id)
print('train length: ',len(partition['train']))
print('test length: ',len(partition['test']))
print('label length: ',len(labels))



# In[7]:


# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['test'], labels, **params_val)


### 设置各种优化器，和相应的学习率策略
### 当我们使用calllback定义的学习率衰减策略时，optimizer中的学习率衰减策略就会被忽视

## 以epoch为参数进行学习率调整，每10个epoch学习率降低为之前的一半
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

### 指数学习率衰减
def exp_decay(epoch):
    initial_lrate = 0.1
    k = 0.1
    lrate = initial_lrate * math.exp(-k*epoch)
    return lrate

## 使用优化器自带的学习率
adagrad = Adagrad(learning_rate=0.01, epsilon=1e-08, decay=0.0)
adadelta = Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
rmsprop = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(learning_rate=0.1, decay=0.001, momentum=0.9, nesterov=True)


# In[9]:


### 设置回调函数（保存h5文件，设置学习率策略、早停等）
### 保存最佳模型
csv_logger = CSVLogger(model_file_current+model_stage+'/'+model_name_list[model_name-1]+str(Size[1])+'_training.log',append=True)
filepath = model_file_current+model_stage+'/'+model_name_list[model_name-1]+'_'+str(Size[1])+'_Epoch.{epoch:02d}-{val_accuracy:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=early_patience, verbose=1, mode='max')
#change_lr = LearningRateScheduler(exp_decay)
#change_lr = LearningRateScheduler(step_decay)
## 动态学习率调整，当10个epoch训练后val_accuracy还没有提升，学习率降低为原来的1/5
change_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,verbose=1, mode='max',patience=learning_patience, min_lr=0)
callback = [checkpoint,change_lr,csv_logger]


# In[10]:


model = get_model(model_name,Size[1],weight_decay,dropout_rate,Class_num)
model.summary()


# In[11]:


model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate = 0.001),metrics=['accuracy'])
model.fit(training_generator,validation_data=validation_generator,
                    epochs=Epoch_num,callbacks=callback,
                    use_multiprocessing=True,workers=12,
                    verbose = 2)




