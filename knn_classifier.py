import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import pickle
import sys
import matplotlib.pyplot as plt
import itertools
from keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras.utils import to_categorical




global folder,imageDict,accuracy,best_k

if len(sys.argv) > 1:
    folder = sys.argv[1] # hhd_dataset
else:
    folder = "hhd_dataset"
imageDict = {}
splitDataDict = {}
accuracy = -1
best_k = -1

def loadModel():
    return  pickle.load(open('Best_KNN_Model', 'rb'))

#help function for ploting confusion matrix
def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def calculateAccPerClass(y_pred,test_y):
    y_pred = y_pred.tolist()
    accuracyDict = {}
    for i in range(0, 27):
        accuracyDict[i] = [0, 0]
    for index in range(0, len(test_y)):
        if y_pred[index] == test_y[index]:
            accuracyDict[test_y[index]][0] += 1
            accuracyDict[test_y[index]][1] += 1
        else:
            accuracyDict[test_y[index]][1] += 1
    for key, value in accuracyDict.items():
        accuracyDict[key] = float("{0:.3f}".format(value[0]/value[1]))
    return accuracyDict

def writeResults(results,reg0,reg1):
    with open('results_NN.txt', 'w') as f:
        f.write("Final model configuration:")
        f.write("    Dropout:")
        f.write(str(reg0))
        f.write("    Regulizer:")
        reg1 = str(reg1)
        reg1 = reg1.split("_")
        reg1 = reg1[0] + "=" + reg1[1] + "." + reg1[2]

        f.write(str(reg1))
        f.write('\n')
        f.write('\n')
        f.write("Letter    Accuracy")
        f.write('\n')
        sum = 0
        for key in results.keys():
            sum += results[key]
            f.write('  ')
            f.write(str(key))
            f.write('        ')
            f.write(str(results[key]))
            f.write('\n')
        f.write('\n')
        f.write('-----------------------')
        f.write('\n')
        f.write(" Avg        ")
        f.write(str(float("{0:.3f}".format(sum/len(results.keys())))))


def plot_confusion_matrix(cm,target_names,title='Confusion matrix',  cmap=None,normalize=True):
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    accuracy = float("{:.3f}".format(accuracy))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(16, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('int')


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.3f}; misclass={:0.3f}'.format(accuracy, misclass))
    plt.show()
    cm = trunc(cm,decs=3)

    pd.DataFrame(cm).to_csv('confusion_matrix_NN.csv')



def loadData():
    '''
    splitDataDict['train_x'] = train_x
    splitDataDict['test_x'] = test_x
    splitDataDict['val_x'] = val_x

    splitDataDict['train_y'] = train_y
    splitDataDict['test_y'] = test_y
    splitDataDict['val_y'] = val_y
    '''
    return pickle.load(open('data_resized_padded_greyscale', 'rb'))



def negative():
    global splitDataDict
    keys = ['train_x','test_x','val_x']
    for key in keys:
        for i in range(0,len(splitDataDict[key])):
            image = cv2.bitwise_not(splitDataDict[key][i])
            splitDataDict[key][i] = image



def createModel(dropout=None,reg=None,size=None,classes_num=None):

    model = Sequential()

    model.add(Dense(1024, activation='relu', input_shape=(size,),kernel_regularizer=reg))

    if (dropout != None):
        model.add(Dropout(dropout))

    model.add(Dense(512,kernel_regularizer=reg))
    if(dropout != None):
        model.add(Dropout(dropout))

    model.add(Dense(512,kernel_regularizer=reg))
    if (dropout != None):
        model.add(Dropout(dropout))

    model.add(Dense(classes_num, activation='softmax'))

    model.summary()

    return model

def saveBestNNModel(model_and_parameters):
    with open('model_and_parameters', 'wb') as f:
        pickle.dump(model_and_parameters, f)

import inspect
def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]

def trainBestModel():

    global splitDataDict
    L1_0_01 = regularizers.l1(0.01)
    L2_0_01 = regularizers.l2(0.01)
    L1_0_001 = regularizers.l1(0.001)
    L2_0_001 =  regularizers.l2(0.001)

    dropout = 0.5
    epoch = 50
    batch = 256

    x = np.array([np.array(xi) for xi in splitDataDict['train_x']])
    y = np.array([np.array(xi) for xi in splitDataDict['train_y']])

    val_x = np.array([np.array(xi) for xi in splitDataDict['val_x']])
    val_y = np.array([np.array(xi) for xi in splitDataDict['val_y']])

    test_x = np.array([np.array(xi) for xi in splitDataDict['test_x']])
    test_y = np.array([np.array(xi) for xi in splitDataDict['test_y']])

    x = x.reshape(x.shape[0], x.shape[1])
    val_x = val_x.reshape(val_x.shape[0], val_x.shape[1])
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1])

    input_size = np.prod(x.shape[1:])
    classes_num = len(np.unique(y))

    x = x.astype('float32')
    val_x = val_x.astype('float32')
    test_x = test_x.astype('float32')

    #x /= 255
    #val_x /= 255
    #test_x /= 255

    y = to_categorical(y)
    val_y = to_categorical(val_y)
    test_y = to_categorical(test_y)

    highest_acc = -1

    AllRegs = [[None, None], [None, L1_0_01], [None, L1_0_001], [None, L2_0_01], [None, L2_0_001], [dropout, None],
                    [dropout, L2_0_01], [dropout, L2_0_001]]

    for regu in AllRegs:

        model = createModel(regu[0], regu[1],input_size,classes_num)
        model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        history = model.fit(x, y, validation_data=(val_x, val_y), epochs=epoch, batch_size=batch)
        [val_loss, val_acc] = model.evaluate(val_x, val_y)
        print(val_acc,highest_acc)
        if val_acc > highest_acc:
            highest_acc = val_acc
            model_and_parameters = [model,history,val_loss,val_acc,regu[0],retrieve_name(regu[1]),test_x,test_y]
            saveBestNNModel(model_and_parameters)


def loss_curve(history):
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curve', fontsize=16)
    plt.savefig("training & validation loss.jpg")
    plt.show()

def loadbestModel():
    #[model, history, val_loss, val_acc, regu[0], regu[1], test_x, test_y]
    return  pickle.load(open('model_and_parameters', 'rb'))

def eval_on_test_set(best_model):
    best_model_NN = best_model[0]
    history = best_model[1]
    val_loss = best_model[2]
    val_acc = best_model[3]
    regu0 = best_model[4]
    regu1 =best_model[5]
    test_x = best_model[6]
    test_y = best_model[7]

    loss_curve(history)
    print("Model Accuracy:",float("{:.3f}".format(val_acc)))
    test_y = np.argmax(test_y, axis=1)

    predictions = best_model_NN.predict(test_x, batch_size=1)
    predictions = np.argmax(predictions, axis=1)

    dictPerClass = calculateAccPerClass(predictions,test_y)
    writeResults(dictPerClass,regu0,regu1)

    cm = confusion_matrix(y_pred=predictions, y_true=test_y)
    plot_confusion_matrix(cm,normalize=True,target_names=[i for i in range(0,27)])



# loading the greyscaled,padded,resized and splited dataset into dict.
#splitDataDict = loadData()
#negative()
#trainBestModel()

'''
best_model = [model,history,val_loss,val_acc,regu[0],regu[1],test_x,test_y]
regu[0] = dropout
regu[1] = L1/L2 00.1 or 0.001

'''
best_model = loadbestModel()
eval_on_test_set(best_model)

