#!/usr/bin/python
import pandas as pd
import numpy as np
import random
import time

path1 = 'data/oakland_part3_am_rf.node_features'
path2 = 'data/oakland_part3_an_rf.node_features'
f1 = open(path1, 'r')
lines1 = f1.readlines()
f1.close()
f2 = open(path2, 'r')
lines2 = f2.readlines()
f2.close()

xyzs = []
labels = []
feats = []
for i, line in enumerate(lines1):
    if i > 2:  # first three lines are dead
        words = line.split()
        xyz = [float(w) for w in words[:3]]
        xyzs.append(xyz)
        label = int(words[4])
        labels.append(label)
        feat = [float(w) for w in words[5:]]
        feats.append(feat)
for i, line in enumerate(lines2):
    if i > 2:  # first three lines are dead
        words = line.split()
        xyz = [float(w) for w in words[:3]]
        xyzs.append(xyz)
        label = int(words[4])
        labels.append(label)
        feat = [float(w) for w in words[5:]]
        feats.append(feat)

print('read all of the data')

everything = list(zip(xyzs, labels, feats))
random.shuffle(everything)

xyzs, labels, feats = zip(*everything)

print('shuffled')

nTotal = len(xyzs)
nTrain = int(0.8*nTotal)
nVal = nTotal-nTrain
print('picked %d for train, %d for val, out of %d total' %
      (nTrain, nVal, nTotal))

xyzs_train = xyzs[:nTrain]
xyzs_val = xyzs[nTrain:]
feats_train = np.asarray(feats[:nTrain])
feats_val = np.asarray(feats[nTrain:])
labels_train = np.asarray(labels[:nTrain])
labels_val = np.asarray(labels[nTrain:])

classes = [1004, 1100, 1103, 1200, 1400]

N = len(feats_train[0])
print('feature length N = %d' % N)

def train_svm(Ttrain, weights, cls_ind):
    cls = classes[cls_ind]
    lamb = 5.0
    W = np.zeros(shape=(Ttrain+1, N), dtype=np.float)
    W[0] = weights
    theta = np.zeros(shape=(1, N), dtype=np.float)
    
    sample = random.sample(range(0, nTrain), Ttrain)
    for t in range(0, Ttrain):
        
        idx = sample[t]
        
        # recieve sample from the environment
        y_d = labels_train[idx]
        x_d = feats_train[idx]
        
        # hinge condition indicator
        y_d = 1 if y_d == cls else -1
        wx_d = np.dot(W[t], x_d)
        
        hinge_cond = y_d * wx_d < 1.0
        
        if hinge_cond:
            # subgradient
            sub_gradient = y_d * x_d.T 
    
            # dual parameter update
            theta += sub_gradient
        
        # mirror projection
        W[t+1] = theta / (lamb * (t+1))
        
        # if (t + 1) % UPDATE_STEP == 0:
        #     print(f"\t[{t+1}/{Ttrain}] steps completed")
        
    final_weights = np.mean(W, axis=0)
    return final_weights


def eval_svm(Tval, final_weights, cls_ind):
    cls = classes[cls_ind]
    nRight = 0
    
    # sample indeces from val batch
    sample = random.sample(range(0, len(labels_val)), Tval)
    
    for t in range(Tval):
        idx = sample[t]
        
        x = feats_val[idx]
        y = labels_val[idx]
        
        y_true = 1 if y == cls else -1
        
        y_pred = 1 if np.dot(x, final_weights.T) >= 0 else -1
        
        if y_pred == y_true:
            nRight += 1
        
    print('got %d/%d  ({%f}) right on class %d' % (nRight, Tval, float(nRight/Tval), cls))
    return float(nRight)/Tval


def eval_multi_svm(Tval, weights0, weights1, weights2, weights3, weights4):
    nRight = 0
    
    # sample indeces from val batch
    sample = random.sample(range(0, len(labels_val)), Tval)
    
    pred = np.zeros(shape=(len(classes), 1), dtype=np.float)
    for t in range(Tval):
        idx = sample[t]
        
        x = feats_val[idx]
        y = labels_val[idx]
        
        pred[0] = np.dot(x, weights0.T)
        pred[1] = np.dot(x, weights1.T)
        pred[2] = np.dot(x, weights2.T)
        pred[3] = np.dot(x, weights3.T)
        pred[4] = np.dot(x, weights4.T)
        
        i = np.argmax(pred)
        if classes[i] == y:
            nRight += 1
        
    print('got %d/%d ({%f}) right on multiclass' % (nRight, Tval,  float(nRight/Tval)))
    return float(nRight)/Tval


Ttrain = 4000
weights_init = np.random.uniform(size=(1, N))
# subgrad is 0 if y_m w^\top x_m \geq 1; -y_m x_m otherwise

cls_ind = 0
Tval = len(labels_val)

start = time.time()
cls0_weights = train_svm(Ttrain, weights_init, 0)
cls0_acc = eval_svm(Tval, cls0_weights, 0)
print(f"\ttraining and prediction single class takes: {time.time() - start}")

cls1_weights = train_svm(Ttrain, weights_init, 1)
cls1_acc = eval_svm(Tval, cls1_weights, 1)

cls2_weights = train_svm(Ttrain, weights_init, 2)
cls2_acc = eval_svm(Tval, cls2_weights, 2)

cls3_weights = train_svm(Ttrain, weights_init, 3)
cls3_acc = eval_svm(Tval, cls3_weights, 3)

cls4_weights = train_svm(Ttrain, weights_init, 4)
cls4_acc = eval_svm(Tval, cls4_weights, 4)

_ = eval_multi_svm(Tval, 
                   cls0_weights, 
                   cls1_weights, 
                   cls2_weights, 
                   cls3_weights, 
                   cls4_weights)
print(f"\ttraining and prediction single class takes: {time.time() - start}")