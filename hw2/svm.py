#!/usr/bin/python
import pandas as pd
import numpy as np
import random

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
feats_train = feats[:nTrain]
feats_val = feats[nTrain:]
labels_train = labels[:nTrain]
labels_val = labels[nTrain:]

classes = [1004, 1100, 1103, 1200, 1400]

N = len(feats_train[0])
print('feature length N = %d' % N)

UPDATE_STEP = 50

def train_svm(Ttrain, weights, cls_ind):
    cls = classes[cls_ind]
    sample_size = 1000
    W = np.zeros(shape=(Ttrain+1, N), dtype=np.float)
    W[0] = weights
    theta = np.zeros(shape=(1, N), dtype=np.float)
    for t in range(0, Ttrain):
        # recieve sample from the environment
        sample = random.sample(range(0, nTrain), sample_size)
        y_d = np.array(labels_train)[sample].reshape(1, sample_size)
        x_d = np.array(feats_train)[sample]
        # hinge condition indicator
        yx_d = y_d @ x_d
        hinge_cond = np.multiply(yx_d, W[t]) < 1.0
        # subgradient
        sub_gradient = yx_d * hinge_cond
        # dual parameter update
        theta += sub_gradient
        # mirror projection
        W[t+1] = theta / (t+1)
        
        if (t + 1) % UPDATE_STEP == 0:
            print(f"[{t+1}/{Ttrain}] steps completed")
        
    final_weights = np.sum(W, axis=0) / Ttrain
    return final_weights


def eval_svm(Tval, final_weights, cls_ind):
    cls = classes[cls_ind]
    nRight = 0
    for t in range(Tval):
        # write ~10 lines here
        pass
    print('got %d/%d right on class %d' % (nRight, Tval, cls))
    return float(nRight)/Tval


def eval_multi_svm(Tval, weights0, weights1, weights2, weights3, weights4):
    nRight = 0
    for t in range(Tval):
        # write ~10 lines here
        pass
    print('got %d/%d right on multiclass' % (nRight, Tval))
    return float(nRight)/Tval


Ttrain = 1000
weights_init = np.random.uniform(size=(1, N))
# subgrad is 0 if y_m w^\top x_m \geq 1; -y_m x_m otherwise

cls_ind = 0
Tval = 1000
cls0_weights = train_svm(Ttrain, weights_init, 0)
# cls0_acc = eval_svm(Tval, cls0_weights, 0)
# cls1_weights = train_svm(Ttrain, weights_init, 1)
# cls1_acc = eval_svm(Tval, cls1_weights, 1)
# cls2_weights = train_svm(Ttrain, weights_init, 2)
# cls2_acc = eval_svm(Tval, cls2_weights, 2)
# cls3_weights = train_svm(Ttrain, weights_init, 3)
# cls3_acc = eval_svm(Tval, cls3_weights, 3)
# cls4_weights = train_svm(Ttrain, weights_init, 4)
# cls4_acc = eval_svm(Tval, cls4_weights, 4)

# _ = eval_multi_svm(Tval, cls0_weights, cls1_weights, cls2_weights, cls3_weights, cls4_weights)
