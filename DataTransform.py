import numpy as np

# 旋转90度
def Rotate1(Xtrain, Ytrain, uprate=1.5):

    bn = Xtrain.shape[0]
    uprate_bn = int(bn * (uprate - 1))
    shuffle_idx = np.array(range(0, bn))
    np.random.shuffle(shuffle_idx)
    Ytrain = Ytrain[:, np.newaxis]
    Xtrain_trans = np.zeros([uprate_bn,
                             1,
                             Xtrain.shape[2],
                             Xtrain.shape[3],
                             Xtrain.shape[4]]).astype('float64')
    Ytrain_trans = np.zeros([uprate_bn, 1]).astype('float64')
    for i in range(uprate_bn):
        curr_idx = shuffle_idx[i]
        Xtrain_trans[i] = np.rot90(Xtrain[curr_idx], k=1, axes=(2, 3))
        Ytrain_trans[i] = Ytrain[curr_idx]

    Ytrain = Ytrain.astype('float64')
    Xtrain = np.concatenate((Xtrain, Xtrain_trans), 0)
    Ytrain = np.concatenate((Ytrain, Ytrain_trans), 0)

    return Xtrain, Ytrain[:, 0]


# 旋转180度
def Rotate2(Xtrain, Ytrain, uprate=1.5):

    bn = Xtrain.shape[0]
    uprate_bn = int(bn * (uprate - 1))
    shuffle_idx = np.array(range(0, bn))
    np.random.shuffle(shuffle_idx)
    Ytrain = Ytrain[:, np.newaxis]
    Xtrain_trans = np.zeros([uprate_bn,
                             1,
                             Xtrain.shape[2],
                             Xtrain.shape[3],
                             Xtrain.shape[4]]).astype('float64')
    Ytrain_trans = np.zeros([uprate_bn, 1]).astype('float64')
    for i in range(uprate_bn):
        curr_idx = shuffle_idx[i]
        Xtrain_trans[i] = np.rot90(Xtrain[curr_idx], k=1, axes=(1, 2))
        Ytrain_trans[i] = Ytrain[curr_idx]

    Ytrain = Ytrain.astype('float64')
    Xtrain = np.concatenate((Xtrain, Xtrain_trans), 0)
    Ytrain = np.concatenate((Ytrain, Ytrain_trans), 0)

    return Xtrain, Ytrain[:, 0]


# 翻转
def Flip(Xtrain, Ytrain, uprate=1.5):
    bn = Xtrain.shape[0]
    uprate_bn = int(bn * (uprate - 1))
    shuffle_idx = np.array(range(0, bn))
    np.random.shuffle(shuffle_idx)
    Ytrain = Ytrain[:, np.newaxis]
    Xtrain_trans = np.zeros([uprate_bn,
                             1,
                             Xtrain.shape[2],
                             Xtrain.shape[3],
                             Xtrain.shape[4]]).astype('float64')
    Ytrain_trans = np.zeros([uprate_bn, 1]).astype('float64')
    for i in range(uprate_bn):

        curr_idx = shuffle_idx[i]
        tmp = Xtrain[curr_idx]
        tmp = tmp[np.newaxis, :, :, :, :]
        for j in range(32):
            Xtrain_trans[i, 0, j, :, :] = tmp[0, 0, 31 - j, :, :]
        Ytrain_trans[i] = Ytrain[curr_idx]

    Ytrain = Ytrain.astype('float64')
    Xtrain = np.concatenate((Xtrain, Xtrain_trans), 0)
    Ytrain = np.concatenate((Ytrain, Ytrain_trans), 0)

    return Xtrain, Ytrain[:, 0]