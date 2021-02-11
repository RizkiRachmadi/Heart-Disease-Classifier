# Final STEP V2.1 (Output Heart Disease = 1, No Heart Disease = 0 **CONVERT**)

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

print("===== Variable =====")

g, h = 6 + 1, 13 + 1;
w = [[0 for x in range(g)] for y in range(h)]
dw = [[0 for x in range(g)] for y in range(h)]

wh = {}
summation = {}
H = {}
Cout = {}
Cout_pred = {}
kelas = {}
kelas_target = {}

dH = {}
dwH = {}

C_true = 0
C_false = 0

# ===============
bias = 1
epoch = 6000
lr = 0.1
MSE = 0
losses = []

# IMPORT DATASET
dataset = pd.read_csv('C:\\Users\\IF\\Desktop\\heart.csv')
dataset

#  Input Data
print("Input Dataset")

train_data = dataset.iloc[:, 0:13]
print(train_data)
print("-" * 100)

a, b = 13 + 1, 270 + 1;
Input = [[0 for x in range(a)] for y in range(b)]

for j in range(270):
    for k in range(13):
        Input[j + 1][k + 1] = train_data.iloc[j, k]
        #Convert Input

        if (Input[j + 1][k + 1] >= 100):
            Input[j + 1][k + 1] = Input[j + 1][k + 1] / 100
        elif (Input[j + 1][k + 1] >= 10):
            Input[j + 1][k + 1] = Input[j + 1][k + 1] / 10

    print("%.2f " * 13 % (
    Input[j][1], Input[j][2], Input[j][3], Input[j][4], Input[j][5],
    Input[j][6], Input[j][7], Input[j][4], Input[j][9],
    Input[j][10], Input[j][11], Input[j][12], Input[j][13]))

# Output Data
print("Kelas Data (Target)")
Output = dataset.iloc[:, -1]
print(Output)
print("-" * 100)

g, h = 13 + 1, 270 + 1;
target = {}

for i in range(270):
    #Convert Output
    if (Output[i] == 2):
        Output[i] = 1
    elif (Output[i] == 1):
        Output[i] = 0
    target[i + 1] = Output[i]
    print("Target[%d] = " % (i + 1), target[i + 1])
print("-" * 100, i)


# Creates weight input layer
print("======= Pembangkitan Weight Untuk Input Layer ======= ")

for i in range(13 + 1):
    for j in range(1, 6 + 1):
        w[i][j] = random.random()
        w[0][j] = 1;
        print("w[%d][%d] = " % (i, j), w[i][j])
print("-" * 100)

# Creates weight hidden layer
print("====== Pembangkitan Weight Untuk Hidden Layer =======")

for i in range(6 + 1):
    wh[i] = random.random()
    wh[0] = 1;
    print("wh[%d] = " % i, wh[i])
print("-" * 100)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Learning
print("==========================================================> Learning")

for i in range(0, epoch):
    for j in range(1, 270 + 1):  # 270 = Jumlah dataset (270 input)
        # Forward

        # Input Layer
        for k in range(1, 6 + 1):  # 1,3 = 2 - Jumlah hidden layer
            H[k] = bias * w[0][k]
            for n in range(1, 13 + 1):  # 1,13+1 = 13 (Jumlah Input)
                H[k] = H[k] + (Input[j][n] * w[n][k])
            #Sigmoid Output
            H[k] = sigmoid(H[k])

        #Hidden Layer 1
        SH = bias * wh[0]
        for k in range(1, 6 + 1):
            SH = SH + (H[k] * wh[k])
        #Sigmoid Output
        C = sigmoid(SH)
        #Learning Performance Analysis
        MSE = MSE + (target[j] - C) ** 2

        # Backward
        # Differensial
        dC = sigmoid_derivative(C) * (target[j] - C)  # Output = TargetC
        for k in range(1, 6 + 1):
            dH[k] = sigmoid_derivative(H[k]) * wh[k] * dC

        #Update Weight Hidden Layer
        H[0] = bias
        for k in range(6 + 1):  # 3 = Jumlah Hidden layer (Bias(H0), H1 dan H2)
            dwH[k] = lr * H[k] * dC
            # Update Weight
            wh[k] = wh[k] + dwH[k]

        #Update Weight Input Layer
        for k in range(1, 13 + 1):  # 1, 13+1 = 13 (Jumlah Input)
            for m in range(1, 6 + 1):  # 1,3 = 2 (Jumlah Hidden Layer)
                dw[k][m] = lr * Input[j][k] * dH[m]
                # Update Weight
                w[k][m] = w[k][m] + dw[k][m]

    #Learning Performance Analysis
    MSE = MSE / 270  # 270 = jumlah baris data
    losses.append(MSE)
    print("%.12f => MSE %d " % (MSE, i))
    # print(MSE)
    MSE = 0

plt.figure()
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("MSE")
plt.title("Error Lost")
plt.show()


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Final Weight
print()
print("==========Bobot Akhir Hasil Learning============")
for i in range(13 + 1):
    for j in range(1, 6 + 1):
        print("w[%d][%d] = " % (i, j), w[i][j])
print("-" * 100)

for i in range(6 + 1):
    print("wh[%d] = " % i, wh[i])
print("-" * 100)

#Testing

print()
print("=" * 100)

print("============ Pengujian Dengan Input Seluruh Dataset =============")
out = dataset.iloc[:, -1]
g, h = 13 + 1, 270 + 1;
kelas_target = {}
for i in range(270):
    if (out[i] == 1):
        out[i] = 2
    elif (out[i] == 0):
        out[i] = 1
    kelas_target[i + 1] = out[i]

for j in range(1, 270 + 1):  # 270 = Jumlah dataset (270 input)
    for k in range(1, 6 + 1):  # 1,3 = 2 - Jumlah hidden layer
        H[k] = bias * w[0][k]
        for n in range(1, 13 + 1):  # 1, 13+1 = 13 (Jumlah Input)
            H[k] = H[k] + (Input[j][n] * w[n][k])
        #Sigmoid Input
        H[k] = sigmoid(H[k])
    #Hidden Layer 1
    SH = bias * wh[0]
    for k in range(1, 6 + 1):
        SH = SH + (H[k] * wh[k])
    # Sigmoid Output
    Cout[j] = sigmoid(SH)
    #Output Threshold
    if (Cout[j] <= 0.9):
        Cout_pred[j] = 0
        kelas[j] = 1
    else:
        Cout_pred[j] = 1
        kelas[j] = 2
#Final Result
print()

print("----------------------------------------------------------------")
print("||    Output    || Kelas Hasil Prediksi || Kelas Sesungguhnya ||")
print("----------------------------------------------------------------")
for j in range(1, 270 + 1):  # 270 = Jumlah dataset (270 input)
    print("||  %.4f  ||" %(Cout[j]), "         %d          " %(kelas[j]),
          "||         %d" %(kelas_target[j]), "         ||")
    # Calculate Error (Positive/Negative)
    if (Cout_pred[j] == target[j]):
        C_true += 1
    else:
        C_false += 1
#Nilai error
Error = C_false / (C_true + C_false)*100

print()
print("Hasil Evaluasi Testing")
print("Jumlah Klasifikasi Data Benar = %d" %C_true)
print("Jumlah Klasifikasi Data Salah = %d" %C_false)
print("Prosentase Error = %.2f" %Error, "%")
