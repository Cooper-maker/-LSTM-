import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import numpy as np
from torch.autograd import Variable
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font",family='YouYuan')
os.chdir('D:\\桌面\\研究生课\\油田大数据和人工智能\\2_训练模型（回归）')

#df = pd.read_excel('data.xlsx')
df = pd.read_excel('油藏静态动态措施数据.xlsx')

df = df.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
vol = df['措施当年增油量']
vol_min=df['措施当年增油量'].min()
vol_max=df["措施当年增油量"].max()
del df['措施当年增油量']
total_len=df.shape[0]
X=[]
Y=[]
for i in range(df.shape[0]):
    X.append(np.array(df.iloc[i,].values,dtype=np.float32))
    Y.append(np.array(vol.iloc[i,],dtype=np.float32))

class Mydataset(Dataset):

    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)


# # 构建batch

trainx, trainy = X[:int(0.8 * total_len)],Y[:int(0.8 * total_len)]
testx, testy = X[int(0.8 * total_len):], Y[int(0.8 * total_len):]
'''
trainx, trainy = X , Y
testx, testy = X , Y
'''
train_loader = DataLoader(dataset=Mydataset(trainx, trainy, transform=None), batch_size=12,
                          shuffle=True)
test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=12, shuffle=True)


class lstm(nn.Module):

    def __init__(self, input_size=18, hidden_size=32, output_size=1):
        super(lstm, self).__init__()
        # lstm的输入 #batch,seq_len, input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = x.view(len(x), 1, -1)
        out, (hidden, cell) = self.rnn(x)  # x.shape : batch,seq_len,hidden_size , hn.shape and cn.shape : num_layes * direction_numbers,batch,hidden_size
        a, b, c = hidden.shape
        out = self.linear(hidden.reshape(a * b, c))
        return out


model = lstm()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

preds = []
labels = []
i_list = []
loss_list = []
for i in range(1000):
    total_loss = 0
    for idx, (data, label) in enumerate(train_loader):
        data1 = data.squeeze(1)
        pred = model(Variable(data1))
        label = label.unsqueeze(1)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if i % 100 == 99:
        print("第{}次迭代的损失函数值为{}".format(i + 1, total_loss))
    i_list.append(i)
    loss_list.append(total_loss)

print(total_loss)
plt.plot(i_list,loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

for idx, (x, label) in enumerate(train_loader):
    x = x.squeeze(1)  # batch_size,seq_len,input_size
    pred = model(x)
    preds.extend(pred.data.squeeze(1).tolist())
    labels.extend(label.tolist())
plt.plot([ele * (vol_max - vol_min) + vol_min for ele in preds], "r", label="pred")
plt.plot([ele * (vol_max - vol_min) + vol_min for ele in labels], "b", label="real")
plt.legend(['pred','real'])
plt.xlabel('样本数')
plt.ylabel('措施增产量（归一化结果）')
plt.show()
# 开始测试


preds = []
labels = []
for idx, (x, label) in enumerate(test_loader):
    x = x.squeeze(1)  # batch_size,seq_len,input_size
    pred = model(x)
    preds.extend(pred.data.squeeze(1).tolist())
    labels.extend(label.tolist())

plt.plot([ele * (vol_max - vol_min) + vol_min for ele in preds], "r", label="pred")
plt.plot([ele * (vol_max - vol_min) + vol_min for ele in labels], "b", label="real")
plt.legend(['pred','real'])
plt.xlabel('样本数')
plt.ylabel('措施增产量（归一化结果）')
plt.show()