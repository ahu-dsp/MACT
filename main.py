import os
import pickle as pkl
import torch
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim

from modelzuizhong import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from scipy.signal import savgol_filter
from eval  import *
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # 设置字体类型
#     "mathtext.fontset":'stix',
}
rcParams.update(config)
# device=torch.device("cpu")
np.random.seed(1234) #initialize random seed to generate same random values every time

def load_data_from_pfile(file_path):  # helper function
    with open(file_path, 'rb') as pfile:
        sample_data = pkl.load(pfile)
    return sample_data

pkzfiles_path = 'D:\论文和代码\lunwen\轴承寿命预测\RUL_Prediction-main\phm\phm\Learning_set' # Here is the training set


class PHMDataset(Dataset):
    '''
    PHM IEEE 2012 Data Challenge Training data set (6 different Mechanical Bearings data)
    '''

    def __init__(self, pfiles=[]):
        self.data = {'x': [], 'y': []}
        for pfile in pfiles:
            _data = load_data_from_pfile(pfile)
            self.data['x'].append(_data['x'])
            self.data['y'].append(_data['y'])
        self.data['x'] = np.concatenate(self.data['x'])
        self.data['y'] = np.concatenate(self.data['y'])[:, np.newaxis]

    def __len__(self):
        return self.data['x'].shape[0]

    def __getitem__(self, i):
        sample = {'x': torch.from_numpy(self.data['x'][i]), 'y': torch.from_numpy(self.data['y'][i])}
        return sample


train_pfiles = [pkzfiles_path+'\\bearing1_2_train_data_origin.pkz', pkzfiles_path+'\\bearing1_1_train_data_origin.pkz', \
                pkzfiles_path+'\\bearing1_4_train_data_origin.pkz',\
                pkzfiles_path+'\\bearing1_5_train_data_origin.pkz',\
                pkzfiles_path+'\\bearing1_7_train_data_origin.pkz', pkzfiles_path+'\\bearing1_6_train_data_origin.pkz']

val_pfiles = [pkzfiles_path+'\\bearing1_2_val_data_origin.pkz', pkzfiles_path+'\\bearing1_1_val_data_origin.pkz', \
               pkzfiles_path+'\\bearing1_4_val_data_origin.pkz',\
              pkzfiles_path+'\\bearing1_5_val_data_origin.pkz',\
              pkzfiles_path+'\\bearing1_7_val_data_origin.pkz', pkzfiles_path+'\\bearing1_6_val_data_origin.pkz']



train_dataset = PHMDataset(pfiles=train_pfiles)
val_dataset = PHMDataset(pfiles=val_pfiles)
# print(len(train_dataset), len(val_dataset))

train_batch_size = 256
val_batch_size = 256

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=1)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=1)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Declaring a variable "device" which will hold the device(i.e. either GPU or CPU) we are                                                                  #training the model on
# print(device)
device

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


model = Transformer(d_model=128,N=2,heads=8,dropout=0.3).to(device)
total = sum([param.nelement() for param in model.parameters()]) #计算总参数量....
print("Number of parameter: %.6f" % (total*4/1056748)) #输出

# criterion = RMSELoss()
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
optimizer = optim.NAdam(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
multistep_lr_sch = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1, last_epoch=-1, verbose=False)

def train_epoch(model, dataloader, criterion, optimizer):
    total_loss = 0
    num_of_samples = 0
    model.train()
    for t, batch in enumerate(dataloader):
        x = batch['x'].to(device, dtype=torch.float)
        y = batch['y'].to(device, dtype=torch.float)
        # print(y)

        optimizer.zero_grad()
        # y_prediction = model(x,t)
        y_prediction = model(x)
        loss = criterion(y_prediction, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.cpu().item()
        num_of_samples += x.shape[0]

        # print(total_loss/num_of_samples)
    return total_loss/num_of_samples

def eval(model, dataloader, criterion):
    total_loss = 0
    num_of_samples = 0
    model.eval()
    for t, batch in enumerate(dataloader):
        x = batch['x'].to(device, dtype=torch.float)
        y = batch['y'].to(device, dtype=torch.float)

        with torch.no_grad():
            # y_prediction = model(x,t)
            y_prediction = model(x)
            loss = criterion(y_prediction, y)

        total_loss += loss.cpu().item()
        num_of_samples += x.shape[0]


    return total_loss/num_of_samples



if __name__ == '__main__':
    max_epochs = 30
    loss_vals = []
    time_start = time.time()
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer)
        val_loss = eval(model, val_dataloader, criterion)
        multistep_lr_sch.step()
        loss_vals.append([train_loss, val_loss])
        print('{0}/{1}: train_loss = {2:.5f}, val_loss = {3:.5f}'.format(epoch+1, max_epochs, train_loss, val_loss) )

    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)

    plt.plot(range(max_epochs), [l[0] for l in loss_vals], 'b.-', label='train loss')
    plt.plot(range(max_epochs), [l[1] for l in loss_vals], 'r.-', label='val loss')
    plt.ylabel('Loss')  # x_label
    plt.xlabel('Epoch')  # y_label
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), 'MACT-test1-3.pth')

    model.load_state_dict(torch.load('MACT-test1-3.pth'))

    def model_inference_helper(model, dataloader):
        results = {'labels': [], 'predictions': []}
        total_loss = 0
        num_of_samples = 0
        model.eval()
        for t, batch in enumerate(dataloader):
            x = batch['x'].to(device, dtype=torch.float)
            y = batch['y'].to(device, dtype=torch.float)

            with torch.no_grad():
                # y_prediction = model(x,t)
                y_prediction = model(x)
                loss=criterion(y_prediction, y)

            total_loss += loss.cpu().item()
            # num_of_samples += x.shape[0]


            results['labels'] += y.squeeze().tolist()
            results['predictions'] += y_prediction.cpu().squeeze().tolist()

        rmse_loss = total_loss / t + 1
        return results,rmse_loss


    def sort_results(results):
        ind = [i[0] for i in sorted(enumerate(results['labels']), key=lambda x: x[1])]
        results['labels'] = [results['labels'][i] for i in ind]
        results['predictions'] = [results['predictions'][i] for i in ind]
        return results



    test_path = 'D:\论文和代码\lunwen\轴承寿命预测\RUL_Prediction-main\phm\phm\Full_Test_Set'  # Here is the path to the test set


    test_pfile = [test_path + '\\bearing1_3_test_data_quan_1d.pkz'] # Here is the test set
    test_dataset = PHMDataset(pfiles=test_pfile)
    # print(test_dataset.data['y'])

    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=1)

    results, rmse_loss = model_inference_helper(model, test_dataloader)

    predictions = np.array(results['predictions']).all()
    labels = np.array(results['labels']).all()
    # rmse_loss=nn.MSELoss(predictions, labels)

    x1 = range(len(results['predictions']))
    y1 = results['predictions']
    y2 = results['labels']
    # predict=np.array(y1)
    # label = np.array(y2)

    a1 = y1
    b1 = []
    for i in range(len(a1)):
        b1.append(np.array(a1[i]))
    c1 = np.array(b1)

    a2 = y2
    b2 = []
    for i in range(len(a2)):
        b2.append(np.array(a2[i]))
    c2 = np.array(b2)

    a11 = y1
    a22 = y2
    Aii=[]
    for i in range(len(a11)-1):
        b11=(np.array(a11[i]))
        b22=(np.array(a22[i]))
        Eri = ((b22 - b11) * 100 / b22)
        if Eri <= 0:
            Ai = np.exp(-np.log(0.5) * (Eri / 5))
        else:
            Ai = np.exp(np.log(0.5) * (Eri / 20))
        Aii.append(Ai)


    Score=np.mean(Aii)

    RMSE = np.sqrt(np.mean(np.square(c1 - c2)))
    MAE = np.mean(np.abs(c1 - c2))
    print(RMSE)
    print(MAE)
    print(Score)

    y_smooth = savgol_filter(y1, 73, 1)

    # X = np.arange(len(results['predictions'])).reshape(-1, 1)
    # y = np.array(results['predictions']).reshape(-1, 1)
    # reg = LinearRegression().fit(X, y)
    # X_test = np.linspace(0, 1803, 10000).reshape(-1, 1)
    # y_test = reg.predict(X_test)
    # plt.plot(X_test, y_test)

    # plt.plot(x1, y_smooth,label='Predicted HI')
    plt.plot(x1, y1, label='Predict')
    plt.plot(x1, y_smooth, label='Smooth')
    plt.plot(x1, y2, label='Actual', linewidth=2)

    plt.ylabel('RUL')  # x_label
    plt.xlabel('Time')  # y_label

    # plt.title('CWT-CNN-transformer-smooth bearing1-3')
    plt.title('Bearing1_3')
    plt.legend()


    plt.show()

