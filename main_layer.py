import torch
from torchvision.transforms import transforms as T
from torch.optim import lr_scheduler
import argparse #argparse模块的作用是用于解析命令行参数，例如python parseTest.py input.txt --port=8080
import unet_4_layer as unet_4
from torch import optim
from dataset import CellDataset
from torch.utils.data import DataLoader
import cv2
import numpy
import torchvision
import pandas as pd
from matplotlib import pyplot as plt
import time
import random
# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#p1 = random.randint(0,1)

#torch.manual_seed(1)
x_transform = T.Compose([
    #T.RandomVerticalFlip(p1),
    T.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    T.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])#torchvision.transforms.Normalize(mean, std, inplace=False)
])

# mask只需要转换为tensor
torch.manual_seed(1)
y_transform = T.Compose([
    #T.RandomVerticalFlip(p1),
    T.ToTensor()
])



def train_model(model,criterion,optimizer,scheduler,dataload,num_epochs=5):
    global epoch,accuracy_points,val_accuracy_points,epoch_points,complete_time,train_loss
    accuracy_points=[]
    val_accuracy_points=[]
    epoch_points=[]
    complete_time=[]
    train_loss=[]
    
    since=time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        epoch_loss = 0
        sum_accuracy=0
        step = 0 #minibatch数
        
        for x, y in dataload:
            optimizer.zero_grad()#每次minibatch都要将梯度(dw,db,...)清零
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs)
            img_y=outputs.detach().squeeze().numpy()
            ret,mask=cv2.threshold(img_y,0.5,1,cv2.THRESH_BINARY)
            target=labels.detach().squeeze().numpy()
            match=mask==target
            unmatch=mask!=target
            accuracy=numpy.sum(match)/(numpy.sum(match)+numpy.sum(unmatch))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()#更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += loss.item()
            sum_accuracy+=accuracy
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))
            print("%d/%d,accuracy:%0.3f" % (step, dataset_size // dataload.batch_size, accuracy))
        scheduler.step()
        torch.save(model.state_dict(),'weight_%d.pth' % epoch)# 返回模型的所有内容
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        print("epoch %d accuracy:%0.3f" % (epoch, sum_accuracy/(dataset_size // dataload.batch_size)))
        train_loss.append(round(epoch_loss,3))
        accuracy_points.append(round(sum_accuracy/(dataset_size // dataload.batch_size),3))
        
        val()
        
        val_accuracy_points.append(round(val_accuracy_point,3))
        epoch_points.append(epoch)
        #print(accuracy_points,val_accuracy_points,epoch_points)
        time_epoch=time.time()-since
        complete_time.append(round(time_epoch,3))
        print(train_loss)
        print(val_accuracy_points)

    x1=numpy.array(epoch_points)
    y1=numpy.array(accuracy_points)
    x2=numpy.array(epoch_points)
    y2=numpy.array(val_accuracy_points)
    plt.subplot(1,2,1)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('epoch_select')
    plt.plot(x1,y1,x2,y2)

    m=numpy.array(complete_time)
    n=numpy.array(val_accuracy_points)
    plt.subplot(1,2,2)
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.title('batchsize_select')
    plt.plot(m,n)

    #plt.suptitle('lr='+str(learning_rate)+'---'+'lambda='+str(lamb))

    #plt.savefig('result_lr0.0001_lamb0.2'+'.png')

    plt.show()
    return model

def val():
    model = unet_4.UNet(4,1)
    model.load_state_dict(torch.load('weight_%d.pth' % epoch,map_location='cpu'))
    '''for name, parameters in model.state_dict().items():
        print(name,':',parameters.detach().numpy())'''
    val_sum_accuracy=0
    val_cell_dataset = CellDataset("/Users/pierce/Desktop/summer internship/learning/starch/val", transform=x_transform, target_transform=y_transform)
    val_dataloaders = DataLoader(val_cell_dataset,batch_size=args.batch_size)
    val_dataset_size = len(val_dataloaders.dataset)
    model.eval()
    with torch.no_grad():
        for x, target in val_dataloaders:
            val_y=model(x)
            val_img_y=torch.squeeze(val_y).numpy()
            val_target=torch.squeeze(target).numpy()
            ret,mask=cv2.threshold(val_img_y,0.5,1,cv2.THRESH_BINARY)
            val_match=mask==val_target
            val_unmatch=mask!=val_target
            val_accuracy=numpy.sum(val_match)/(numpy.sum(val_match)+numpy.sum(val_unmatch))
            val_sum_accuracy+=val_accuracy
            global val_accuracy_point
            val_accuracy_point=val_sum_accuracy/(val_dataset_size // val_dataloaders.batch_size)




#训练模型
def train():
    #global learning_rate,lamb,num_epochs,times
    #for para_lr in range(3):
        #learning_rate=round(random.uniform(0.0001,0.01),4)
        #for para_lamb in range(6):
            #lamb=round(random.uniform(0.005,0.1),4)
            #print('--------This is',str(times),'model:lr=',str(learning_rate),'lambda='+str(lamb)+'--------\n')
    model = unet_4.UNet(4,1).to(device)
    batch_size = args.batch_size
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.01, weight_decay=0)#model.parameters():Returns an iterator over module parameters
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.2)
    cell_dataset = CellDataset("/Users/pierce/Desktop/summer internship/learning/starch/train", transform=x_transform, target_transform=y_transform)
    dataloader = DataLoader(cell_dataset, batch_size=batch_size, shuffle=True,num_workers=3)
    train_model(model,criterion,optimizer,scheduler,dataloader)

#测试
def test():
    global dict,index,k
    dict={}
    index=[]
    k=1
    model = unet_4.UNet(4,1)

    '''ckp=torch.load('weight_1.pth',map_location=torch.device('cpu'))
    parameters=[]
    for i in ckp.keys():
        parameters.append(i)'''

    model.load_state_dict(torch.load(args.ckp,map_location='cpu'))

    cell_dataset = CellDataset("/Users/pierce/Desktop/summer internship/learning/starch/test", transform=x_transform, target_transform=y_transform)
    dataloaders = DataLoader(cell_dataset)#batch_size默认为1
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x, target in dataloaders:
            y=model(x)
            img_y=torch.squeeze(y).numpy()
            _,out=cv2.threshold(img_y,0.5,255,cv2.THRESH_BINARY)

            '''target=torch.squeeze(target).numpy()
            ret,mask=cv2.threshold(img_y,0.5,1,cv2.THRESH_BINARY)
            TP = numpy.logical_and(target, mask)
            union = numpy.logical_or(target, mask)
            iou_score = numpy.sum(TP) / numpy.sum(union)
            FN=target!=TP
            FP=mask!=TP
            kernel=numpy.ones((6,6),numpy.uint8)
            erosion_out=cv2.erode(out,kernel)
            erosion_target=cv2.erode(target,kernel)
            erosion_out = erosion_out.astype(numpy.uint8)
            erosion_target = erosion_target.astype(numpy.uint8)
            num_labels_out, labels, stats, centroids = cv2.connectedComponentsWithStats(erosion_out,connectivity=4)  
            num_labels_target, labels, stats, centroids = cv2.connectedComponentsWithStats(erosion_target,connectivity=4)          
            print('Detected Counts:',num_labels_out)
            print('Original Counts:',num_labels_target)
            print('IoU:',iou_score)
            print('Precision:',numpy.sum(TP)/(numpy.sum(TP)+numpy.sum(FP)))
            print('Recall:',numpy.sum(TP)/(numpy.sum(TP)+numpy.sum(FN)))
            print('F1-Score:',numpy.sum(TP)*2/(numpy.sum(TP)*2+numpy.sum(FP)+numpy.sum(FN)))
            
            #dict.setdefault('Detected Counts',[]).append(num_labels_out)
            #dict.setdefault('Original Counts',[]).append(num_labels_target)
            dict.setdefault('IoU',[]).append(iou_score)
            dict.setdefault('F1-Score',[]).append(numpy.sum(TP)*2/(numpy.sum(TP)*2+numpy.sum(FP)+numpy.sum(FN)))
            dict.setdefault('Precision',[]).append(numpy.sum(TP)/(numpy.sum(TP)+numpy.sum(FP)))
            dict.setdefault('Recall',[]).append(numpy.sum(TP)/(numpy.sum(TP)+numpy.sum(FN)))
            index.append('seg_'+str(k))'''
            plt.imshow(out)
            plt.pause(0.1)
            cv2.imwrite('seg_'+str(k)+'.png',out)
            #torchvision.utils.save_image(y, 'seg_'+str(k)+'.png')
            k=k+1
        plt.show()
    '''dataframe=pd.DataFrame(dict,index=index)
    print(dataframe)
    dataframe.to_csv('dataframe.csv',index=True,header=True)'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser() #创建一个ArgumentParser对象
    parser.add_argument('action', type=str, help='train or test')#添加参数
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--ckp', type=str, help='the path of the mode weight file')
    args = parser.parse_args()
    
    if args.action == 'train':
        train()
    elif args.action == 'test':
        test()