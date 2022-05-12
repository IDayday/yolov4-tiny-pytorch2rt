# -*- coding: utf-8
#-------------------------------------#
#     训练
#-------------------------------------#
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import os
from collections import OrderedDict
from collections import namedtuple
from itertools import product
import torch
import torchvision
import pandas as pd
import json
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.yolo4_tiny import YoloBody
from nets.yolo_training import YOLOLoss
from utils.dataloader import YoloDataset, yolo_dataset_collate

# 自带的 torch.utils.tensorboard 有bug
from tensorboardX import SummaryWriter

#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# epoch内各种训练、验证、tensorboard记录、日志记录、模型保存的完整定义        
def fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda,writer,run_data,run):
    global train_tensorboard_step, val_tensorboard_step
    total_loss = 0
    val_loss = 0
    epoch_start_time = time.time()
    # 每个epoch的训练阶段
    net.train()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            #------------------------------------------------------#
            # 开启tensorboard记录
            # 以下记录的是图片数据，以缩略图展示
            # 记录图片数据会大幅增加训练时间
            #------------------------------------------------------#
            # grid = torchvision.utils.make_grid(images)
            # writer.add_image('images',grid)            
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = net(images)
            losses = []
            num_pos_all = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for i in range(2):
                loss_item, num_pos = yolo_losses(outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos_all
            
            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            #------------------------------------------------------#
            # 开启tensorboard记录
            # add_scalar给tensorboard添加标量数据，对应'名称'，'Y轴','X轴'
            # 以下记录train_loss 和 total_loss
            #------------------------------------------------------#    
            writer.add_scalar('Train_loss', loss, train_tensorboard_step)
            train_tensorboard_step += 1

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    # 每个epoch的验证阶段
    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}' ,postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                for i in range(2):
                    loss_item, num_pos = yolo_losses(outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos

                loss = sum(losses) / num_pos_all
                val_loss += loss.item()

                # 将loss写入tensorboard, 下面注释的是每一步都写
                writer.add_scalar('Val_loss', loss, val_tensorboard_step)
                val_tensorboard_step += 1

                pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
                pbar.update(1)
    epoch_duration = time.time() - epoch_start_time
    print("Training time {}".format(epoch_duration))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    #--------------------------------------------------#
    # 训练日志设置(不是tensorboard，另存为文件)
    #--------------------------------------------------#
    results = OrderedDict()
    results['epoch'] = epoch
    results['loss'] = loss.item()
    results['epoch duration'] = epoch_duration
    for k,v in run._asdict().items():
        results[k] = v
    run_data.append(results)

    # # 将loss写入tensorboard，每个世代保存一次
    # writer.add_scalar('Val_loss',val_loss / (epoch_size_val+1), epoch)

    #------------------------------------------------------#
    # 以下记录网络权重及权重梯度
    #------------------------------------------------------#    
    for name, param in net.named_parameters():
        writer.add_histogram(name, param, epoch)
        if run.freezetrain == True:
            continue # 结束本轮循环，重新进入主循环
        else:
            writer.add_histogram(f'{name}.grad', param.grad, epoch)
    # 需要将此tensorboard记录关闭
    writer.close()
   
    #--------------------------------------------------#
    # 保存模型权重以及训练日志（csv）
    #--------------------------------------------------#    
    print('Saving state, iter:', str(epoch+1))
    torch.save(net, 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pt'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    pd.DataFrame.from_dict(run_data,orient='columns').to_csv('xinjiang.csv')
    with open('xinjiang.json','w', encoding='utf-8') as f:
        json.dump(run, f, ensure_ascii=False, indent=4)


# 调参库
class RunBuilder():
    # @staticmethod目的是获得静态方法并且不需要实例化
    @staticmethod
    # 返回的是一个list，里面包含着重组后的具名元组
    def get_runs(params):
        # Run是一个具名元组的方法，会将传入的参数依次对应到设置的名称下。
        Run = namedtuple('run',params.keys())
        runs = []
        # 参数重组
        # *表示自动对应多个参数
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs

params = OrderedDict(
    lr = [.001],
    gpus = ['0,1'],
    val_split = [0.1],
    stratify = [False],  # 是否采取分层抽样，因为考虑到数据不均衡以及长尾分布的特点
    freezetrain = [True],
    pretrained = [True],
    optim = ['Adam'],
    mosaic = [False],  # 开启mosaic数据增强会延长训练时间
    cosine_lr = [False],
    smooth_label = [0]
)

if __name__ == "__main__":
    #-------------------------------#
    #   所使用的主干特征提取网络
    #   mobilenetv1
    #   mobilenetv2
    #   mobilenetv3
    #   mini
    #-------------------------------#
    #-------------------------------#
    #   是否使用Cuda
    #-------------------------------#
    Cuda = False
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True
    #------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    #------------------------------------------------------#
    normalize = False
    #-------------------------------#
    #   输入的shape大小
    #-------------------------------#
    input_shape = (608,608)
    #----------------------------------------------------#
    #   classes和anchor的路径，训练前修改classes_path，使其对应自己的数据集
    #----------------------------------------------------#
    anchors_path = 'model_data/yolo_anchors.txt'
    classes_path = 'model_data/xj_classes.txt'   
    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)

    run_data = []
    

    # 训练参数循环
    for run in RunBuilder.get_runs(params):
        # run依次获得list中的各个具名元组，所以可以将名称作为属性直接调出例如run.batch_size
        print(run)
        #----------------------------------------------------#
        #   获得图片路径和标签
        #----------------------------------------------------#
        annotation_path = '2007_train.txt'
        #----------------------------------------------------------------------#
        #   验证集的划分在train.py代码里面进行
        #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
        #   当前划分方式下，验证集和训练集的比例为1:9
        #----------------------------------------------------------------------#
        val_split = run.val_split
        with open(annotation_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines)*val_split)
        num_train = len(lines) - num_val

        #------------------------------------------------------#
        #   创建yolo模型
        #   训练前要修改classes_path和对应的txt文件
        #------------------------------------------------------#
        model = YoloBody(len(anchors[0]), num_classes)
        #------------------------------------------------------#
        #   权值文件
        #------------------------------------------------------#
        os.environ["CUDA_VISIBLE_DEVICES"] = run.gpus
        if run.pretrained == True:
            model_path = "model_data/yolov4_tiny_weights_coco.pth"
            #   加快模型训练的效率
            print('Loading weights into state dict...')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #   不载入预训练权重时，将对应语句注释掉
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_path, map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print('Finished!')
        else:
            print('Train from the beginning')
        #------------------------------------------------------#
        #   Yolov4的tricks应用
        #   mosaic 马赛克数据增强 True or False 
        #   Cosine_scheduler 余弦退火学习率 True or False
        #   label_smoothing 标签平滑一般 0.01以下 如 0.01、0.005
        #------------------------------------------------------#
        mosaic = run.mosaic
        cosine_lr = run.cosine_lr
        smooth_label = run.smooth_label

        net = model.train()
        # 是否开启多GPU训练
        if Cuda:
            net = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            net = net.cuda()
            print('DataParallel already!')
        

        # 统计可学习的模型参数量
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('The number of parameters of model is', num_params)

        # 建立loss函数
        yolo_losses = YOLOLoss(np.reshape(anchors,[-1,2]),num_classes, (input_shape[1], input_shape[0]), smooth_label, Cuda, normalize)
        #------------------------------------------------------#
        # 开启tensorboard记录
        # f''格式化字符串，{}中表示被替换的内容
        # SummaryWriter是给运行日志命名
        # 以下记录的是网络结构，并以图片尺寸作为输入
        # 在模型载入后，再写入tensorboard
        #------------------------------------------------------#
        writer = SummaryWriter(comment=f'-{run.gpus}',flush_secs=60)
        # 这里在Cuda条件下，加了  .cuda()   反而报错
        if Cuda:
            graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],input_shape[1])).type(torch.FloatTensor).cuda()
        else:
            graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],input_shape[1])).type(torch.FloatTensor)
        # writer.add_graph(model, graph_inputs)

        train_tensorboard_step = 1
        val_tensorboard_step = 1
        # 冻结backbone方式训练（先冻结，再解冻）
        if run.freezetrain == True:
            lr = run.lr
            Batch_size = 16
            Init_Epoch = 0
            Freeze_Epoch = 40
            
            #----------------------------------------------------------------------------#
            #   我在实际测试时，发现optimizer.Adam的weight_decay起到了反作用，
            #   optimizer.SGD 模型无法收敛
            #   所以去除掉了weight_decay，也可以打开试试，一般是weight_decay=5e-4
            #   一般选择这两种优化器
            #----------------------------------------------------------------------------#
            if run.optim == 'SGD':
                optimizer = optim.SGD(net.parameters(),lr, momentum=0.9, weight_decay=1e-4)
            else:
                optimizer = optim.Adam(net.parameters(),lr)
            # 是否采用余弦退火学习率           
            if cosine_lr:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.6)


            if Use_Data_Loader:
                train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
                val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
                gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                        drop_last=True, collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                        drop_last=True, collate_fn=yolo_dataset_collate)
            else:
                gen = Generator(Batch_size, lines[:num_train],
                                (input_shape[0], input_shape[1])).generate(train=True, mosaic = mosaic)
                gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate(train=False, mosaic = False)
            
            epoch_size = max(1, num_train//Batch_size)
            epoch_size_val = num_val//Batch_size
            #------------------------------------#
            #   冻结一定部分训练
            #------------------------------------#
            for param in model.backbone.parameters():
                param.requires_grad = False

            for epoch in range(Init_Epoch,Freeze_Epoch):
                fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda,writer,run_data,run)
                lr_scheduler.step()


            lr = run.lr 
            Batch_size =16
            Freeze_Epoch = 40
            Unfreeze_Epoch = 300

            #----------------------------------------------------------------------------#
            #   我在实际测试时，发现optimizer.Adam的weight_decay起到了反作用，
            #   optimizer.SGD 模型无法收敛
            #   所以去除掉了weight_decay，也可以打开试试，一般是weight_decay=5e-4
            #   一般选择这两种优化器
            #----------------------------------------------------------------------------#
            if run.optim == 'SGD':
                optimizer = optim.SGD(net.parameters(),lr, momentum=0.9, weight_decay=1e-4)
            else:
                optimizer = optim.Adam(net.parameters(),lr)
            # 是否采用余弦退火学习率           
            if cosine_lr:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.4)
            
            if Use_Data_Loader:
                train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
                val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
                gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                        drop_last=True, collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                        drop_last=True, collate_fn=yolo_dataset_collate)
            else:
                gen = Generator(Batch_size, lines[:num_train],
                                (input_shape[0], input_shape[1])).generate(train=True, mosaic = mosaic)
                gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate(train=False, mosaic = False)
            
            epoch_size = max(1, num_train//Batch_size)
            epoch_size_val = num_val//Batch_size
            #------------------------------------#
            #   解冻后训练
            #------------------------------------#
            for param in model.backbone.parameters():
                param.requires_grad = True

            for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
                fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda,writer,run_data,run)
                lr_scheduler.step()


        # 直接解冻backbone方式训练
        elif run.freezetrain == False:
            lr = run.lr
            Batch_size = 8
            Init_Epoch = 200
            Unfreeze_Epoch = 300
            
            #----------------------------------------------------------------------------#
            #   我在实际测试时，发现optimizer.Adam的weight_decay起到了反作用，
            #   所以去除掉了weight_decay，也可以打开试试，一般是weight_decay=5e-4
            #   一般选择这两种优化器
            #----------------------------------------------------------------------------#
            if run.optim == 'SGD':
                optimizer = optim.SGD(net.parameters(),lr, momentum=0.9, weight_decay=1e-4)
            else:
                optimizer = optim.Adam(net.parameters(),lr)
            # 是否采用余弦退火学习率  
            if cosine_lr:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)


            if Use_Data_Loader:
                train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
                val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
                gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                        drop_last=True, collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                        drop_last=True, collate_fn=yolo_dataset_collate)
            else:
                gen = Generator(Batch_size, lines[:num_train],
                                (input_shape[0], input_shape[1])).generate(train=True, mosaic = mosaic)
                gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate(train=False, mosaic = False)
            
            epoch_size = max(1, num_train//Batch_size)
            epoch_size_val = num_val//Batch_size
            #------------------------------------#
            #   解冻后训练
            #------------------------------------#
            for param in model.backbone.parameters():
                param.requires_grad = True
            
            # 训练epoch循环
            for epoch in range(Init_Epoch,Unfreeze_Epoch):
                fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda,writer,run_data,run)
                lr_scheduler.step()
