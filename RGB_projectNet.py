from __future__ import print_function, division
from ModelsRGB import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip)
from torchvision.transforms import Resize 
from torchvision.transforms import ToTensor as TT
from torchvision.transforms import Compose as Cp
from tensorboardX import SummaryWriter
from makeDatasetRGB import *
import argparse
import sys


def main_run( stage, model, supervision, train_data_dir, val_data_dir, stage1_dict, out_dir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1,lr_suphead, lr_resnet, alpha,decay_factor, decay_step,lossSupervision, memSize):
    

    num_classes = 61

    
    if model=='ConvLSTMAttention':
      model=ConvLSTMAttention(num_classes=num_classes, mem_size=memSize,supervision=supervision,loss_supervision=lossSupervision)
    elif model=='ConvLSTM':
      model=ConvLSTM(num_classes=num_classes, mem_size=memSize,supervision=supervision,loss_supervision=lossSupervision)
    elif model=='SupervisedLSTMMod':
      model=SupervisedLSTMMod(num_classes=num_classes, mem_size=memSize,supervision=supervision,loss_supervision=lossSupervision)
    elif model == 'MyNetIDT':
      model = MyNetIDT(num_classes=num_classes, mem_size=memSize,supervision=supervision,loss_supervision=lossSupervision)
    else:
      print('Model not found')
      sys.exit()
    
    model_folder = os.path.join('./', out_dir,  'rgb', 'stage'+str(stage))  # Dir for saving models and log files
    # Create the dir
    if os.path.exists(model_folder):
        print('Directory {} exists!'.format(model_folder))
        sys.exit()
    os.makedirs(model_folder)

    # Log files
    writer = SummaryWriter(model_folder)
    train_log_loss = open((model_folder + '/train_log_loss.txt'), 'w')
    train_log_acc = open((model_folder + '/train_log_acc.txt'), 'w')
    val_log_loss = open((model_folder + '/val_log_loss.txt'), 'w')
    val_log_acc = open((model_folder + '/val_log_acc.txt'), 'w')


    # Data loader
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                 ToTensor(), normalize])
    spatial_transform_map = Cp([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224), Resize((7,7)),
                                 TT()])
    spatial_transform_map_2 = Cp([ Resize((7,7)),
                                 TT()])
    vid_seq_train = makeDataset_supervision(train_data_dir,train=True,
                                spatial_transform=spatial_transform, spatial_transform_map=spatial_transform_map,seqLen=seqLen, fmt='.png')

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=trainBatchSize,
                            shuffle=True, num_workers=8, pin_memory=True)
    if val_data_dir is not None:

        vid_seq_val = makeDataset_supervision(val_data_dir,train=False,spatial_transform_map=spatial_transform_map_2,
                                   spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                                   seqLen=seqLen, fmt='.png')

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=valBatchSize,
                                shuffle=False, num_workers=8, pin_memory=True)
        valInstances = vid_seq_val.__len__()
    trainInstances = vid_seq_train.__len__()
    train_params = []
    train_params3 = []
    train_params2 = []
    if stage == 0:
      for params in model.resNet.parameters():
          params.requires_grad = True
          train_params += [params]
      if stage1_dict is not None:
        model.load_state_dict(torch.load(stage1_dict))
    elif stage == 1:
      supervision=False
      model.eval()
      for params in model.parameters():
            params.requires_grad = False
    else:
        model.load_state_dict(torch.load(stage1_dict))
        model.train()
        for params in model.parameters():
            params.requires_grad = False
        #
        for params in model.resNet.layer4[0].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[0].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[1].conv1.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[1].conv2.parameters():
            params.requires_grad = True
            train_params += [params]

        for params in model.resNet.layer4[2].conv1.parameters():
            params.requires_grad = True
            train_params += [params]
        #
        for params in model.resNet.layer4[2].conv2.parameters():
            params.requires_grad = True
            train_params += [params]
        #
        for params in model.resNet.fc.parameters():
            params.requires_grad = True
            train_params += [params]

        model.resNet.layer4[0].conv1.train(True)
        model.resNet.layer4[0].conv2.train(True)
        model.resNet.layer4[1].conv1.train(True)
        model.resNet.layer4[1].conv2.train(True)
        model.resNet.layer4[2].conv1.train(True)
        model.resNet.layer4[2].conv2.train(True)
        model.resNet.fc.train(True)
        model.sup_head.train()
        
    for params in model.lstm_cell.parameters():
        params.requires_grad = True
        train_params2 += [params]

    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params2 += [params]
    for params in model.sup_head.parameters():
        params.requires_grad = True
        train_params3 += [params]
        
    model.lstm_cell.train()
    model.classifier.train()
    model.cuda()
    if lossSupervision=="classification":
      loss_sup=nn.CrossEntropyLoss()
    elif lossSupervision=="regression":
      loss_sup=nn.L1Loss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = torch.optim.Adam([{"params":train_params,"lr": lr_resnet},{"params":train_params3,"lr":lr_suphead},
      {"params":train_params2,"lr":lr1}] , lr=lr1, weight_decay=4e-5, eps=1e-4)
    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=decay_step,
                                                           gamma=decay_factor)

    train_iter = 0
    min_accuracy = 0
    
    for epoch in range(numEpochs):
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        epoch_loss_ = 0
        loss_=0
        model.lstm_cell.train(True)
        model.classifier.train(True)
        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch+1)
        if stage == 0:
            model.train()
        if stage == 2:
            model.resNet.layer4[0].conv1.train(True)
            model.resNet.layer4[0].conv2.train(True)
            model.resNet.layer4[1].conv1.train(True)
            model.resNet.layer4[1].conv2.train(True)
            model.resNet.layer4[2].conv1.train(True)
            model.resNet.layer4[2].conv2.train(True)
            model.sup_head.train()
            model.resNet.fc.train(True)
        for i, (inputs, targets,maps) in enumerate(train_loader):
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            if lossSupervision=="classification":
              maps=torch.ceil(maps)
              maps=maps.type(torch.LongTensor)
              maps = maps.permute(1,0,2,3,4).squeeze(2).cuda()
              maps=maps.reshape(maps.shape[0]*maps.shape[1],maps.shape[2],maps.shape[3])
            else:
              maps = maps.permute(1,0,2,3,4).cuda()
              maps=maps.reshape(maps.shape[0]*maps.shape[1],maps.shape[2],maps.shape[3],maps.shape[4])
            inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).cuda())
            labelVariable = Variable(targets.cuda())
            trainSamples += inputs.size(0)
            output_label, _,output_super = model(inputVariable)
            if supervision==True:
              loss_=alpha*loss_sup(output_super,maps)
              loss_.backward(retain_graph=True)
              epoch_loss_ += loss_.data.item()
            loss = loss_fn(output_label, labelVariable)
            loss.backward()
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += (predicted == targets.cuda()).sum()
            epoch_loss += loss.data.item()
        optim_scheduler.step()
        avg_loss = epoch_loss/iterPerEpoch
        trainAccuracy = (numCorrTrain / float(trainSamples)) * 100
        
        avg_loss_ = epoch_loss_/float(iterPerEpoch)
        print('Train: Epoch = {} | Loss = {} | Accuracy = {} | supervision_loss {}'.format(epoch+1, avg_loss, trainAccuracy,avg_loss_))
        train_log_loss.write('Train Loss after {} epochs = {}\n'.format(epoch + 1, avg_loss))
        train_log_acc.write('Train Accuracy after {} epochs = {}%\n'.format(epoch + 1, trainAccuracy))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch+1)
        writer.add_scalar('train/accuracy', trainAccuracy, epoch+1)
        if val_data_dir is not None:
            if (epoch+1) % 1 == 0:
                model.eval()
                val_loss_epoch = 0
                val_iter = 0
                val_samples = 0
                numCorr = 0
                for j, (inputs, targets,_) in enumerate(val_loader):
                    val_iter += 1
                    val_samples += inputs.size(0)
                    with torch.no_grad():
                      inputVariable = Variable(inputs.permute(1, 0, 2, 3, 4).cuda())
                      labelVariable = Variable(targets.cuda(non_blocking=True))
                      output_label, _,_ = model(inputVariable)
                      val_loss = loss_fn(output_label, labelVariable)
                      val_loss_epoch += val_loss.data.item()
                      _, predicted = torch.max(output_label.data, 1)
                      numCorr += (predicted == targets.cuda()).sum()
                val_accuracy = (numCorr / float(val_samples)) * 100
                avg_val_loss = val_loss_epoch / val_iter
                print('val: Epoch = {} | Loss = {} | Accuracy = {} '.format(epoch+1, avg_val_loss, val_accuracy))
                writer.add_scalar('val/epoch_loss', avg_val_loss, epoch + 1)
                writer.add_scalar('val/accuracy', val_accuracy, epoch + 1)
                val_log_loss.write('Val Loss after {} epochs = {}\n'.format(epoch + 1, avg_val_loss))
                val_log_acc.write('Val Accuracy after {} epochs = {}%\n'.format(epoch + 1, val_accuracy))
                if val_accuracy > min_accuracy:
                    save_path_model = (model_folder + '/model_rgb_state_dict.pth')
                    torch.save(model.state_dict(), save_path_model)
                    min_accuracy = val_accuracy
            else:
                if (epoch+1) % 10 == 0:
                    save_path_model = (model_folder + '/model_rgb_state_dict_epoch' + str(epoch+1) + '.pth')
                    torch.save(model.state_dict(), save_path_model)

    train_log_loss.close()
    train_log_acc.close()
    val_log_acc.close()
    val_log_loss.close()
    writer.export_scalars_to_json(model_folder + "/all_scalars.json")
    writer.close()


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=None, help='Training stage')
    parser.add_argument('--model', type=str, default=None, help='model implementation')
    parser.add_argument('--supervision', type=str, default=None, help='Self-supervision task or not')
    parser.add_argument('--trainDatasetDir', type=str, default='./dataset/gtea_warped_flow_61/split2/train',
                        help='Train set directory')
    parser.add_argument('--valDatasetDir', type=str, default=None,
                        help='Val set directory')
    parser.add_argument('--outDir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--stage1Dict', type=str, default=None,
                        help='Stage 1 model path')
    parser.add_argument('--seqLen', type=int, default=25, help='Length of sequence')
    parser.add_argument('--trainBatchSize', type=int, default=32, help='Training batch size')
    parser.add_argument('--valBatchSize', type=int, default=64, help='Validation batch size')
    parser.add_argument('--numEpochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_suphead', type=float, default=None, help='Learning rate')
    parser.add_argument('--lr_resnet', type=float, default=None, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=1, help='supervised loss multiplier')
    parser.add_argument('--stepSize', type=float, default=[25, 75, 150], nargs="+", help='Learning rate decay step')
    parser.add_argument('--decayRate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--lossSupervision', type=str, default="classification", help='type of loss, regression or classification')
    parser.add_argument('--memSize', type=int, default=512, help='ConvLSTM hidden state size')

    args = parser.parse_args()
    if args.lr_suphead==None:
     lr_suphead=args.lr
    else:
     lr_suphead=args.lr_suphead
    
    if args.lr_resnet==None:
     lr_resnet=args.lr
    else:
     lr_resnet=args.lr_resnet
    
    
    if args.supervision=="True":
     supervision=True
    elif args.supervision=="False":
     supervision=False
    else: 
     print('invalid value for supervision')
     sys.exit()
    stage1Dict = args.stage1Dict
    alpha = args.alpha
    stage = args.stage
    model= args.model
    trainDatasetDir = args.trainDatasetDir
    valDatasetDir = args.valDatasetDir
    outDir = args.outDir
    seqLen = args.seqLen
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    numEpochs = args.numEpochs
    lr1 = args.lr
    stepSize = args.stepSize
    decayRate = args.decayRate
    memSize = args.memSize
    lossSupervision = args.lossSupervision
    main_run( stage, model,supervision,trainDatasetDir, valDatasetDir, stage1Dict, outDir, seqLen, trainBatchSize,
             valBatchSize, numEpochs, lr1,lr_suphead, lr_resnet, alpha,decayRate, stepSize,lossSupervision, memSize)

__main__()
