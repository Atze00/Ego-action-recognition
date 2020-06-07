from __future__ import print_function, division
from ModelsRGB import *
from spatial_transforms import (Compose, ToTensor, CenterCrop, Scale, Normalize, MultiScaleCornerCrop,
                                RandomHorizontalFlip,UnNormalize)
from tensorboardX import SummaryWriter
from makeDatasetRGB_flowsupervision import *
from torchvision.utils import save_image
import argparse
import sys


def main_run(model, supervision, train_data_dir, val_data_dir, out_dir, seq_len, train_batch_size,
             val_batch_size, num_epochs, lr1, lr_suphead, lr_resnet, alpha, decay_factor, decay_step, 
             mem_size):

    num_classes = 61
    if model == 'MyNet':
        model = MyNet(num_classes=num_classes, mem_size=mem_size)
    else:
        print('Model not found')
        sys.exit()
    
    model_folder = os.path.join('./', out_dir, 'rgb')  # Dir for saving models and log files
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
    unnormalize = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spatial_transform = Compose([Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
                                 ToTensor(), normalize])
    vid_seq_train = MakeDataset(train_data_dir, train=True,
                                spatial_transform=spatial_transform, 
                                seq_len=seq_len, fmt='.png')

    train_loader = torch.utils.data.DataLoader(vid_seq_train, batch_size=train_batch_size,
                                               shuffle=True, num_workers=8, pin_memory=True)
    if val_data_dir is not None:
        vid_seq_val = MakeDataset(val_data_dir, train=False,
                                  spatial_transform=Compose([Scale(256), CenterCrop(224), ToTensor(), normalize]),
                                  seq_len=seq_len, fmt='.png')

        val_loader = torch.utils.data.DataLoader(vid_seq_val, batch_size=val_batch_size,
                                                 shuffle=False, num_workers=8, pin_memory=True)

    train_params = []
    train_params3 = []
    train_params2 = []
    for params in model.resNet.parameters():
        params.requires_grad = True
        train_params += [params]
        
    for params in model.lstm_cell.parameters():
        params.requires_grad = True
        train_params2 += [params]

    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params2 += [params]
    for params in model.sup_head.parameters():
        params.requires_grad = True
        train_params3 += [params]

    model.train()
    model.cuda()
    loss_sup = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer_fn = torch.optim.Adam([{"params": train_params, "lr": lr_resnet}, 
                                     {"params": train_params3, "lr": lr_suphead},
                                     {"params": train_params2, "lr": lr1}], 
                                    lr=lr1, weight_decay=4e-5, eps=1e-4)
    
    optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=decay_step,
                                                           gamma=decay_factor)

    train_iter = 0
    min_accuracy = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_corr_train = 0
        train_samples = 0
        iter_per_epoch = 0
        epoch_loss_ = 0
        writer.add_scalar('lr', optimizer_fn.param_groups[0]['lr'], epoch+1)
        model.train()
        for i, (inputs, targets, m) in enumerate(train_loader):
            train_iter += 1
            iter_per_epoch += 1
            optimizer_fn.zero_grad()
            images = inputs.permute(1, 0, 2, 3, 4).cuda()
            labels = targets.cuda()
            m = m.permute(1, 0, 2, 3, 4).cuda()
            train_samples += inputs.size(0)
            output_label, _, output_super = model(images)
            if supervision:
                loss_ = loss_sup(output_super, m.cuda())
                epoch_loss_ += loss_.data.item()

            loss = loss_fn(output_label, labels)
            epoch_loss += loss.data.item()
            if supervision:
                loss = loss+loss_*alpha
            loss.backward()
            optimizer_fn.step()
            _, predicted = torch.max(output_label.data, 1)
            num_corr_train += (predicted == targets.cuda()).sum()
            
        optim_scheduler.step()
        avg_loss = epoch_loss/iter_per_epoch
        train_acc = (num_corr_train / float(train_samples)) * 100
        output_super = output_super.cpu()
        m = m.cpu()
        if (epoch % 2 == 0) and supervision:
            d = f"./Results/epoch{epoch}"
            os.makedirs(d)
            for i in range(m.shape[0]):
                save_image(unnormalize(m[i, 0]), d+f'/inp{i}.png')
                save_image(unnormalize(output_super[i, 0].detach()), d+f'/out{i}.png')
        avg_loss_ = epoch_loss_/float(iter_per_epoch)
        print('Train: Epoch = {} | Loss = {} | Accuracy = {} | supervision_loss {}'.format(epoch+1, avg_loss,
                                                                                           train_acc, avg_loss_))
        train_log_loss.write('Train Loss after {} epochs = {}\n'.format(epoch + 1, avg_loss))
        train_log_acc.write('Train Accuracy after {} epochs = {}%\n'.format(epoch + 1, train_acc))
        writer.add_scalar('train/epoch_loss', avg_loss, epoch+1)
        writer.add_scalar('train/accuracy', train_acc, epoch+1)

        if val_data_dir is not None:
            if (epoch+1) % 1 == 0:
                model.eval()
                val_loss_epoch = 0
                val_iter = 0
                val_samples = 0
                num_corr = 0
                for j, (inputs, targets, _) in enumerate(val_loader):
                    val_iter += 1
                    val_samples += inputs.size(0)
                    with torch.no_grad():
                        images = inputs.permute(1, 0, 2, 3, 4).cuda()
                        labels = targets.cuda(non_blocking=True)
                        output_label, _, _ = model(images)
                        val_loss = loss_fn(output_label, labels)
                        val_loss_epoch += val_loss.data.item()
                        _, predicted = torch.max(output_label.data, 1)
                        num_corr += (predicted == targets.cuda()).sum()
                val_accuracy = (num_corr / float(val_samples)) * 100
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
    parser.add_argument('--model', type=str, default="MyNet", help='model implementation')
    parser.add_argument('--supervision', type=str, default=None, help='Self-supervision task or not')
    parser.add_argument('--train_data_dir', type=str, default='./dataset/gtea_warped_flow_61/split2/train',
                        help='Train set directory')
    parser.add_argument('--val_data_dir', type=str, default=None,
                        help='Val set directory')
    parser.add_argument('--out_dir', type=str, default='experiments', help='Directory to save results')
    parser.add_argument('--seq_len', type=int, default=25, help='Length of sequence')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Validation batch size')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_suphead', type=float, default=None, help='Learning rate')
    parser.add_argument('--lr_resnet', type=float, default=None, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='supervised loss multiplier')
    parser.add_argument('--step_size', type=float, default=[25, 75, 150], nargs="+", help='Learning rate decay step')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--mem_size', type=int, default=512, help='ConvLSTM hidden state size')

    args = parser.parse_args()
    if args.lr_suphead is None:
        lr_suphead = args.lr
    else:
        lr_suphead = args.lr_suphead
    
    if args.lr_resnet is None:
        lr_resnet = args.lr
    else:
        lr_resnet = args.lr_resnet

    if args.supervision == "True":
        supervision = True
    elif args.supervision == "False":
        supervision = False
    else: 
        print('invalid value for supervision')
        sys.exit()

    model = args.model
    alpha = args.alpha
    train_data_dir = args.train_data_dir
    val_data_dir = args.val_data_dir
    out_dir = args.out_dir
    seq_len = args.seq_len
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    num_epochs = args.num_epochs
    lr1 = args.lr
    decay_step = args.step_size
    decay_factor = args.decay_rate
    mem_size = args.mem_size
    main_run(model, supervision, train_data_dir, val_data_dir, out_dir, seq_len, train_batch_size,
             val_batch_size, num_epochs, lr1, lr_suphead, lr_resnet, alpha, decay_factor, decay_step,
             mem_size)



__main__()
