import os
import sys
sys.path.insert(0, '../../')
import time
import glob
import numpy as np
import torch
import optimizers.darts.utils as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from resnet import *
from resnet import ResNet18


import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from sota.cnn.model_search_pcdarts import PCDARTSNetwork as Network
from optimizers.darts.architect import Architect
from sota.cnn.spaces import spaces_dict
from sota.cnn.helper import progress_bar

from attacker.perturb import Linf_PGD_alpha, Random_alpha, AttackPGD

from copy import deepcopy
from numpy import linalg as LA

from torch.utils.tensorboard import SummaryWriter
##############################################################################################################################
import torchvision.models as models
# resnet18 = models.resnet18()
# # torch.cuda.clear_memory_allocated()
# del Variable
# gc.collect()
# torch.cuda.empty_cache()
# resnet18 = resnet18.cuda()
# print(torch.cuda.memory_summary(torch.device('cuda:0')))
##############################################################################################################################

parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='../../data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate2', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--weight_decay2', type=float, default=5e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--search_space', type=str, default='s5', help='searching space to choose from')
parser.add_argument('--perturb_alpha', type=str, default='pgd_linf', help='perturb for alpha')
parser.add_argument('--epsilon_alpha', type=float, default=0.3, help='max epsilon for alpha')
args = parser.parse_args()

args.save = '../../experiments/sota/{}/search-pcdarts-{}-{}-{}-{}'.format(
    args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"), args.search_space, args.seed)

if args.unrolled:
    args.save += '-unrolled'
if not args.weight_decay == 3e-4:
    args.save += '-weight_l2-' + str(args.weight_decay)
if not args.arch_weight_decay == 1e-3:
    args.save += '-alpha_l2-' + str(args.arch_weight_decay)
if args.cutout:
    args.save += '-cutout-' + str(args.cutout_length) + '-' + str(args.cutout_prob)
if not args.perturb_alpha == 'none':
    args.save += '-alpha-' + args.perturb_alpha + '-' + str(args.epsilon_alpha)
args.save += '-' + str(np.random.randint(10000))

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save + '/runs')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.dataset == 'cifar100':
    n_classes = 100
else:
    n_classes = 10


    
def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    
    
#     if args.perturb_alpha == 'none':
#         perturb_alpha = None
#     elif args.perturb_alpha == 'pgd_linf':
#         perturb_alpha = Linf_PGD_alpha
#     elif args.perturb_alpha == 'random':
#         perturb_alpha = Random_alpha

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    #######################################
#     resnet18 = models.resnet18()
#     resnet18 = resnet18.cuda()
#     model2 = resnet18
    model2 = ResNet18()
#     model2 = model2.cuda()
    model2 = model2.to(device)
    if device == 'cuda':
        model2 = torch.nn.DataParallel(model2)
        cudnn.benchmark = True
    ######################################
    model = Network(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space])
    model = model.cuda()
#     model_adv = AttackPGD(model)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model2.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    
    optimizer2 = torch.optim.SGD(
        model2.parameters(),
        args.learning_rate2,
        momentum=args.momentum,
        weight_decay=args.weight_decay2)
    

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)


    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
#     if 'debug' in args.save:
#         split = args.batch_size
#         num_train = 2 * args.batch_size

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)


    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        scheduler2.step()
        lr = scheduler.get_lr()[0]
        lr2 = scheduler2.get_lr()[0]
        if args.cutout:
            # increase the cutout probability linearly throughout search
            train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr %e', epoch, lr)
        
        
#         if args.perturb_alpha:
#             epsilon_alpha = 0.03 + (args.epsilon_alpha - 0.03) * epoch / args.epochs
#             logging.info('epoch %d epsilon_alpha %e', epoch, epsilon_alpha)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer,optimizer2, lr,lr2, 
                                           model2, epoch)
        logging.info('train_acc %f', train_acc)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Obj/train', train_obj, epoch)

        # validation
#         valid_acc, valid_obj = infer(valid_queue, model, criterion)
############################################################################################################
        valid_acc, valid_obj = infer(valid_queue, model2, criterion)
############################################################################################################
        logging.info('valid_acc %f', valid_acc)
        writer.add_scalar('Acc/valid', valid_acc, epoch)
        writer.add_scalar('Obj/valid', valid_obj, epoch)
        utils.save(model, os.path.join(args.save, 'weights.pt'))
    writer.close()


def train(train_queue, valid_queue, model, architect, criterion, optimizer,optimizer2, lr,lr2, model2, epoch):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    
    train_loss = 0
    correct = 0
    total = 0
    max_step = 0

    for step, (input, target) in enumerate(train_queue):
        model.train()
#         model2.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)


        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
        architect.optimizer.zero_grad()


        
        optimizer.zero_grad()       
#         logits, diff, x = model(input, target)
        logits = model(input, updateType='weight')
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()  
        
        
        
#         model_adv = AttackPGD(model)
#         logits1, diff, x = model_adv(input, target)
#         deltas = torch.round(torch.abs(diff) * 255/8 + 0.499 - (epoch/300))
#         pert_inp = torch.mul (input, deltas)
# #         pert_inp = torch.mul (input, torch.abs(diff))
#         optimizer2.zero_grad()       
#         logits2 = model2(pert_inp)
#         loss2 = criterion(logits2, target)
#         loss2.backward()
#         nn.utils.clip_grad_norm_(model2.parameters(), args.grad_clip)
#         optimizer2.step() 
 
#         if epoch<0:
#             optimizer2.zero_grad()       
#             logits2 = model2(input)
#             loss2 = criterion(logits2, target)
#             loss2.backward()
#             nn.utils.clip_grad_norm_(model2.parameters(), args.grad_clip)
#             optimizer2.step() 
        

#         else:
#             model_adv = AttackPGD(model)
#             logits1, diff, x = model_adv(input, target)
#             deltas = torch.round(torch.abs(diff) * 255/8 + 0.499 - (epoch/300))
#             pert_inp = torch.mul (input, deltas)
#     #         pert_inp = torch.mul (input, torch.abs(diff))
#             optimizer2.zero_grad()       
#             logits2 = model2(pert_inp)
#             loss2 = criterion(logits2, target)
#             loss2.backward()
#             nn.utils.clip_grad_norm_(model2.parameters(), args.grad_clip)
#             optimizer2.step() 
            
#         train_loss += loss2.item()
#         _, predicted = logits2.max(1)
#         total += target.size(0)
#         correct += predicted.eq(target).sum().item()
#         max_step = step


        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            if 'debug' in args.save:
                break

#         progress_bar(step, len(train_queue), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                      % (train_loss/(step+1), 100.*correct/total, correct, total))


#     return  100.*correct/total, train_loss/(max_step+1)
    return top1.avg, objs.avg




def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    max_step = 0
    best_acc = 0
    
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)
            
            
            test_loss += loss.item()
            _, predicted = logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            max_step = step
            

            progress_bar(step, len(valid_queue), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(step+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        

    return 100.*correct/total, test_loss/(max_step+1)


def trainres(train_queue,model2, criterion, optimizer):
    epoch = 50
    print('\nEpoch: %d' % epoch)
    model2.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_queue):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model2(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_queue), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


if __name__ == '__main__':
    main() 
