import os 
import torch
import torch.nn as nn
from torch.nn import functional as F 
import numpy as np
from torchvision import datasets, transforms 
from torch.utils import data
from DiMIM import DiMIMLoss, DinoHead, Convit
from utils import clip_gradient, MultiCropWrapper, DataAugmentation, MaskingGenerator
from evaluate import evaluate


def train_func(train_loader, student, teacher, optimizer, loss_func, momentum_teacher, max_epochs = 100,  
                validation_loader = None, batch_size = 128, scheduler = None, device = None, test_loader = None, 
                train_loader_plain = None, clip_grad = 2.0, start_epoch = 0, mask_generator = None):

    """Train function for dino. It takes two identical models, the teacher and student, 
    and only the student model is trained. Note that Both the teacher and the student
    model share the same architecture, and initially, they also have the same parameters.
    The parameters of the teacher are updated using the exponential moving average of 
    the student model.


    Parameters
    ----------
    train_loader: Instance of `torch.utils.data.DataLoader`

    student: Instance of `torch.nn.Module'
            The Vision Transformer as the student model.

    teacher: Instance of `torch.nn.Module'
        The Vision Transformer as the teacher model. 

    optimizer: Instance of `torch.optim`
            Optimizer for training.

    loss_func: Instance of `torch.nn.Module'
            Loss function for the training.


    Returns
    -------
    history: dict 
            Returns training and validation loss. 

    """

    n_batches_train = len(train_loader)
    n_batches_val = len(validation_loader)
    n_samples_train = batch_size * n_batches_train
    n_samples_val = batch_size * n_batches_val


    losses = []
    accuracy = []
    validation_acc_knn = []
    validation_accuracy_logistic = []


    for epoch in range(start_epoch, max_epochs):
        running_loss, correct = 0, 0
        for images, labels in train_loader:
            if device:
                images = [img.to(device, non_blocking = True) for img in images]
                labels = labels.to(device, non_blocking = True)
                mask = mask_generator().to(device, non_blocking = True)

            mask = mask.flatten().unsqueeze(-1)
            ####print(mask.shape)

            #Training
            student.train()
            student.training = True

            #The models should return patches for the training part
            student.backbone.return_patches = True
            teacher.backbone.return_patches = True

            cls_student = student(images, bool_masked_position = mask)
            cls_teacher = teacher(images) #Teacher only gets the global crops
            loss = loss_func(student_output = cls_student, teacher_output = cls_teacher, mask = mask)

            running_loss += loss.item()


            #================= BACKWARD AND OPTIMZIE  ====================================   
            optimizer.zero_grad()
            loss.backward()
            clip_gradient(student, clip_grad)
            optimizer.step()

                    
            #================== Updating the teacher's parameters ========================
            with torch.no_grad():
                for student_params, teacher_params in zip(student.parameters(), teacher.parameters()):
                    teacher_params.data.mul_(momentum_teacher)
                    teacher_params.data.add_((1 - momentum_teacher) * student_params.detach().data)


        loss_epoch = running_loss / n_batches_train
        losses.append(loss_epoch)
        scheduler.step()

        print('Epoch [{}/{}], Training Loss: {:.4f}'
            .format(epoch + 1, max_epochs, loss_epoch), end = '  ')


        #====================== Validation ============================
        if validation_loader:

            student.eval()   

            acc_logistic, acc_val_logistic, _ = evaluate(student.backbone, train_loader_plain, validation_loader)
            #validation_acc_knn.append(acc_val_knn)
            accuracy.append(acc_logistic)
            validation_accuracy_logistic.append(acc_val_logistic)

            print('Training accuracy Logistic [{:.3f}], Validation accuracy Logistic [{:.3f}]]'
                .format(acc_logistic, acc_val_logistic))
        
        #====================== Saving the Model ============================  
        checkpoint = {
            'start_epoch': epoch,
            'student': student.state_dict(), 
            'teacher': teacher.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'scheduler' : scheduler.state_dict(),
            'loss': None
        }

        path = F"{os.getcwd()}/checkpoint.pt"
        torch.save(checkpoint, path)
    
    #====================== Testing ============================      
    if test_loader:
        correct = 0
        total = 0

        for images, labels in test_loader:
            if device:
                images = images.to(device)
                labels = labels.to(device)

            n_data = images[0]
            total += n_data
            outputs = student(images)
            predictions = outputs.argmax(1)
            correct += int(sum(predictions == labels))

        accuracy = correct / total 
        print('Test Accuracy: {}'.format(accuracy))



    return  {'loss': losses, 'accuracy': accuracy, 
            'val_acc_logistic': validation_accuracy_logistic, 'val_accuracy_knn': validation_acc_knn}



def main(parameters):

    #=============================Preparing Data==================================
    cudnn.benchmark = True
    path = F"../kaggle/input"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Starting Dino With Convit Backbone....')
    print(device)
    plain_augmentation = transforms.Compose([
        #transforms.Resize(32),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    dino_augmentation = DataAugmentation(n_local_crops = parameters['n_crops'] - 2)
    dataset_train = datasets.CIFAR10(path, download = True, train = True, transform = dino_augmentation)
    dataset_test = datasets.CIFAR10(path, download = True, train = False, transform = plain_augmentation)
    dataset_train_evaluation = datasets.CIFAR10(path, download = True, train = True, transform = plain_augmentation)
    dataset_validation, dataset_test = torch.utils.data.random_split(dataset_test, [8000, 2000])


    train_loader = data.DataLoader(dataset_train, batch_size = parameters['batch_size'], drop_last = True, num_workers = 2)
    val_loader = data.DataLoader(dataset_validation, batch_size = parameters['batch_size'], drop_last = True, num_workers = 2)
    train_loader_plain = data.DataLoader(dataset_train_evaluation, batch_size = parameters['batch_size'], drop_last = True, num_workers = 2)

    #=============================Preparing mask generator=============================
    n_patches = int(parameters['img_size']/parameters['patch_size'])
    mask_generator = MaskingGenerator((n_patches, n_patches), num_masking_patches = 20, min_num_patches = 3, max_num_patches = 20, 
                                      min_aspect = 0.3, max_aspect = None, patch_size = parameters['patch_size'])
   
    #=============================Preparing The Model==================================
    student = Convit(img_size = parameters['img_size'], patch_size = parameters['patch_size'], 
        n_classes = parameters['n_classes'], embed_dim = parameters['embed_dim'], 
        n_heads = parameters['n_heads'], mlp_ratio = parameters['mlp_ratio'], qkv_bias = parameters['qkv_bias'], 
        drop = parameters['drop'], attn_drop = parameters['attn_drop'], local_layers = parameters['local_layers'], 
        locality_strength = parameters['locality_strength'], depth = parameters['depth'], use_pos_embed = parameters['use_pos_embed'])

    teacher = Convit(img_size = parameters['img_size'], patch_size = parameters['patch_size'], 
        n_classes = parameters['n_classes'], embed_dim = parameters['embed_dim'],
        n_heads = parameters['n_heads'], mlp_ratio = parameters['mlp_ratio'], qkv_bias = parameters['qkv_bias'], 
        drop = parameters['drop'], attn_drop = parameters['attn_drop'], local_layers = parameters['local_layers'], 
        locality_strength = parameters['locality_strength'], depth = parameters['depth'], use_pos_embed = parameters['use_pos_embed'])
    
    n_parameters = sum(param.numel() for param in student.parameters() if param.requires_grad)
    print('The number of trainable parameters is : {}'.format(n_parameters))

    head_student = DinoHead(in_dim = parameters['embed_dim'], hidden_dim = 384, out_dim = parameters['out_dim'], n_layers = 3, norm_last_layer = True)
    head_teacher = DinoHead(in_dim = parameters['embed_dim'], hidden_dim = 384, out_dim = parameters['out_dim'], n_layers = 3, norm_last_layer = True)
    student = MultiCropWrapper(student, head_student)
    teacher = MultiCropWrapper(teacher, head_teacher)
    student, teacher = student.to(device), teacher.to(device)
    #path = '{}'.format(os.getcwd())
    path = '/kaggle/input/dino-mim-local-2/checkpoint (439).pt'
    
    #=========Loading Checkpoints=================
    #checkpoint = torch.load(path)
    #student.load_state_dict(checkpoint['student'])
    #teacher.load_state_dict(checkpoint['teacher'])
    #learning_rate = checkpoint['lr']
    #start_epoch = checkpoint['start_epoch']
    teacher.load_state_dict(student.state_dict()) #Making sure that the two networks' parameters are the same initially
    
    for params in teacher.parameters(): 
        params.requires_grad = False

    criterion = DiMIMLoss(parameters['out_dim'], teacher_temp = parameters['teacher_temp'], 
        student_temp = parameters['student_temp'], center_momentum = parameters['center_momentum'], ncrops = parameters['n_crops']).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr = parameters['lr'], weight_decay = parameters['weight_decay'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min = 1e-6)
    #base_scheduler.load_state_dict(checkpoint['scheduler'])
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch = 5, after_scheduler = base_scheduler)
    
    print(base_scheduler.get_last_lr())
    momentum_teacher = parameters['momentum_teacher']
    history = train_func(train_loader = train_loader, student = student, teacher = teacher, 
        optimizer = optimizer, loss_func = criterion, validation_loader = val_loader, 
        device = device, scheduler = scheduler, batch_size = parameters['batch_size'], 
        max_epochs = parameters['max_epochs'], momentum_teacher = momentum_teacher, 
        train_loader_plain = train_loader_plain, clip_grad = parameters['clip_grad'], 
        start_epoch = 0, mask_generator = mask_generator)

    return student, history



if __name__ == '__main__':

    parameters = {'batch_size': 64, 'lr': 0.0005, 'weight_decay': 0.03, 'img_size': 32, 'n_crops': 6, 
                'n_heads' : 4, 'patch_size' : 4, 'n_classes' : 0, 
                'embed_dim' : 192, 'out_dim': 1024, 'teacher_temp' : 0.04, 'student_temp' : 0.1, 
                'center_momentum' : 0.9, 'max_epochs' : 600, 'momentum_teacher': 0.996, 'clip_grad': 2.0, 
                'mlp_ratio': 2, 'qkv_bias': False, 'drop': 0., 'attn_drop': 0., 'local_layers':6, 
                'locality_strength': 1, 'depth': 8, 'use_pos_embed': True}

    model, history = main(parameters)
