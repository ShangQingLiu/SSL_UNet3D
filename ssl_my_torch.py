# Before everythings
# import comet_ml



from random import shuffle
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import SimpleITK as sitk
import torch.nn as nn
from medpy.io import load
import torchio as tio
import nibabel as nib
import os
import numpy
from pytorch3dunet.datasets.utils import SliceBuilder 

import pytorch_lightning as pl
import pytorch3dunet.unet3d.model as model3dUnet
import torch.nn.functional as F
import torch


from torch.utils.data import random_split,DataLoader 
import torchio.transforms as transforms 
from pytorch_lightning.loggers import CometLogger
from torch.utils.tensorboard import SummaryWriter




def read_text_file(filename):
    lines = []
    # print(f"start to read text file:{filename}")
    with open(filename, 'r') as file:
        for line in file: 
            # print(f"line:{line}")
            line = line.strip() #or some other preprocessing
            lines.append(line)
    # print(f"length of text file:{len(lines)}")
    return lines

def nii_read(filename):
    # print(f"nii_resad:{filename}")
    # voxel_obj =  sitk.ReadImage(filename)
    # print(voxel_obj)
    # voxel_arr = sitk.GetArrayFromImage(voxel_obj) 
    # print(voxel_arr.shape)
    voxel_arr,voxel_header = load(filename)
    voxel_arr = numpy.swapaxes(voxel_arr,0,2)
    return voxel_arr
########
# # test nii_read code
# import numpy
# import torchio.transforms as transforms
# test_data_file = "./data/CHAOST2/chaos_MR_T2_normalized/image_1.nii.gz"
# vox_arr = nii_read(test_data_file)
# print(vox_arr.shape)
# swap_ax =numpy.swapaxes(vox_arr,0,2)
# cut_depth = swap_ax[:31,:,:]
# add_di = cut_depth[None,:]
# print(add_di.shape)
# result = torch.as_tensor(add_di)
# print(result.shape)

#########

class SSLDataset(Dataset):
    def __init__(self, root_dir, sup_file, sup_label_file, qry_file, qry_label_file, params):
        self.root_dir = root_dir
        self.sup_file = read_text_file(os.path.join(self.root_dir, sup_file))
        self.sup_label_file = read_text_file(os.path.join(self.root_dir , sup_label_file))
        self.qry_file = read_text_file(os.path.join(self.root_dir , qry_file))
        self.qry_label_file = read_text_file(os.path.join(self.root_dir , qry_label_file))
        self.same_depth =params["same_depth"]
        self.adaptive_size = params["adaptive_size"]
        assert len(self.sup_file) == len(self.sup_label_file), "lengths of image files and fixation files do not match!"

    def __len__(self):
        return len(self.sup_file)

    def __getitem__(self, idx):
        
        sup_img = self.compose_action(self.sup_file[idx])
        sup_label_fg = self.compose_action(self.sup_label_file[idx],type='label')
        sup_label_bg = torch.subtract(torch.ones_like(sup_label_fg), sup_label_fg)

        
        #qry_img = self.compose_transformation(self.qry_file[idx])
        #qry_label_fg = self.compose_transformation(self.qry_label_file[idx])
        
    
        qry_img = self.compose_transformation(self.sup_file[idx])
        qry_label_fg = self.compose_transformation(self.sup_label_file[idx],type='label')
        qry_label_bg = torch.subtract(torch.ones_like(qry_label_fg),  qry_label_fg)

        sample = {"sup_img":sup_img, "sup_fg_label":sup_label_fg, "qry_img":qry_img, "qry_label_fg":qry_label_fg,
         "sup_bg_label":sup_label_bg, "qry_label_bg":qry_label_bg}

        return sample

   

    def compose_action(self, file_name, type='img'):

        data = self.nii_read_and_cut(file_name)
        if type=='label':
            cls_num = numpy.unique(data)
            cls_num = numpy.delete(cls_num,0) 
            cls =numpy.random.choice(cls_num.shape[0], 1, replace=False)[0]
            zero_data = numpy.zeros_like(data)
            zero_data[numpy.where(data==cls)] = 1
            data = zero_data
        # read data as numpy array 3d
        

        # Add another dummpy dimention for channel as 1
        data = data[None,:]
        # [1,same_depth, 256,256]

        # To tensor
        data = torch.as_tensor(data,dtype=torch.double)

        # Avgpool
        # ap = nn.AvgPool3d((2, 2, 1), stride=(2, 2, 2))
        ap = nn.AdaptiveAvgPool3d(self.adaptive_size)
        # ap = nn.AdaptiveAvgPool3d((None,64,64))
        data = ap(data)
        # [1, same_depteh, 128, 128]

        return data


    def compose_transformation(self, file_name,type='img'):

        data = sitk.ReadImage(file_name)

        elastic_tr = tio.RandomElasticDeformation(num_control_points=7,locked_borders=2)
        affine_tr = tio.transforms.RandomAffine(scales=(0.95, 1.05),degrees=(1,1,1))
        gamma_tr = tio.RandomGamma(log_gamma=(-0.3,0.3))

        ts = tio.Compose([gamma_tr,affine_tr,elastic_tr])

        transformed = ts(data)
        array = sitk.GetArrayFromImage(transformed)
        # sitk.WriteImage(transformed,"transformed_data.nii.gz")
# To tensor

        if type=='label':
            cls_num = numpy.unique(data)
            cls_num = numpy.delete(cls_num,0) 
            cls =numpy.random.choice(cls_num.shape[0], 1, replace=False)[0]
            zero_data = numpy.zeros_like(data)
            zero_data[numpy.where(data==cls)] = 1
            data = zero_data

        array = array[None,:]
        data = torch.as_tensor(array,dtype=torch.double)        


        # Avgpool
        # ap = nn.AvgPool3d((2, 2, 1), stride=(2, 2, 2))
        ap = nn.AdaptiveAvgPool3d(self.adaptive_size)
        # ap = nn.AdaptiveAvgPool3d((None,64,64))
        data = ap(data)
        # [1, same_depteh, 128, 128]

        return data
        
    
    def nii_read_and_cut(self, file_name):

        # numpy.array 3D
        data = nii_read(filename=file_name)
        # cut depth the same as same_depth
        data = data[:self.same_depth,:,:]

        return data
    
    
class Diceloss(torch.nn.Module):
    def init(self):
        super(DiceLoss, self).init()
    def forward(self,pred, target):
       smooth = 1.
       iflat = pred.contiguous().view(-1)
       tflat = target.contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class SSL3DUNet(torch.nn.Module):
    def __init__(self,  params):
        super().__init__()
        self.learning_rate = params["lr"]
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCEWithLogitsLoss()
        self.thresh = .95
        unet3d = model3dUnet.UNet3D(1,1, f_maps=params["f_maps"])
        self.unet3d = unet3d.double()
        self.unet3d.final_conv = Identity() 
        # self.params = list(unet3d.parameters())
        self.params = unet3d.parameters()
        self.automatic_optimization = False # use own optimizer
        # self.save_hyperparameters() # for comet.ML logger
        print("Finish init SSL3DUNet Model")
    # def forward(batch, batch_idx):
    #     return batch 


    def forward(self,batch):
        # batch: {"sup_img":sup_img, "sup_label":sup_label, "qry_img":qry_img, "qry_label":qry_label}
        
        sup_img = batch["sup_img"] 
        qry_img = batch["qry_img"] 

        sup_fts = self.unet3d(sup_img) 
        qry_fts = self.unet3d(qry_img) 
        predictions = {"sup_fts":sup_fts, "qry_fts":qry_fts} 
        return predictions

    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW( self.params, lr=self.learning_rate)
    #     #################
    #     # pretrained_wg_pth = "best_checkpoint.pytorch" # from https://github.com/wolny/pytorch-3dunet
    #     # state = torch.load(pretrained_wg_pth)
    #     # optimizer.load_state_dict(state['optimizer_state_dict']) 
    #     ###############
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches)
    #     # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.1)
    #     return [optimizer], [scheduler]
    
def _pass_compose(sup_fts, qry_fts, sup_fg_label, sup_bg_label, qry_label_fg, qry_label_bg,params, thresh=.95):
    
    raw_score_bg = _alp_module(qry_fts,sup_fts,sup_bg_label,params,mode='local',thresh=thresh)
    NO_OVER_THRES =  torch.all(torch.flatten(sup_fg_label < thresh))
    NO_OVER_THRES = False 
    raw_score_fg = _alp_module(qry_fts,sup_fts,sup_fg_label,params,mode='global' if not NO_OVER_THRES else 'mask',thresh=thresh)
    qry_pred = torch.div(torch.add(raw_score_bg, raw_score_fg),2)

    qry_pred_fg = torch.clone(qry_pred)
    qry_pred_bg = torch.subtract(torch.ones_like(qry_pred_fg),  qry_pred_fg)
    # print(f"qry_pred_bg shape:{qry_pred_bg.shape}")
    # print(f"qry_pred_fg shape:{qry_pred_fg.shape}")
    
    # print(f"label fg shape:{qry_label_fg.shape}")
    # print(f"label bg shape:{qry_label_bg.shape}")
    re_raw_score_bg = _alp_module(sup_fts,qry_fts,qry_pred_fg,params,mode='local',thresh=thresh)
    RES_NO_OVER_THRES =  torch.all(torch.flatten(qry_pred_bg < thresh))
    RES_NO_OVER_THRES = False 
    re_raw_score_fg = _alp_module(sup_fts,qry_fts,qry_pred_bg,params,mode='global' if not RES_NO_OVER_THRES else 'mask',thresh=thresh)
    sup_pred = torch.div(torch.add(re_raw_score_bg, re_raw_score_fg),2)

    return qry_pred, sup_pred

def _alp_module(qry_fts, sup_fts, sup_label,params=None, mode='global', thresh=.95):
    avgPool = nn.AdaptiveAvgPool3d((int(params["proto_grid_size"]/2),params["proto_grid_size"],params["proto_grid_size"]))  # None is same as input
    # [1, same_depteh, 16, 16]
    if mode == 'local':
        
        ch = qry_fts.shape[1] # temp: 64

        n_sup_x = avgPool(sup_fts)
        sup_nshot = sup_fts.shape[0] # 1
        # print(f"sup_nshot:{sup_nshot}")

        n_sup_x = n_sup_x.view(sup_nshot,ch,-1).permute(0,2,1).unsqueeze(0)
        n_sup_x = n_sup_x.reshape(1,-1,ch).unsqueeze(0)

        sup_y = avgPool(sup_label) 
        sup_y = sup_y.view(sup_nshot,1,-1)
        sup_y = sup_y.permute(1,0,2).view(1,-1).unsqueeze(0)
        # print(f"sup_y shape:{sup_y.shape}")
        # print(f"n_sup_x shape:{n_sup_x.shape}")
        # print(f"thresh:{thresh}")
        pro_n = n_sup_x[sup_y > thresh,:]*1

        qry_n = _safe_norm(qry_fts) 
        # print(f"qry_n shape:{qry_n.shape}")
        # print(f"pro_n shape:{pro_n.shape}")
        dists = F.conv3d(qry_n,pro_n[...,None,None,None]) * 20
        pred_grid =  torch.sum(F.softmax(dists,dim=1)*dists, dim=1,keepdim=True)
        return pred_grid
        
    elif mode == 'global':
        
        ch = qry_fts.shape[1] # temp: 64

        n_sup_x = avgPool(sup_fts)
        sup_nshot = sup_fts.shape[0] # 1

        n_sup_x = n_sup_x.view(sup_nshot,ch,-1).permute(0,2,1).unsqueeze(0)
        n_sup_x = n_sup_x.reshape(1,-1,ch).unsqueeze(0)

        sup_y = avgPool(sup_label) 
        sup_y = sup_y.view(sup_nshot,1,-1)
        sup_y = sup_y.permute(1,0,2).view(1,-1).unsqueeze(0)

        pro_n = n_sup_x[sup_y > thresh,:]*1

        ##################
        # global
        glb_proto = torch.sum(sup_fts * sup_label, dim=(-1,-2,-3)) / (sup_label.sum(dim=(-1,-2,-3)) + 1e-5)
        pro_n = _safe_norm(torch.cat([pro_n, glb_proto], dim=0))
        qry_n = _safe_norm(qry_fts) 
        ##############

        dists = F.conv3d(qry_n,pro_n[...,None,None,None]) * 20
        pred_grid =  torch.sum(F.softmax(dists,dim=1)*dists, dim=1,keepdim=True)
        
        return pred_grid
        
    elif mode == 'mask':
        proto = torch.sum(sup_fts * sup_label, dim=(-1,-2,-3)) / (sup_label.sum(dim=(-1,-2,-3)) + 1e-5)
        proto = proto.mean(dim=0, keepdim=True)
        pred_mask = F.cosine_similarity(qry_fts, proto[..., None,None], dim=1, eps=1e-4) * 20

        return pred_mask.unsqueeze(1)
    else:
        pass

def _safe_norm( input, p=2, dim = 1, eps = 1e-4):
    x_norm = torch.norm(input,p=p,dim=dim)
    x_norm = torch.max(x_norm,torch.ones_like(x_norm,requires_grad=False)*eps )
    output = input.div(x_norm.unsqueeze(1).expand_as(input))
    return output

def MyLoss( sup_fg_label,sup_bg_label,qry_label_fg,qry_label_bg, qry_fts, sup_fts,params, writer:SummaryWriter):


    criterion = Diceloss()

    qry_pred, sup_pred = _pass_compose(sup_fts, qry_fts, sup_fg_label, sup_bg_label, qry_label_fg, qry_label_bg,params)
    query_loss = criterion((qry_pred),(qry_label_fg))
    sup_loss = criterion((sup_pred),(sup_fg_label))

    loss = torch.add(query_loss , sup_loss )
    return loss,sup_loss, query_loss




# # Create an experiment with your api key
# experiment = comet_ml.Experiment(
#     api_key="9ocfckVqiaW61cZMkTd0iW0OH",
#     project_name="ssl",
# )


from pytorch_lightning.loggers import TensorBoardLogger
import GPUtil
from os.path import exists
from torch.utils.data.sampler import SubsetRandomSampler

if __name__ == '__main__':
    writer = SummaryWriter()
    logger = TensorBoardLogger("tb_logs", name="my_model")

    base_dir ="/homeL/1sliu/code/SSL_UNet3D/data/CHAOST2/chaos_MR_T2_normalized"
    sup_img_re = "image*.nii.gz" 
    sup_label_re = "superpix3d*.nii.gz" 
    qry_img_re = "image*.nii.gz" 
    qry_label_re = "label*.nii.gz"

    sup_img_file = "sup_img_file.txt"
    sup_label_file = "sup_label_file.txt"
    qry_img_file = "qry_img_file.txt"
    qry_label_file = "qry_label_file.txt"

    pretrained_wg_pth = "best_checkpoint.pytorch" # from https://github.com/wolny/pytorch-3dunet

    torch.manual_seed(0) # set default seed for torch operation
     
    data_param = { "adaptive_size":(24, 128,128), "same_depth":24 }
    model_param = {"lr":1e-4, "proto_grid_size": 8, "f_maps":32}

    dataset = SSLDataset(base_dir,sup_img_file,sup_label_file,qry_img_file,qry_label_file, data_param)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSL3DUNet(model_param)
    model_load_path = "save_model/saved_model_1000.pth"
    if exists(model_load_path):
        model.load_state_dict(torch.load(model_load_path),strict=False) 
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model,device_ids=[0,1])
    model.to(device)


    # tune the classifier parameters
    params_to_update = model.parameters()
    print("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    opt = torch.optim.AdamW( params_to_update, lr=model_param["lr"])
    lambda1 = lambda epoch: 0.65**epoch 

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=1e-3, total_steps=60000)


    print("strat trainning")
    epochs = 20000
    min_valid_loss = numpy.inf
    val_loss = 0
    k_fold = 5
    for epoch in range(epochs):
        writer.add_scalar("epoch",epoch,epoch)

        # cross valid 
        seg = epoch % k_fold
        total_data_size = len(dataset)
        indices = list(range(total_data_size))
        base = int(numpy.floor(1/k_fold * total_data_size))
        split_start = base * seg 
        split_end = base* (seg+1) 
        train_sampler = SubsetRandomSampler(indices[:split_start]+indices[split_end:])
        valid_sampler = SubsetRandomSampler(indices[split_start:split_end])

        train, val = random_split(dataset,[16,4])
        train_loader = DataLoader(train,batch_size=1,num_workers=24,shuffle=True,persistent_workers=True)
        val_loader = DataLoader(val,batch_size=1,num_workers=24,persistent_workers=True)


        # TRAINING
        train_loss_sum = 0
        train_count = 0
        model.train()
        for batch in train_loader:
            sup_fg_label = batch["sup_fg_label"].to(device)
            sup_bg_label = batch["sup_bg_label"].to(device)
            qry_label_fg = batch["qry_label_fg"].to(device)
            qry_label_bg = batch["qry_label_bg"].to(device)
            sup_img = batch["sup_img"].to(device)
            qry_img = batch["qry_img"].to(device)

            # GPUtil.showUtilization()
            prediction = model(batch) 

            train_loss,sup_loss,query_loss = MyLoss(sup_fg_label,sup_bg_label,
            qry_label_fg,qry_label_bg, prediction["qry_fts"],prediction["sup_fts"],model_param,writer) 

            opt.zero_grad()
            train_loss.backward()
            writer.add_scalar('loss/train', train_loss,epoch)
            writer.add_scalar("Accuracy/train_query_loss", query_loss,epoch)
            writer.add_scalar("Accuracy/train_sup_loss", sup_loss,epoch)
            train_loss_sum += train_loss.item()
            opt.step()
            train_count +=1

        # writer.add_scalar('avg Loss/train', train_loss_sum/train_count, epoch)

        #VALIDATION
        #model.train(False)
        if epoch %10 == 0:
            val_loss_sum = 0
            model.eval()
            val_count = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_sup_fg_label = val_batch["sup_fg_label"].to(device)
                    val_sup_bg_label = val_batch["sup_bg_label"].to(device)
                    val_qry_label_fg = val_batch["qry_label_fg"].to(device)
                    val_qry_label_bg = val_batch["qry_label_bg"].to(device)
                    val_sup_img = val_batch["sup_img"].to(device)
                    val_qry_img = val_batch["qry_img"].to(device)

                    val_prediction = model(val_batch) 

                    val_loss,val_sup_loss,val_query_loss = MyLoss(val_sup_fg_label,val_sup_bg_label,val_qry_label_fg,val_qry_label_bg, val_prediction["qry_fts"],val_prediction["sup_fts"],model_param,writer) 
                    writer.add_scalar("Accuracy/val_query_loss", val_query_loss,epoch)
                    writer.add_scalar("Accuracy/val_sup_loss", val_sup_loss,epoch)
                    writer.add_scalar('loss/valid', val_loss,epoch)
                    val_loss_sum += val_loss.item()
                    val_count +=1
                # writer.add_scalar('avg Loss/valid', val_loss_sum/val_count, epoch)
    
        # update learning rate
        scheduler.step()
        if epoch % 200 == 0:
        # if min_valid_loss > val_loss:
        #     min_valid_loss = val_loss
            # Saving State Dict
            model_pth = 'save_model/saved_model_'+ str(epoch)+'.pth'
            print(f"saving model: {model_pth}")
            torch.save(model.state_dict(), model_pth)

        if min_valid_loss > val_loss:
            min_valid_loss = val_loss
            # Saving State Dict
            best_model_pth = 'save_model/best_model_'+ str(epoch)+'.pth'
            print(f"saving model: {best_model_pth}")
            torch.save(model.state_dict(), best_model_pth)
    print("end trainning")

    writer.flush()
    writer.close()
######################################################################################################## STOP HERE ########################################################