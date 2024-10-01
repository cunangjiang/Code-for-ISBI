import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from utils import calc_psnr, convert_rgb_to_y, denormalize
# from utils import save_imgs
from torchvision import transforms
import os


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        ori, ref, ref_lr, tar, img_name = data
        ori, ref, ref_lr, tar = ori.cuda(non_blocking=True).float(), ref.cuda(non_blocking=True).float(), ref_lr.cuda(non_blocking=True).float(), tar.cuda(non_blocking=True).float()

        # Forward pass
        sr_output = model(tar, ref)
        
        # Compute losses for segmentation and super-resolution (or other tasks if needed)
        loss = criterion(sr_output, ori)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step



def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    val_psnr = 0
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            ori, ref, ref_lr, tar, img_name = data
            ori, ref, ref_lr, tar = ori.cuda(non_blocking=True).float(), ref.cuda(non_blocking=True).float(), ref_lr.cuda(non_blocking=True).float(), tar.cuda(non_blocking=True).float()
            # Forward pass
            sr_output = model(tar, ref)

            # Compute loss
            loss = criterion(sr_output, ori)
            loss_list.append(loss.item())

            sr_output_y = convert_rgb_to_y(denormalize(sr_output.squeeze(0)), dim_order='chw')
            ori = convert_rgb_to_y(denormalize(ori.squeeze(0)), dim_order='chw')

            psnr_val = calc_psnr(ori, sr_output_y)
            val_psnr += psnr_val

            # sr_img = sr_output.squeeze(0).cpu()
            # sr_img = torch.clamp(sr_img, min=-1, max=1)
            # sr_img = (sr_img + 1) / 2.0
            # sr_img = transforms.ToPILImage()(sr_img)
            # sr_img.save(os.path.join(config.work_dir, 'outputs', img_name[0]))

    if epoch % config.val_interval == 0:
        avg_psnr = val_psnr / len(test_loader)

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, psnr: {avg_psnr:.4f}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list), avg_psnr



def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None,):
    # switch to evaluate mode
    model.eval()
    val_psnr = 0
    best_score = float('-inf')
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            ori, ref, ref_lr, tar, img_name = data
            ori, ref, ref_lr, tar = ori.cuda(non_blocking=True).float(), ref.cuda(non_blocking=True).float(), ref_lr.cuda(non_blocking=True).float(), tar.cuda(non_blocking=True).float()
            # Forward pass
            sr_output = model(tar, ref)

            # Compute losses for segmentation and super-resolution
            loss = criterion(sr_output, ori)
            loss_list.append(loss.item())

            loss_list.append(loss.item())

            sr_output_y = convert_rgb_to_y(denormalize(sr_output.squeeze(0)), dim_order='chw')
            ori = convert_rgb_to_y(denormalize(ori.squeeze(0)), dim_order='chw')

            psnr_val = calc_psnr(ori, sr_output_y)
            val_psnr += psnr_val
            
            # if i % config.save_interval == 0:
            sr_img = sr_output.squeeze(0).cpu()
            sr_img = torch.clamp(sr_img, min=-1, max=1)
            sr_img = (sr_img + 1) / 2.0
            sr_img = transforms.ToPILImage()(sr_img)
            sr_img.save(os.path.join(config.work_dir, 'outputs', img_name[0]))

        avg_psnr = val_psnr / len(test_loader)

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f}, psnr: {avg_psnr:.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
