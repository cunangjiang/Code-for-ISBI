import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import BraTs_datasets
from tensorboardX import SummaryWriter
from model.multisup import WavMCVM

from engine import *
import sys

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger) 





    print('#----------GPU init----------#')
    set_seed(config.seed)
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    train_dataset = BraTs_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers)
    val_dataset = BraTs_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)





    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'wavmcvm':
        model = WavMCVM(
            upscale=model_cfg['upscale'],
            inchans=model_cfg['inchans'],
            outchans=model_cfg['outchans'],
            dim=model_cfg['dim'],
            depth=model_cfg['depths'],
            d_state=model_cfg['d_state'],
            drop=model_cfg['drop_rate'],
            attn_drop=model_cfg['attn_drop_rate'],
            drop_path=model_cfg['drop_path'],
            norm_layer=model_cfg['norm_layer'],
            patch_size=model_cfg['patch_size'],
            patch_norm=model_cfg['patch_norm'],
        )
    elif config.network == 'minet':
        model = MINet(scale=model_cfg['upscale'], n_resgroups=2,n_resblocks=2, n_feats=64)
    elif config.network == 'mcmrsr':
        model = McMRSR(upscale=model_cfg['upscale'], img_size=(112, 112),
                   window_size=8, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2)
    elif config.network == 'swinir':
        model = SwinIR(upscale=model_cfg['upscale'], in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    elif config.network == 'wavtrans':
        model = WavTrans(upscale=model_cfg['upscale'], img_size=(56, 56),
                   window_size=8, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2)
    elif config.network == 'srcnn':
        model = SRCNN(upscale=model_cfg['upscale'])
    elif config.network == 'edt':
        model = EDT(upscale=model_cfg['upscale'])
    elif config.network == 'hat':
        model = HAT(img_size=64, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upscale=model_cfg['upscale'])
    elif config.network == 'rdn':
        model = RDN(scale_factor=model_cfg['upscale'], num_channels=3, num_features=64, growth_rate=64, num_blocks=8, num_layers=4)
    elif config.network == 'edsr':
        model = EDSR(scale_factor=model_cfg['upscale'])
    else: raise Exception('network in not right!')
    model = model.cuda()





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    max_psnr = float('-inf')





    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, max_psnr, min_epoch, loss = checkpoint['min_loss'], checkpoint['max_psnr'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, max_psnr: {max_psnr:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)




    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )


        # loss = test_one_epoch(
        #         val_loader,
        #         model,
        #         criterion,
        #         logger,
        #         config,
        #     )

        loss, avg_psnr = val_one_epoch(
                val_loader,
                model,
                criterion,
                epoch,
                logger,
                config
            )

        if loss < min_loss:
            # torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss

        if avg_psnr > max_psnr:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            max_psnr = avg_psnr
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'max_psnr': max_psnr,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
                val_loader,
                model,
                criterion,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )      


if __name__ == '__main__':
    config = setting_config
    main(config)
