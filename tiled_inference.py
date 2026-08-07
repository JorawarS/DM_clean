import torch
import data as Data
import model as Model
import argparse
import PIL.Image as Image
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import time
'''In the config file, keep the r_resolution(also tile size) = diffusion model input size. The tiler will automatically calculate 
the number of tiles needed to cover the original image.
 The tiler will also merge the tiles back together after processing.
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/underwater_tiled.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger('base', opt['path']['log'], 'val', level=logging.INFO,screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')
    diffusion.set_new_noise_schedule(
    opt['model']['beta_schedule']['val'], schedule_phase='val')

    # Create results directory if it doesn't exist
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True) 

    #Create guassian kernel for merging tiles
    tile_size = opt['datasets']['val']['r_resolution']
    weight_matrix = Data.util.generate_gaussian_weight_map(tile_size=tile_size, sigma=None, device='cpu')


    #Load paths of images
    img_paths = Data.util.get_paths_from_images(opt['datasets']['val']['dataroot'])
    for i, img_path in enumerate(img_paths):


        # Load image into our custom class TiledImage
        tiled_image = Data.create_tiled_dataset(img_path, opt['datasets']['val'], 'val', tiler_opt=opt['tiler'])
        tile_loader = Data.create_dataloader(tiled_image, opt['datasets']['val'], 'val')

        #Initialise final image tensor
        enhanced_image = torch.zeros((3,tiled_image.image_height, tiled_image.image_width), device='cpu')
        weight_sum = torch.zeros((1,tiled_image.image_height, tiled_image.image_width), device='cpu')
        original_image = tiled_image.image

        print(f"Processing image {i+1}/{len(img_paths)}: {img_path}")

        for idx, tile in enumerate(tile_loader):
            # Process each tile with the model
            diffusion.feed_data(tile)
            start_time = time.time()
            diffusion.test(continous=True)
            end_time = time.time()
            print(f"Time taken to process tile {idx+1} for image {i+1}: {end_time - start_time:.2f} seconds")
            visuals = diffusion.get_current_visuals(need_LR=False)

            # Merge the processed tile into the final image
            enhanced_tile = visuals['SR'][-1]  # the last element is the final output of the diffusion model
            weighted_tile = enhanced_tile * weight_matrix # Apply the weight matrix to the enhanced tile
            x1, y1, x2, y2 = tiled_image.boxes[idx] #get the coordinates of the tile in the original image
            enhanced_image[:, y1:y2, x1:x2] += weighted_tile # Add the weighted tile to the final image
            weight_sum[:, y1:y2, x1:x2] += weight_matrix # Add the weight matrix to the weight sum
        final_enhanced_image = enhanced_image / weight_sum #Normalize the image to fix the brightness and contrast issues caused by overlapping tiles
        final_enhanced_image = Metrics.tensor2img(final_enhanced_image) 
        Metrics.save_img(final_enhanced_image, '{}/{}_enhanced.png'.format(result_path, os.path.splitext(os.path.basename(img_path))[0])) 
        original_image.save('{}/{}_original.png'.format(result_path, os.path.splitext(os.path.basename(img_path))[0]))




    tb_logger.close()




