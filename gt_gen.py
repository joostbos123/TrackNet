import numpy as np
import pandas as pd
import os
import cv2
import argparse

VAL_VIDEOS = ['video4_Padel_i_Halmstad_Stormatch']

def gaussian_kernel(size, variance):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2+y**2)/float(2*variance))
    return g

def create_gaussian(size, variance):
    gaussian_kernel_array = gaussian_kernel(size, variance)
    gaussian_kernel_array =  gaussian_kernel_array * 255/gaussian_kernel_array[int(len(gaussian_kernel_array)/2)][int(len(gaussian_kernel_array)/2)]
    gaussian_kernel_array = gaussian_kernel_array.astype(int)
    return gaussian_kernel_array

def create_gt_images(path_input, path_output, size, variance, width, height):
    gaussian_kernel_array = create_gaussian(size, variance)
    for game in os.listdir(path_input):
        clips = os.listdir(os.path.join(path_input, game))
        for clip in clips:
            print('game = {}, clip = {}'.format(game, clip))

            path_out_game_gts = os.path.join(path_output, 'gts', game)
            if not os.path.exists(path_out_game_gts):
                os.makedirs(path_out_game_gts)
            
            path_out_game_images = os.path.join(path_output, 'images', game)
            if not os.path.exists(path_out_game_images):
                os.makedirs(path_out_game_images)

            path_out_clip_gts = os.path.join(path_output, 'gts', game, clip)    
            if not os.path.exists(path_out_clip_gts):
                os.makedirs(path_out_clip_gts) 
            
            path_out_clip_images = os.path.join(path_output, 'images', game, clip)
            if not os.path.exists(path_out_clip_images):
                os.makedirs(path_out_clip_images)

            path_labels = os.path.join(os.path.join(path_input, game, clip), 'Label.csv')
            labels = pd.read_csv(path_labels)
            for idx in range(labels.shape[0]):
                file_name, vis, x, y, _ = labels.loc[idx, :]
                image = cv2.imread(os.path.join(path_input, game, clip, file_name))
                image_height, image_width, _ = image.shape
                heatmap = np.zeros((height, width, 3), dtype=np.uint8)
                if vis != 0:
                    x = int(x / image_width * width)
                    y = int(y / image_height * height)
                    for i in range(-size, size+1):
                        for j in range(-size, size+1):
                                if x+i<width and x+i>=0 and y+j<height and y+j>=0 :
                                    temp = gaussian_kernel_array[i+size][j+size]
                                    if temp > 0:
                                        heatmap[y+j,x+i] = (temp,temp,temp)

                cv2.imwrite(os.path.join(path_output, 'gts', game, clip, file_name), heatmap)
                cv2.imwrite(os.path.join(path_output, 'images', game, clip, file_name), cv2.resize(image, (width, height)))
                
def create_gt_labels(path_input, path_output, width, height):
    df = pd.DataFrame()
    for game in os.listdir(path_input):
        clips = os.listdir(os.path.join(path_input, game))
        for clip in clips:
            labels = pd.read_csv(os.path.join(path_input, game, clip, 'Label.csv'))
            image = cv2.imread(os.path.join(path_input, game, clip, labels['file name'][0]))
            labels['x-coordinate'] = labels['x-coordinate'] / image.shape[1] * width
            labels['y-coordinate'] = labels['y-coordinate'] / image.shape[0] * height
            labels['gt_path'] = 'gts/' + game + '/' + clip + '/' + labels['file name']
            labels['path1'] = 'images/' + game + '/' + clip + '/' + labels['file name']
            labels_target = labels[2:]
            labels_target.loc[:, 'path2'] = list(labels['path1'][1:-1])
            labels_target.loc[:, 'path3'] = list(labels['path1'][:-2])
            df = pd.concat([df, labels_target], ignore_index=True)
    df = df.reset_index(drop=True) 
    df = df[['path1', 'path2', 'path3', 'gt_path', 'x-coordinate', 'y-coordinate', 'status', 'visibility']]
    df_train = df[df['gt_path'].apply(lambda x: x.split('/')[-3] not in VAL_VIDEOS)]
    df_test = df[df['gt_path'].apply(lambda x: x.split('/')[-3] in VAL_VIDEOS)]
    df_train.to_csv(os.path.join(path_output, 'labels_train.csv'), index=False)
    df_test.to_csv(os.path.join(path_output, 'labels_val.csv'), index=False)  


if __name__ == '__main__':
    PATH_INPUT = "model_training/ball_tracking/datasets/PadelCustom/padel_only"
    PATH_OUTPUT = "model_training/ball_tracking/datasets/PadelCustom/padel_only_processed"
    SIZE = 20
    VARIANCE = 10
    WIDTH = 1280
    HEIGHT = 720   
    
    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)
        
    create_gt_images(PATH_INPUT, PATH_OUTPUT, SIZE, VARIANCE, WIDTH, HEIGHT)
    create_gt_labels(PATH_INPUT, PATH_OUTPUT, WIDTH, HEIGHT)
