from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
import itertools
import subprocess
import glob 
import os
import h5py
import cv2
import io
import pyvista as pv
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.pylab import cm

pv.global_theme.font.size = 26
pv.global_theme.font.label_size = 22
pv.global_theme.font.color = 'black'

def start_xvfb(display : int = 99, is_jupyter : bool = False):
    print(f"Starting pyvista xvfb server for display: ':{display}'") 
    xvfb = subprocess.check_output("ps -ef | grep Xvfb | grep screen", shell=True)
    display = f':{display}'
    if display in str(xvfb):
        os.environ['DISPLAY'] = display
        print(f"Xvfb process was working, using DISPLAY=':{display}')")
    else:
        pv.start_xvfb()
        print(f"Xvfb started, using DISPLAY=':{display}'")
    if is_jupyter:
        pv.set_jupyter_backend('panel')

class Logger():
    def __init__(self,
                 num_classes : int = -1,
                 is_log_3d : bool = False,
                 camera_views : list[int] = [3,5,6,7],
                 use_slicer_colors : bool = True) -> None:
        
        self.classes_num = num_classes
        camera_positions = list(map(list, itertools.product([-1, 1], repeat=3)))
        self.camera_positions = [ camera_positions[i] for i in camera_views]
        self.camera_positions_LR = [[1,0,0],[-1,0,0]]
        self.camera_positions_AP = [[0,1,0],[0,-1,0]]
        self.log_counter = 0
        self.display_id = 99

        if is_log_3d:
            print("Starting pyvista xvfb server") 
            xvfb = subprocess.check_output("ps -ef | grep Xvfb | grep screen", shell=True)
            if f':{self.display_id}' in str(xvfb):
                os.environ['DISPLAY'] = f':{self.display_id}'
                print(f"Xvfb process was working, using DISPLAY=':{self.display_id}'")
            else:
                pv.start_xvfb(display=self.display_id)
                print(f"Xvfb started, using DISPLAY=':{self.display_id}'")
            pv.set_jupyter_backend('panel')
            
        if use_slicer_colors: 
            tooth_colors = pd.read_csv('misc/slicer_33_colormap.txt', delimiter=" ", header=None)
            tooth_colors_df = tooth_colors.iloc[:,2:5]
            tooth_colors_df.columns = ['r', 'g', 'b']
            colorspace = tooth_colors_df.to_numpy()/255

            if self.classes_num == -1:
                self.color_map = colorspace
            else:
                self.color_map = colorspace[:(self.classes_num+1)]
            self.listed_color_map = colors.ListedColormap(self.color_map, 'slicer_colors')
        
        
    def pad_to_square(self, a, pad_value=0):
        new_shape=np.array(3 * [max(a.shape)])
        padded = pad_value * np.ones(new_shape, dtype=a.dtype)
        #trivial padding - without centering
        padded[:a.shape[0], :a.shape[1], :a.shape[2]] = a
        return padded
    
    def symmetric_padding_3d(self, array, target_shape, pad_value=0):
        if len(array.shape) != 3:
            raise ValueError("Input array must be 3-dimensional.")
        
        if all(array.shape == target_shape):
            return array.copy()  # No padding required, return a copy of the original array
        
        padded_array = np.full(target_shape, pad_value, dtype=array.dtype)
        pad_widths = []
        for dim in range(3):
            pad_total = target_shape[dim] - array.shape[dim]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            pad_widths.append((pad_left, pad_right))
        
        padded_array[
            pad_widths[0][0]:target_shape[0]-pad_widths[0][1],
            pad_widths[1][0]:target_shape[1]-pad_widths[1][1],
            pad_widths[2][0]:target_shape[2]-pad_widths[2][1]
        ] = array
        
        return padded_array

    def log_image(self, prediction: torch.tensor, label: torch.tensor, image: torch.tensor) -> np.array:

        x,y,z = prediction, label, image

        if x.shape[0] != x.shape[2]:
            new_shape = np.array(3 * [max(x.shape)])
            x = self.symmetric_padding_3d(x, new_shape, pad_value=0)
            y = self.symmetric_padding_3d(y, new_shape, pad_value=0)
            z = self.symmetric_padding_3d(z, new_shape, pad_value=0)

        #  slice in the middle of scan
        w, h, d = x.shape[0]//2, x.shape[1]//2, x.shape[2]//2
        slices = []

        #labels
        for img in [x, y]:
            w_sl = img[w, :, :]
            h_sl = img[:, h, :]
            d_sl = img[:, :, d]
            if self.classes_num > 1:
                slices.extend([self.color_map[w_sl], self.color_map[h_sl], self.color_map[d_sl]])
            else:
                slices.extend([w_sl, h_sl, d_sl])
        #source image
        w_sl = z[w, :, :]
        h_sl = z[:, h, :]
        d_sl = z[:, :, d]
        slices.extend([w_sl, h_sl, d_sl])

        slices_norm = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1) for img in slices]
        if self.classes_num > 1:
            horizontal_imgs = [cv2.hconcat([slices_norm[0], slices_norm[3], cv2.cvtColor(slices_norm[6], cv2.COLOR_GRAY2RGB)]),
                               cv2.hconcat([slices_norm[1], slices_norm[4], cv2.cvtColor(slices_norm[7], cv2.COLOR_GRAY2RGB)]),
                               cv2.hconcat([slices_norm[2], slices_norm[5], cv2.cvtColor(slices_norm[8], cv2.COLOR_GRAY2RGB)])]
        else:
            horizontal_imgs = [cv2.hconcat([slices_norm[0], slices_norm[3], slices_norm[6]]),
                               cv2.hconcat([slices_norm[1], slices_norm[4], slices_norm[7]]),
                               cv2.hconcat([slices_norm[2], slices_norm[5], slices_norm[8]])]

        img_log = cv2.vconcat(horizontal_imgs)
        return img_log

    def log_image_multitask(self, 
                            distance_map_pred: torch.tensor, distance_map_gt: torch.tensor,
                            seed_map_pred: torch.tensor, seed_map_gt: torch.tensor,
                            seg_map_pred: torch.tensor, seg_map_gt: torch.tensor,
                            input_image: torch.tensor) -> np.array:
        input_data = [distance_map_pred, distance_map_gt, seed_map_pred, seed_map_gt, seg_map_pred, seg_map_gt, input_image]

        #pad to square te be able to concatenate images
        if input_image.shape[0]!=input_image.shape[2]:
            input_data = [self.pad_to_square(input_img) for input_img in input_data]

        w, h, d = input_image.shape[0]//2-15, input_image.shape[1]//2, input_image.shape[2]//2
        slices = []

        for img in input_data:
            w_sl = np.rot90(img[w, :, :])
            h_sl = np.rot90(img[:, h, :])
            d_sl = img[:, :, d]
            slices.extend([w_sl, h_sl, d_sl])

        slices_norm = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1) for img in slices]
        
        column = []
        n_row=3

        for i in range(n_row):
            row = []
            for j in range(len(input_data)):
                row.append(slices_norm[i+n_row*j])
            column.append(row)
            
        img_log = cv2.vconcat([cv2.hconcat(i) for i in column])
        return img_log

    def log_binary(pred : torch.tensor, mask : torch.tensor, image : torch.tensor) -> np.array:
        
        x,y,z = pred, mask, image
        w,h,d = x.shape[0]//2, x.shape[1]//2, x.shape[2]//2

        #prediction
        w_sl = np.rot90(x[w,:,:])
        h_sl = np.rot90(x[:,h,:])
        d_sl = x[:,:,d]
        
        #ground truth
        w_sl_lbl = np.rot90(y[w,:,:])
        h_sl_lbl = np.rot90(y[:,h,:])
        d_sl_lbl = y[:,:,d]

        #source scan
        w_sl_im = np.rot90(z[w,:,:])
        h_sl_im = np.rot90(z[:,h,:])
        d_sl_im = z[:,:,d]

        slices = [w_sl, h_sl, d_sl, w_sl_lbl, h_sl_lbl, d_sl_lbl, w_sl_im, h_sl_im, d_sl_im]
        slices_norm = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1) for img in slices]
        dims = [cv2.hconcat([slices_norm[0],slices_norm[3], slices_norm[6]]),
                cv2.hconcat([slices_norm[1],slices_norm[4], slices_norm[7]]),
                cv2.hconcat([slices_norm[2],slices_norm[5], slices_norm[8]])]
        img_log = cv2.vconcat(dims)     
        return img_log  
        # experiment.log_image(img_log, name=f'{epoch:03}_{batch_idx:02}')

    def log_scene(self, volume: np.array, num_classes: int = -1, add_volume_outline=False, scene_size: int = 480, is_zoom: bool = False, val: Optional[Union[str, int]] = 2.0, view: int = 3) -> np.array:
        
        scene_size = (scene_size,) * 2
        zoomed = None
        labels = dict(xlabel='R', ylabel='P', zlabel='S')

        if num_classes == -1:
            num_classes = self.classes_num
        data = pv.UniformGrid()

        data.dimensions = np.array(volume.shape) + 1
        data.cell_data['values'] = volume.ravel(order='F')
        tresh_data = data.threshold(1, scalars='values')

        p = pv.Plotter(window_size=scene_size, off_screen=True, lighting='three lights')
        p.set_background('#c1c3e8', top='#7579be')
        # p.enable_shadows()
        p.add_axes(line_width=6, ambient=0.5, **labels)

        sargs = dict(
            title='tooth_class',
            title_font_size=16,
            label_font_size=12,
            shadow=True,
            n_labels=self.classes_num+1,
            italic=False,
            fmt="%.0f",
            font_family="arial",
            )

        p.add_mesh(tresh_data, cmap=self.listed_color_map, scalars="values", clim=[-.5, num_classes + 0.5], 
                   scalar_bar_args=sargs, smooth_shading=False)

        # ANGLED VIEWS FROM LIST
        views = []
        p.camera.zoom(1.0)
        if add_volume_outline:
            #bounds, center, faces - EMPTY, points - vertices
            outline = data.outline()
            p.add_mesh(outline, color="k")
        for camera_pos in self.camera_positions:
            p.camera_position = camera_pos
            views.append(p.screenshot(return_img=True))

        # LR VIEWS
        for camera_pos in self.camera_positions_LR:
            p.camera_position = camera_pos
            views.append(p.screenshot(return_img=True))

        # AP VIEWS
        for camera_pos in self.camera_positions_AP:
            p.camera_position = camera_pos
            views.append(p.screenshot(return_img=True))

        out_image = cv2.vconcat(
            [cv2.hconcat([views[0], views[1], views[4], views[5]]), cv2.hconcat([views[2], views[3], views[6], views[7]])])

        # ZOOMED CHOSEN VIEW
        if is_zoom:
            # edges = tresh_data.extract_feature_edges(0)
            # p.add_mesh(edges, color="black", line_width=1)
            p.camera_position = self.camera_positions[view]
            p.camera.zoom(val)
            zoomed = p.screenshot(return_img=True)
            return out_image, zoomed

        return out_image

    def log_3dscene_comp(self, volume: np.array, volume_gt: np.array, num_classes: int = -1, scene_size: int = 480, camera_pos : list = [0,-1,0]) -> np.array:
            
            # solve issue with too many plots for xvbf buffor
            if self.log_counter >= 200:
                print("There are 200 3d logs buffered - it wont fit anymore - from now on will not log 3d meshes.")
                return np.zeros(shape=(scene_size[0]*2, scene_size[0]*2, 3), dtype=np.uint8)
                # self.display_id -=1
                # start_xvfb(display = self.display_id)
            else:
                self.log_counter +=1
            
            scene_size = (scene_size,) * 2
            labels = dict(xlabel='R', ylabel='P', zlabel='S')

            if num_classes == -1:
                num_classes = self.classes_num-1
            
            data = pv.UniformGrid()
            data_gt = pv.UniformGrid()

            #prediction
            data.dimensions = np.array(volume.shape) + 1
            data.cell_data['values'] = volume.ravel(order='F')
            tresh_data = data.threshold(1, scalars='values')

            #ground_truth
            data_gt.dimensions = np.array(volume_gt.shape) + 1
            data_gt.cell_data['values'] = volume_gt.ravel(order='F')
            tresh_data_gt = data_gt.threshold(1, scalars='values')

            #plotter
            p = pv.Plotter(window_size=scene_size, off_screen=True, lighting='three lights')
            p.set_background('#c1c3e8', top='#7579be')
            p.add_axes(line_width=6, ambient=0.5, **labels)

            sargs = dict(
                title='inferior alveolar nerve',
                title_font_size=16,
                label_font_size=12,
                shadow=True,
                n_labels=self.classes_num,
                italic=False,
                fmt="%.0f",
                font_family="arial",
                )
            
            #PLOT SCENES
            # prediction
            if tresh_data.n_points > 0 and tresh_data_gt.n_points > 0:
                pred = p.add_mesh(tresh_data, cmap=self.listed_color_map, scalars="values", clim=[-0.5, num_classes + 0.5], 
                                scalar_bar_args=sargs, smooth_shading=False)

                p.camera_position= camera_pos
                pred_scene_PA = p.screenshot(return_img=True)
                p.camera_position= [-p for p in camera_pos]
                pred_scene_AP = p.screenshot(return_img=True)
                _ = p.remove_actor(pred)
                pred_image = cv2.hconcat([pred_scene_PA, pred_scene_AP])
                
                # ground_truth
                gt = p.add_mesh(tresh_data_gt, cmap=self.listed_color_map, scalars="values", clim=[-0.5, num_classes + 0.5], 
                                scalar_bar_args=sargs, smooth_shading=False)
                p.camera_position= camera_pos
                gt_scenePA = p.screenshot(return_img=True)    
                p.camera_position= [-p for p in camera_pos]
                gt_sceneAP = p.screenshot(return_img=True)
                _ = p.remove_actor(gt)
                gt_image = cv2.hconcat([gt_scenePA, gt_sceneAP])

                out_image = cv2.vconcat([pred_image, gt_image])
                return out_image
            else:
                print("Empty meshes cannot be plotted. Input mesh has zero points. Returning zeros array.")
                return np.zeros(shape=(scene_size[0]*2, scene_size[0]*2, 3), dtype=np.uint8)

    
def log_simple(volume: np.array, color_map : colors.ListedColormap, clim : list = [-0.5, 32.5], tresh_background : int = 1, original_volume: np.array = None, draw_volume : bool = True, draw_bounding : bool = False, bounding_margin :float = 0.1, camera_pos : list= None) -> pv.Plotter:

    data = pv.UniformGrid()
    data.dimensions = np.array(volume.shape) + 1
    data.cell_data['values'] = volume.ravel(order='F')
    p = pv.Plotter(window_size=(500,500), off_screen=True, lighting='three lights')

    if draw_bounding:
        p.add_mesh(data.outline(), color="b")

    if draw_volume:
        if original_volume is not None:
            data = pv.UniformGrid()
            data.dimensions = np.array(original_volume.shape) + 1
            data.cell_data['values'] = original_volume.ravel(order='F')
                                            
        tresh_data = data.threshold(tresh_background, scalars='values')
        p.add_mesh(tresh_data, scalars="values", cmap=color_map, clim=clim, smooth_shading=True)
    if camera_pos is not None:
        p.camera_position = camera_pos
    return p

def get_img_from_fig(fig, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def log_gradient_map(gt, pred, angle_errors, binary_mask_vector, mask):

    idx=0
    fig, ax = plt.subplots(nrows=2, ncols=4,figsize=(18,10))
    ax[0,0].imshow(pred[idx,1,:,:,64].detach().cpu()*mask[idx,0,:,:,64].cpu(),interpolation='none',)
    ax[0,1].imshow(pred[idx,2,:,:,64].detach().cpu()*mask[idx,0,:,:,64].cpu(),interpolation='none',)
    ax[0,2].imshow(pred[idx,0,:,64,:].detach().cpu()*mask[idx,0,:,64,:].cpu(),interpolation='none',)
    im1=ax[0,3].imshow((angle_errors[idx].detach().cpu()*binary_mask_vector[idx].detach().cpu()).reshape(gt.shape[2:])[:,:,64], interpolation='none')
    fig.colorbar(im1,ax=ax[0,3])
    ax[1,0].imshow(gt[idx,1,:,:,64].cpu(),interpolation='none',)
    ax[1,1].imshow(gt[idx,2,:,:,64].cpu(),interpolation='none',)
    ax[1,2].imshow(gt[idx,0,:,64,:].cpu(),interpolation='none',)
    im2=ax[1,3].imshow((angle_errors[idx].detach().cpu()*binary_mask_vector[idx].detach().cpu()).reshape(gt.shape[2:])[:,64,:], interpolation='none')
    fig.colorbar(im2,ax=ax[1,3])
    fig.tight_layout()
    # fig.savefig(f'test{idx}.png', dpi=600) 
    img = get_img_from_fig(fig)
    return img

def log_angles(pred, gt, mask, idx=0):
    
    s = pred.shape[2]//2

    x_pred = pred[idx,1,:,:,s]*mask[idx,0,:,:,s]
    y_pred = pred[idx,2,:,:,s]*mask[idx,0,:,:,s]
    z_pred = pred[idx,0,:,s,:]*mask[idx,0,:,s,:]
    preds = cv2.hconcat([x_pred, y_pred, np.rot90(z_pred)])
    if gt is not None:
        x_gt = gt[idx,1,:,:,s]
        y_gt = gt[idx,2,:,:,s]
        z_gt = gt[idx,0,:,s,:]
        gts = cv2.hconcat([x_gt, y_gt, np.rot90(z_gt)])
        output = cv2.vconcat([preds, gts])
        output_norm = (output+1)/2
        return cm.viridis(output_norm)[:,:,:3]
    else:
        output_norm = (preds+1)/2
        return cm.viridis(output_norm)[:,:,:3]
    
if __name__ == "__main__":
    h5_files = sorted(glob.glob(os.path.join('data/china', "scans_h5_8class/*.h5"), recursive=False))
    label = []
    logger = Logger()

    # HXWXD
    with h5py.File(h5_files[0], 'r') as f:
            label = (f['label'][:])
    
    #simulate batch
    label = np.expand_dims(label, axis=0)
    label = torch.from_numpy(label)

    img = logger.log_image(label, label, label)
    plt.imsave('test.png',img)



