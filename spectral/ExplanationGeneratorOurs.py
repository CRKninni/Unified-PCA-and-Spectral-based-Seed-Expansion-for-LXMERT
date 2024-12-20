import numpy as np
import torch
import copy
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eig

import torch.nn.functional as F
from torch.nn import Linear

from pymatting.util.util import row_sum
from scipy.sparse import diags
from scipy.stats import skew
import cv2
from sklearn.decomposition import PCA
import math


from spectral.get_fev import get_eigs, get_grad_eigs, avg_heads, get_grad_cam_eigs

def eig_seed(feats, iters):
    patch_scores_norm = get_eigs(feats, "image", 2)
        
    num_patches = int(np.sqrt(len(patch_scores_norm)))
    heatmap = patch_scores_norm.reshape(num_patches, num_patches)
    
    seed_index = np.argmax(patch_scores_norm)

    # Convert the 1D index to 2D indices
    seed_row = seed_index // num_patches
    seed_col = seed_index % num_patches


    # Initialize a mask for the expanded seed region
    seed_mask = np.zeros_like(heatmap)
    seed_mask[seed_row, seed_col] = 1

    # Define the number of expansion iterations
    num_expansion_iters = iters
    
    # Perform seed expansion
    for _ in range(num_expansion_iters):
        # Find neighboring patches
        neighbor_mask = cv2.dilate(seed_mask, np.ones((3, 3), np.uint8), iterations=1)
        neighbor_mask = neighbor_mask - seed_mask  # Exclude already included patches
        neighbor_indices = np.where(neighbor_mask > 0)
        
        # For each neighbor, decide whether to include it based on similarity
        for r, c in zip(*neighbor_indices):
            # Use heatmap values as similarity scores
            similarity = heatmap[r, c]
            # Define a threshold for inclusion
            threshold = 0.5  # Adjust this value as needed
            
            if similarity >= threshold:
                seed_mask[r, c] = 1  # Include the neighbor
            else:
                seed_mask[r, c] = 0.001

    # Apply the seed mask to the heatmap
    refined_heatmap = heatmap * seed_mask
    
    return refined_heatmap.flatten()
    
def get_pca_component(feats, modality, component=0, device="cpu"):
    if feats.size(0) == 1:
        feats = feats.detach().squeeze()

    if modality == "image":
        n_image_feats = feats.size(0)
        val = int(math.sqrt(n_image_feats))
        if val * val == n_image_feats:
            feats = F.normalize(feats, p=2, dim=-1).to(device)
        elif val * val + 1 == n_image_feats:
            feats = F.normalize(feats, p=2, dim=-1)[1:].to(device)
        else:
            print(f"Invalid number of features detected: {n_image_feats}")
    else:
        feats = F.normalize(feats, p=2, dim=-1)[1:-1].to(device)

    # Reshape feats to apply PCA on (1296, 768) as desired
    feats_reshaped = feats.cpu().detach().numpy()

    # Apply PCA on the reshaped data to get the second principal component
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(feats_reshaped)

    # Extract the second principal component and expand to original shape
    second_pc = principal_components[:, component]

    # Convert to tensor and move to the specified device
    second_pc = torch.tensor(second_pc, dtype=torch.float32).to(device)

    if modality == 'text':
        second_pc = torch.cat( ( torch.zeros(1), second_pc, torch.zeros(1)  ) )


    second_pc = torch.abs(second_pc)
    
    # Normalize the second principal component for visualization
    second_pc_norm = (second_pc - second_pc.min()) / (second_pc.max() - second_pc.min() + 1e-8)
    
    return second_pc_norm

def get_grad_cam(grads,cams,modality):
    final_gradcam = []
    num_layers = len(cams)
    for i in range(num_layers):
        # Get gradients and attention maps for this layer
        grad = grads[i] 
        cam = cams[i]
        
        if modality == "image":
            cam = cam[:, :, :, :]  
            grad = grad[:, :, :, :].clamp(0) 
        elif modality == "text":
            cam = cam[:,:,1:-1,1:-1]
            grad = grad[:,:,1:-1,1:-1].clamp(0)
        else:
            print("Invalid modality")
            return None

        # Multiply gradients by attention maps (element-wise)
        layer_gradcam = cam * grad

        layer_gradcam = layer_gradcam.mean(1) 

        final_gradcam.append(layer_gradcam.cpu())
        
        
    if modality == "image":
        final_gradcam_temp = np.mean(final_gradcam, axis=0)
        final_gradcam_temp = final_gradcam_temp.squeeze()
        
        gradcam = final_gradcam_temp
        heatmap = np.mean(gradcam, axis=0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap.flatten() , final_gradcam
        
        
    elif modality == "text":
        final_gradcam_temp = np.mean(final_gradcam, axis=0)
        final_gradcam_temp = final_gradcam_temp.squeeze(0)
        text_relearance = np.mean(final_gradcam_temp, axis=0)
        text_relearance = (text_relearance - text_relearance.min()) / (text_relearance.max() - text_relearance.min() + 1e-8)
        text_relearance = torch.cat( ( torch.zeros(1), torch.tensor(text_relearance), torch.zeros(1)  ) )
        return text_relearance , final_gradcam


def get_rollout(cams, modality):
    num_layers = len(cams)
    x = None

    for i in range(num_layers):
        
        if modality == "image":
            cam_i = cams[i][0]
        elif modality == "text":
            cam_i = cams[i][0][:,1:-1,1:-1]
        else:
            print("Invalid modality")
            return None
            
        cam_i_avg = cam_i.mean(dim=0) 
        
        if x is None:
            x = cam_i_avg.clone()
        else:
            x = x * cam_i_avg  # Element-wise multiplication
            x = x / torch.norm(x, p=2)
            
            
    if modality == "image":
        final_gradcam_temp = x.reshape(36, 6, 6)
        final_gradcam_temp = final_gradcam_temp.cpu().detach().numpy()
        gradcam = final_gradcam_temp
    elif modality == "text":
        try:
            gradcam = x.cpu().numpy()
        except:
            gradcam = x.detach().cpu().numpy()


    heatmap = np.mean(gradcam, axis=0)
    # normalise heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    if modality == "text":
        text_relearance = heatmap.flatten()
        text_relearance = torch.cat( ( torch.zeros(1), torch.tensor(text_relearance), torch.zeros(1)  ) )
        return text_relearance


   
    
    return heatmap.flatten()


class GeneratorOurs:
    def __init__(self, model_usage, save_visualization=False):
        self.model_usage = model_usage

    def generate_ours(self, input, how_many = 2, index = None):
        output = self.model_usage.forward(input).question_answering_score
        model = self.model_usage.model
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(model.device) * output) #baka
        # one_hot = torch.sum(one_hot.cuda() * output) #baka
        model.zero_grad()
        one_hot.backward(retain_graph=True)


        image_flen1 = len(model.lxmert.encoder.visual_feats_list_x)
        text_flen1 = len(model.lxmert.encoder.lang_feats_list_x)

        def get_layer_wise_fevs1 (flen, modality, how_many):
            blk = model.lxmert.encoder.x_layers
            
            grads = []
            cams = []

            for i in range(flen):
                if modality == "image":
                    grad = blk[i].visn_self_att.self.get_attn_gradients().detach()
                    cam = blk[i].visn_self_att.self.get_attn().detach()
                    grads.append(grad)
                    cams.append(cam)
                else:
                    grad = blk[i].lang_self_att.self.get_attn_gradients().detach()
                    cam = blk[i].lang_self_att.self.get_attn().detach()
                    grads.append(grad)
                    cams.append(cam)
            
            
            if modality == "image":
                feats = model.lxmert.encoder.visual_feats_list_x[-2]
                dsm = get_eigs(feats, "image", how_many)
                lost = eig_seed(feats, 5)
                # pca_0 = get_pca_component(feats, "image", 0)
                
                dsm = np.array(dsm)
                lost = np.array(lost)
                # pca_0 = np.array(pca_0)

                
                x = np.array([dsm,lost])
                x = np.sum(x,axis=0)
                
                grad_cam, a = get_grad_cam(grads, cams,"image")
                grad_cam = grad_cam * 2
                rollout = get_rollout(cams,"image")
                pca_1 = get_pca_component(feats, "image", 1)
                pca_1 = np.array(pca_1) * 0.01

                y = np.array([grad_cam,rollout, pca_1])
                
                y = np.sum(y,axis=0)
                            
                z = x + y
                z = (z - z.min()) / (z.max() - z.min() + 1e-8)
            
            else:
                feats = model.lxmert.encoder.lang_feats_list_x[-1]
    
                dsm = get_eigs(feats, "text", how_many)
                # pca_0 = get_pca_component(feats, "text", 0)
                
                dsm = np.array(dsm)
                print("dsm",dsm.shape)
                # pca_0 = np.array(pca_0)
                
                x = np.array([dsm])
                x = np.sum(x,axis=0)
                
                grad_cam, a = get_grad_cam(grads, cams,"text")
                grad_cam = grad_cam * 2
                rollout = get_rollout(cams,"text")
                pca_1 = get_pca_component(feats, "text", 1)
                pca_1 = np.array(pca_1) * 0.01
                
                y = np.array([grad_cam,rollout, pca_1])
                
                y = np.sum(y,axis=0)
                            
                z = x + y
                z = (z - z.min()) / (z.max() - z.min() + 1e-8)
            
            return z


        image_fevs = get_layer_wise_fevs1( image_flen1 - 1, "image", how_many)
        
        lang_fevs = get_layer_wise_fevs1(text_flen1-1, "text", how_many)


        # new_fev_image = torch.stack(image_fevs, dim=0).sum(dim=0)
        new_fev_image = torch.tensor(image_fevs)
        # new_fev_lang = torch.stack(lang_fevs, dim=0).sum(dim=0)
        new_fev_lang = torch.tensor(lang_fevs)
        

        return [new_fev_lang], [new_fev_image]


 