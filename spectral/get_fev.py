import numpy as np
import torch
# import copy
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eig
import torch.nn.functional as F
from pymatting.util.util import row_sum
from scipy.sparse import diags
import math

def get_diagonal (W):
    D = row_sum(W)
    D[D < 1e-12] = 1.0  # Prevent division by zero.
    D = diags(D)
    return D

def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

# def handle_fev (fev):
#     temp = torch.abs(fev)
#     idx = torch.argmax(temp)
#     if fev[idx] < 0:
#         return fev * -1
#     return fev

def get_eigs (feats, modality, how_many = None):
    if feats.size(0) == 1:
        feats = feats.detach().squeeze()


    if modality == "image":
        n_image_feats = feats.size(0)
        val = int( math.sqrt(n_image_feats) )
        if val * val == n_image_feats:
            feats = F.normalize(feats, p = 2, dim = -1)
            # feats = feats
        elif val * val + 1 == n_image_feats:
            feats = F.normalize(feats, p = 2, dim = -1)[1:]
            # feats = F[1:]

        else:
            print(f"Invalid number of features detected: {n_image_feats}")

    else:
        feats = F.normalize(feats, p = 2, dim = -1)[1:-1]
        # feats = feats[1:-1]


    W_feat = (feats @ feats.T)
    W_feat = (W_feat * (W_feat > 0))
    W_feat = W_feat / W_feat.max() 

    W_feat = W_feat.detach().cpu().numpy()

    
    D = np.array(get_diagonal(W_feat).todense())

    L = D - W_feat

    L_shape = L.shape[0]
    if how_many >= L_shape - 1: 
        how_many = L_shape - 2

    try:
        eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM', sigma = -0.5, M = D)
    except:
        try:
            eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM', sigma = -0.5)
        except:
            eigenvalues, eigenvectors = eigs(L, k = how_many, which = 'LM')
    eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()
    
    n_tuple = torch.kthvalue(eigenvalues.real, 2)
    fev_idx = n_tuple.indices
    fev = eigenvectors[fev_idx]

    if modality == 'text':
        fev = torch.cat( ( torch.zeros(1), fev, torch.zeros(1)  ) )

    return torch.abs(fev)
    # return handle_fev(fev)




def get_grad_eigs (feats, modality, grad, device = "cpu", how_many = None):
    fev = get_eigs(feats, modality, how_many)
    n_feats = fev.size(0)
    if n_feats == grad.size(2) - 1: #images
        grad = grad[:, :, 1:, 1:]
    elif modality == "text": #text
        grad = grad[:, :, 1:-1, 1:-1]
        fev = fev[1:-1]

    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    grad = grad.clamp(min=0).mean(dim=0)
    fev = fev.to(device)
    fev = grad @ fev.unsqueeze(1)
    fev = fev[:, 0]

    if modality == 'text':
        fev = torch.cat( ( torch.zeros(1).to(device), fev, torch.zeros(1).to(device)  ) )
    
    return torch.abs(fev)



def get_grad_cam_eigs (feats, modality, grad, cam, device = "cpu", how_many = None):
    fev = get_eigs(feats, modality, how_many)
    n_feats = fev.size(0)
    if n_feats == grad.size(2) - 1: #images
        grad = grad[:, :, 1:, 1:]
    elif modality == "text": #text
        grad = grad[:, :, 1:-1, 1:-1]
        fev = fev[1:-1]

    if n_feats == cam.size(2) - 1: #images
        cam = cam[:, :, 1:, 1:]
    elif modality == "text": #text
        cam = cam[:, :, 1:-1, 1:-1]

    cam = avg_heads(cam, grad)
    fev = fev.to(device)
    fev = cam @ fev.unsqueeze(1)
    fev = fev[:, 0]
    if modality == "text":
        fev = torch.cat( ( torch.zeros(1).to(device), fev, torch.zeros(1).to(device) ) )

    return torch.abs(fev)