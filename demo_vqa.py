#%%
from lxmert.lxmert.src.modeling_frcnn import GeneralizedRCNN
import lxmert.lxmert.src.vqa_utils as utils
from lxmert.lxmert.src.processing_image import Preprocess
from transformers import LxmertTokenizer
from lxmert.lxmert.src.huggingface_lxmert import LxmertForQuestionAnswering
from lxmert.lxmert.src.lxmert_lrp import LxmertForQuestionAnswering as LxmertForQuestionAnsweringLRP
from lxmert.lxmert.src.lxmert_lrp import LxmertAttention
from tqdm import tqdm
from lxmert.lxmert.src.ExplanationGenerator import (GeneratorBaselines,
                                                    GeneratorRMAblationNoAggregation)

from spectral.ExplanationGeneratorOurs import GeneratorOurs

import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from captum.attr import visualization
import requests

import os
import glob
import sys
from param import args

DEVICE = "cpu"


# OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
OBJ_URL = "util_files/objects_vocab.txt"
# ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
ATTR_URL = "util_files/attributes_vocab.txt"
# VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
VQA_URL = "util_files/trainval_label2ans.json"




class ModelUsage:
    def __init__(self, use_lrp=False):
        self.vqa_answers = utils.get_data(VQA_URL)

        # load models and model components
        self.frcnn_cfg = utils.Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn_cfg.MODEL.DEVICE = DEVICE

        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config = self.frcnn_cfg)

        self.image_preprocess = Preprocess(self.frcnn_cfg)

        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        if use_lrp:
            self.lxmert_vqa = LxmertForQuestionAnsweringLRP.from_pretrained("unc-nlp/lxmert-vqa-uncased").to(DEVICE)
        else:
            self.lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased").to(DEVICE)

        self.lxmert_vqa.eval()
        self.model = self.lxmert_vqa

        # self.vqa_dataset = vqa_data.VQADataset(splits="valid")

    def forward(self, item):
        URL, question = item

        self.image_file_path = URL

        # run frcnn
        images, sizes, scales_yx = self.image_preprocess(URL)
        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections= self.frcnn_cfg.max_detections,
            return_tensors="pt"
        )
        inputs = self.lxmert_tokenizer(
            question,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        self.question_tokens = self.lxmert_tokenizer.convert_ids_to_tokens(inputs.input_ids.flatten())
        self.text_len = len(self.question_tokens)
        # Very important that the boxes are normalized
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")
        self.image_boxes_len = features.shape[1]
        self.bboxes = output_dict.get("boxes")
        self.output = self.lxmert_vqa(
            input_ids=inputs.input_ids.to(DEVICE),
            attention_mask=inputs.attention_mask.to(DEVICE),
            visual_feats=features.to(DEVICE),
            visual_pos=normalized_boxes.to(DEVICE),
            token_type_ids=inputs.token_type_ids.to(DEVICE),
            return_dict=True,
            output_attentions=False,
        )
        return self.output


def save_image_vis(model_lrp, image_file_path, bbox_scores):
    # bbox_scores = image_scores
    _, top_bboxes_indices = bbox_scores.topk(k=1, dim=-1)
    img = cv2.imread(image_file_path)
    mask = torch.zeros(img.shape[0], img.shape[1])
    for index in range(len(bbox_scores)):
        [x, y, w, h] = model_lrp.bboxes[0][index]
        curr_score_tensor = mask[int(y):int(h), int(x):int(w)]
        new_score_tensor = torch.ones_like(curr_score_tensor)*bbox_scores[index].item()
        mask[int(y):int(h), int(x):int(w)] = torch.max(new_score_tensor,mask[int(y):int(h), int(x):int(w)])
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.unsqueeze_(-1)
    mask = mask.expand(img.shape)
    img = img * mask.cpu().data.numpy()
    cv2.imwrite('lxmert/lxmert/experiments/paper/new.jpg', img)


def test_save_image_vis(model_lrp, image_file_path, bbox_scores, evs, layer_num):
    # print(bbox_scores)
    # bbox_scores = image_scores
    _, top_bboxes_indices = bbox_scores.topk(k=5, dim=-1)

    img = cv2.imread(image_file_path)
    mask = torch.zeros(img.shape[0], img.shape[1])
    for index in top_bboxes_indices:
        img = cv2.imread(image_file_path)
        [x, y, w, h] = model_lrp.bboxes[0][index]
        cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 10)
        cv2.imwrite('saved_images/{}.jpg'.format(index), img)

    count = 1
    plt.figure(figsize=(15, 10))

    for idx in top_bboxes_indices:
      idx = idx.item()
      plt.subplot(1, len(top_bboxes_indices), count)
      plt.title(str(idx) + " " + evs + " " + layer_num)
      plt.axis('off')
      plt.imshow(cv2.imread('saved_images/{}.jpg'.format(idx)))
      count += 1

      
def text_map(model_lrp, text_scores, layer_num):
    # n_layers = len(text_scores)
    # print(f"Text n_layers: {n_layers}")
    plt.figure(figsize=(10, 8))
    # for j in range(len(text_scores)):
        # if j == 3:
            # print(text_scores[j])
        # text_scores[j] = torch.cat( ( torch.zeros(1), text_scores[j], torch.zeros(1)  ) )
    plt.subplot(1, 1, 1)
    plt.title(args.method_name + " word impotance " + layer_num)
    # plt.xticks(np.arange(len(text_scores[j])), model_lrp.question_tokens[1:-1])
    plt.xticks(np.arange(len(text_scores)), model_lrp.question_tokens)
    # plt.imshow(torch.abs(text_scores[j].unsqueeze(dim = 0)).numpy())
    plt.imshow(text_scores.unsqueeze(dim = 0).detach().numpy())
    plt.colorbar(orientation = "horizontal")


        # plt.title("SA word impotance " + str(j))
        # plt.imshow(text_scores[j].unsqueeze(dim = 0).numpy())
        # plt.colorbar(orientation = "horizontal")



def spectral_stuff():
    model_lrp = ModelUsage(use_lrp=True)
    ours = GeneratorOurs(model_lrp)
    baselines = GeneratorBaselines(model_lrp)
    vqa_answers = utils.get_data(VQA_URL)


    URL = args.image_path
    qs = args.question
        
    if args.method_name == "ours":
        text_relevance, image_relevance = ours.generate_ours((URL, qs), how_many = 5)


    elif args.method_name == "transformer_att":
        text_relevance, image_relevance = baselines.generate_transformer_attr((URL, qs))

    elif args.method_name == "raw_attn":
        text_relevance, image_relevance = baselines.generate_raw_attn((URL, qs))

    elif args.method_name == "partial_lrp":
        text_relevance, image_relevance = baselines.generate_partial_lrp((URL, qs))

    elif args.method_name == "gradcam":
        text_relevance, image_relevance = baselines.generate_attn_gradcam((URL, qs))

    elif args.method_name == "rollout":
        text_relevance, image_relevance = baselines.generate_rollout((URL, qs))


    elif args.method_name == "rm":
        text_relevance, image_relevance = baselines.generate_relevance_maps((URL, qs),
                                     use_lrp=False, normalize_self_attention=True)


    text_scores = text_relevance[0]
    image_scores = image_relevance[0]


    test_save_image_vis(model_lrp, URL, image_scores, args.method_name, '')
        

    text_map(model_lrp, text_scores, '')



    save_image_vis(model_lrp, URL, image_scores)
    orig_image = Image.open(model_lrp.image_file_path)

    fig, axs = plt.subplots(ncols=2, figsize=(20, 5))
    axs[0].imshow(orig_image)
    axs[0].axis('off')
    axs[0].set_title('original')

    masked_image = Image.open('lxmert/lxmert/experiments/paper/new.jpg')
    axs[1].imshow(masked_image)
    axs[1].axis('off')
    axs[1].set_title('masked')

    print(f"QUESTION: {qs}")
    print("ANSWER:", vqa_answers[model_lrp.output.question_answering_score.argmax()])
    

    plt.show()

    

if __name__ == '__main__':
    # main()


    files = glob.glob('saved_images/*')
    for f in files:
        os.remove(f)
    spectral_stuff()



# python demo_vqa.py --image-path /home/pranav/ExplanableAI/Transformer-spectral-interpretability-main/CLIP/astronaut.png --question "an astronaut and a rocket" --method-name ours