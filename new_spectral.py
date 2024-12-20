from lxmert.lxmert.src.modeling_frcnn import GeneralizedRCNN
import lxmert.lxmert.src.vqa_utils as utils
from lxmert.lxmert.src.processing_image import Preprocess
from transformers import LxmertTokenizer
from lxmert.lxmert.src.huggingface_lxmert import LxmertForQuestionAnswering
from lxmert.lxmert.src.lxmert_lrp import LxmertForQuestionAnswering as LxmertForQuestionAnsweringLRP
from lxmert.lxmert.src.lxmert_lrp import LxmertAttention
from tqdm import tqdm
from lxmert.lxmert.src.ExplanationGenerator import (GeneratorOurs, GeneratorBaselines,
                                                    GeneratorOursAblationNoAggregation)
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
import json

DEVICE = "cpu"


OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"



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


def test_save_image_vis(model_lrp, image_file_path, bbox_scores, ap_type):
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
      plt.title(ap_type + " " + str(idx))
      plt.axis('off')
      plt.imshow(  cv2.cvtColor(cv2.imread('saved_images/{}.jpg'.format(idx)), cv2.COLOR_BGR2RGB)  )
      count += 1


def their_stuff(model_lrp, lrp, vqa_answers, img_id, qs):

    # model_lrp = ModelUsage(use_lrp=True)
    # lrp = GeneratorOurs(model_lrp)
    # baselines = GeneratorBaselines(model_lrp)
    # vqa_answers = utils.get_data(VQA_URL)

    image_ids = [
        # giraffe
        'COCO_val2014_000000185590',
        # baseball
        'COCO_val2014_000000127510',
        # bath
        'COCO_val2014_000000324266',
        # frisbee
        'COCO_val2014_000000200717',

        'COCO_val2014_000000159282'
    ]

    test_questions_for_images = [
        ################## paper samples
        # giraffe
        "is the animal eating?",
        # baseball
        "did he catch the ball?",
        # bath
        "is the tub white ?",
        # frisbee
        "did the man just catch the frisbee?",

        # "What kind of flowers are those?"
        "What is at the bottom of the vase?"
        ################## paper samples
    ]

    # URL = 'lxmert/lxmert/experiments/paper/{0}/{0}.jpg'.format(image_ids[idx])
    URL = '../../data/root/val2014/{0}.jpg'.format(img_id)

    R_t_t, R_t_i = lrp.generate_ours((URL, qs),
                                     use_lrp=False, normalize_self_attention=True, method_name="ours")

    image_scores = R_t_i[0]
    text_scores = R_t_t[0]

    test_save_image_vis(model_lrp, URL, image_scores, "HC RM")

    save_image_vis(model_lrp, URL, image_scores)
    orig_image = Image.open(model_lrp.image_file_path)

    fig, axs = plt.subplots(ncols=2, figsize=(20, 5))
    axs[0].imshow(orig_image);
    axs[0].axis('off');
    axs[0].set_title('HC RM original');

    masked_image = Image.open('lxmert/lxmert/experiments/paper/new.jpg')
    axs[1].imshow(masked_image);
    axs[1].axis('off');
    axs[1].set_title('HC RM masked');

    # text_scores = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min())
    # vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,model_lrp.question_tokens,1)]
    # visualization.visualize_text(vis_data_records)

    print(f"\nQUESTION: {qs}")
    print("ANSWER:", vqa_answers[model_lrp.output.question_answering_score.argmax()])
    # print(image_scores)

    plt.show()



def spectral_stuff(model_lrp, lrp, vqa_answers, img_id, qs):
    # model_lrp = ModelUsage(use_lrp=True)
    # lrp = GeneratorOurs(model_lrp)
    # baselines = GeneratorBaselines(model_lrp)
    # vqa_answers = utils.get_data(VQA_URL)

    # baselines.generate_transformer_attr(None)
    # baselines.generate_attn_gradcam(None)
    # baselines.generate_partial_lrp(None)
    # baselines.generate_raw_attn(None)
    # baselines.generate_rollout(None)

    # image_ids = [
    #     # giraffe
    #     'COCO_val2014_000000185590',
    #     # baseball
    #     'COCO_val2014_000000127510',
    #     # bath
    #     'COCO_val2014_000000324266',
    #     # frisbee
    #     'COCO_val2014_000000200717',

    #     'COCO_val2014_000000159282'
    # ]

    # test_questions_for_images = [
    #     ################## paper samples
    #     # giraffe
    #     "is the animal eating?",
    #     # baseball
    #     "did he catch the ball?",
    #     # bath
    #     "is the tub white ?",
    #     # frisbee
    #     "did the man just catch the frisbee?",

    #     # "What kind of flowers are those?"
    #     "What is at the bottom of the vase?"
    #     ################## paper samples
    # ]

    URL = '../../data/root/val2014/{0}.jpg'.format(img_id)
    # URL = 'giraffe.jpg'

    R_t_t, R_t_i = lrp.generate_ours_dsm((URL, qs), sign_method="mean", use_lrp=False, 
                                         normalize_self_attention=True, method_name="dsm")
    text_scores = R_t_t
    image_scores = torch.abs(R_t_i)
    # print(image_scores)

    # print(image_scores.topk(k = 5).indices)
    test_save_image_vis(model_lrp, URL, image_scores, "spectral+")
    # test_save_image_vis(model_lrp, URL, image_scores * -1, "spectral-")



    save_image_vis(model_lrp, URL, image_scores)
    orig_image = Image.open(model_lrp.image_file_path)

    fig, axs = plt.subplots(ncols=2, figsize=(20, 5))
    axs[0].imshow(orig_image)
    axs[0].axis('off')
    axs[0].set_title('spectral original')

    masked_image = Image.open('lxmert/lxmert/experiments/paper/new.jpg')
    axs[1].imshow(masked_image)
    axs[1].axis('off')
    axs[1].set_title('spectral masked')

    # text_scores = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min())
    # vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,model_lrp.question_tokens,1)]
    # visualization.visualize_text(vis_data_records)
    print(f"\nQUESTION: {qs}")
    print("ANSWER:", vqa_answers[model_lrp.output.question_answering_score.argmax()])
    # print(image_scores)

    

    plt.show()



# def transformer_attn ():
#     model_lrp = ModelUsage(use_lrp=True)
#     lrp = GeneratorOurs(model_lrp)
#     baselines = GeneratorBaselines(model_lrp)
#     vqa_answers = utils.get_data(VQA_URL)

#     # baselines.generate_transformer_attr(None)
#     # baselines.generate_attn_gradcam(None)
#     # baselines.generate_partial_lrp(None)
#     # baselines.generate_raw_attn(None)
#     # baselines.generate_rollout(None)

#     image_ids = [
#         # giraffe
#         'COCO_val2014_000000185590',
#         # baseball
#         'COCO_val2014_000000127510',
#         # bath
#         'COCO_val2014_000000324266',
#         # frisbee
#         'COCO_val2014_000000200717',

#         'COCO_val2014_000000159282'
#     ]

#     test_questions_for_images = [
#         ################## paper samples
#         # giraffe
#         "is the animal eating?",
#         # baseball
#         "did he catch the ball?",
#         # bath
#         "is the tub white ?",
#         # frisbee
#         "did the man just catch the frisbee?",

#         "What kind of flowers are those?"
#         ################## paper samples
#     ]
#     URL = 'lxmert/lxmert/experiments/paper/{0}/{0}.jpg'.format(image_ids[4])
#     # URL = 'giraffe.jpg'

#     R_t_t, R_t_i = baselines.generate_transformer_attr((URL, test_questions_for_images[4]), method_name="transformer_attr")

#     text_scores = R_t_t[0]
#     image_scores = R_t_i[0]
#     test_save_image_vis(model_lrp, URL, image_scores)


#     save_image_vis(model_lrp, URL, image_scores)
#     orig_image = Image.open(model_lrp.image_file_path)

#     fig, axs = plt.subplots(ncols=2, figsize=(20, 5))
#     axs[0].imshow(orig_image)
#     axs[0].axis('off')
#     axs[0].set_title('original')

#     masked_image = Image.open('lxmert/lxmert/experiments/paper/new.jpg')
#     axs[1].imshow(masked_image)
#     axs[1].axis('off')
#     axs[1].set_title('masked')

#     text_scores = (text_scores - text_scores.min()) / (text_scores.max() - text_scores.min())
#     vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,model_lrp.question_tokens,1)]
#     visualization.visualize_text(vis_data_records)
#     print("ANSWER:", vqa_answers[model_lrp.output.question_answering_score.argmax()])


#     plt.show()
    

if __name__ == '__main__':
    files = glob.glob('saved_images/*')
    for f in files:
        os.remove(f)
    model_lrp = ModelUsage(use_lrp = True)
    lrp = GeneratorOurs(model_lrp)
    vqa_answers = utils.get_data(VQA_URL)

    max_qs = 214354
    f = open('data/vqa/valid_mega.json')
    data = json.load(f)


    while True:
        print("Enter an option:\n\t1. Spectral Approach followed by Hila Chefer Relevance Maps\n\t2. Quit loop")
        opt = int(input("Your choice: "))
        idx = random.randint(0, max_qs - 1)
        img_id, qs = data[idx]["img_id"], data[idx]["sent"]

        if opt == 1:
            print(img_id)
            spectral_stuff(model_lrp, lrp, vqa_answers, img_id, qs)
        # elif opt == 2:
            their_stuff(model_lrp, lrp, vqa_answers, img_id, qs)
            
        else:
            break
    # transformer_attn()
