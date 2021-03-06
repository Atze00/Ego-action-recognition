import numpy as np
from torchvision import transforms
import cv2
from attentionMapModel import attentionMap
from ModelsRGB import *
from PIL import Image

def generateAttentionMap(model_state_dict,fl_name_in,fl_name_out):
    num_classes = 61 # Classes in the pre-trained model
    mem_size = 512 # Weights of the pre-trained model

    model=ConvLSTMAttention(num_classes=num_classes, mem_size=mem_size,supervision=True)

    model.load_state_dict(torch.load(model_state_dict))
    model_backbone = model.resNet
    attentionMapModel = attentionMap(model_backbone).cuda()
    attentionMapModel.train(False)
    for params in attentionMapModel.parameters():
        params.requires_grad = False

    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    preprocess1 = transforms.Compose([
       transforms.Scale(256),
       transforms.CenterCrop(224),
    ])

    preprocess2 = transforms.Compose([
        transforms.ToTensor(),
        normalize])


    img_pil = Image.open(fl_name_in)
    img_pil1 = preprocess1(img_pil)
    img_size = img_pil1.size
    size_upsample = (img_size[0], img_size[1])
    img_tensor = preprocess2(img_pil1)
    img_variable = Variable(img_tensor.unsqueeze(0).cuda())
    img = np.asarray(img_pil1)
    with torch.no_grad():
        attentionMap_image = attentionMapModel(img_variable, img, size_upsample)
    cv2.imwrite(fl_name_out, attentionMap_image)
