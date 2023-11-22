import torch
import numpy as np
from PIL import Image


def convert_image_to_tensor(img):
    img = img.convert("RGB")#RGB색상 모드로 변환 
    img_np = np.array(img).astype(np.float32)#이미지를 numpy 배열로 변환한 후 데이터 타입을 np.float32로 변환
    img_np = np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]#numpy 배열의 축을 재배열/ 높이,너비,채널 -> 채널,높이,너비
    img_pt = torch.from_numpy(img_np)#Numpy배열에 저장된 데이터를 PyTorch Tensor로 가져옴. GPU 사용
    return img_pt


def convert_tensor_to_image(img_pt):
    img_pt = img_pt[0, ...].permute(1, 2, 0)#배열을 다시 높이,너비,채널 순으로
    result_rgb_np = img_pt.cpu().numpy().astype(np.uint8)#gpu->cpu, numpy배열로 변환, unit8로 변환(8비트 정수 타입, 픽셀값)
    return Image.fromarray(result_rgb_np)#numpy 배열을 pillow 이미지로 변환
