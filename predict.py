import argparse
import torch
from torchvision import models
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json

def get_input_args():
    parser=argparse.ArgumentParser(description='predict flower name')
    parser=add_argument('image_path',type=str,help='image directory')
    parser.add_argument('checkpoint',type=str, help='checkpint directory')
    parser.add_argument('--top_k',type=int,default=5,help='top 5 classes')
    parser.add_argument('--category_names',type=str, default=None, help='converting categories to real names')
    parser.add_argument'--gpu',action='store_true',help='use gpu if it is available')
    return parser.parse_args()
def load_checkpoint(filepath):
    checkpoint=torch.load(filepath)
    arch=checkpoint['architecture']
    hidden_units=checkpoint['hidden_units']
    if arch=='vgg16':
        model=models.vgg16(pretrained=True)
        input_size=25088
    elif arch='vgg13':
        model=models.vgg13(pretrained=True)
        input_size=25088
    else:
        print(f"not supported")
        exit()
    for param in model.parameters():
        param.requires_grad=False
    model.classifier=checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx=checkpoint['class_to_idx']
    return model
def process_image(image_path):
    pil_image=Image.open(image_path)
    aspect=pil_image.width/pil_image.height
    if aspect>1:
        pil_image.thumbnail((aspect*256,256))
    else:
        pil_image.thumbnail((256,256/aspect))
    left=(pil_image.width-224)/2
    top=(pil_image.height-224)/2
    right=left+224
    bottom=top+224
    pil_image=pil_image.crop((left,top,right,bottom))
    
    np_image=np.array(pil_image)/255.0
    mean=np.array([0.485,0.456,0.406])
    stddev=np.array([0.229,0.224,0.225])
    np_image=(np_image-mean)/stddev
    return np_image

def predict(image_path,model,device,top_k):
    model.eval()
    model.to(device)
    np.image=process_image(image_path)
    tensor_image=torch.from_numpy(np_image).type(torch.FloatTensor).unsqueeze(0).to(device)
    with torch.no_grad():
        output=model(tensor_image)
    probs=torch.softmax(output,dim=1)
    top_probs,top_indices=probs.topk(top_k,dim=1)
    top_probs=top_probs.cpu().numpy().flatten()
    top_indices=top_indices.cpu().numpy().flatten()
    idx_to_class={v:k for k,v in model.class_to_idx.items()}
    top_classes=[idx_to_class[idx] for idx in top_indices]
    return top_probs,top_classes

def main():
    args=get_input_args()
    device=torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model= load_checkpoint(args.checkpoint)
    probs,classes=predict(args.image_path,model, device,args.top_k)
    if args.category_names:
        with open(args.category_names,'r') as f:
            cat_to_name=json.load(f)
        class_names=[cat_to_name.get(cls,cls) for cls in classes]
        else:
            class_names=classes
        print('prdicted class and probabilities are:')
        for prob,cls in zip(probs,class_names):
            print(f"{clas}:{prob:.4f}")
if __name__=='__main__':
    main()
