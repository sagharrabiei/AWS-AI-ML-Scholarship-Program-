import torch
import argparse
from torchvision import datasets,models,transforms
from torch import optim,nn
import torch.nn.functional as F
import json
import os

def get_input_args():
    parser=argparse.ArgumentParser(description='train a neural network')
    parser.add_argument('data_dir',type=str,help='directory of dataset')
    parser.add_argument('--save_dir', type=str, help='directory for saving checkpoints')
    parser.add_argument('--arch',type=str,default='vgg16',help='model we are using(vgg16)')
    parser.add_argument('learning_rate', type=float, default=0.0001,help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=2048, help='number of hidden units')
    parser.add_argument('--gpu',action='store_true', help='use gpu if available')
    return parser.parse_args()
def load_data(data_dir):
    train_dir=os.path.join(data_dir,'train')
    valid_dir=os.path.join(data_dir,'valid')
    test_dir=os.path.join(data_dir,'test')
    
    data_transforms={
        'train':transoforms.Compose([
            transforms.RandomResizedCrop([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],
                                      [0.229,0.224,0.225])
                                     ]),
            'test':transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225])
            ]),
    image_datasets={
        'train':datasets.ImageFolder(train_dir,transform=data_transforms['train']),
 'valid':datasets.ImageFolder(valid_dir,transform=data_transforms['valid']),
 'test':datasets.ImageFolder(test_dir,transform.=data_transforms['test'])
    }
    dataloaders={
        'train': torch.utils.data.DataLoader(image_datasets['train'],batch_size=64,shuffle=True),
        'valid':torch.utils.data.DataLoader(image_datasets['valid'],batch_size=64,shuffle=False),
 'test':torch.utils.data.DataLoader(image_datasets['test'],batch_size=64,shuffle=False)
    }
return dataloaders,image_datasets['train'].class_to_idx
def create_classifier(input_size,hidden_units,output_size):
            classifier=nn.Sequential(
                nn.Linear(input_size,hidden_units),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_units,512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512,output_size),
            )
            return classifier
 def train_model(model,criterion,optimizer,dataloaders,device,epochs=15,patience=3):
            steps=0
            print_every=10
            train_loss=0
            best_validation_loss=float('inf')
            patience_counter=0
            for epoch in range(epochs):
                model.train()
                for inputs,labels in dataloaders['train']:
                    steps+=1
                    inputs,labels=inputs.to(device),labels.to(device)
                    optimizer.zero_grad()
                    outputs=model(inputs)
                    loss=criterion(outputs,labels)
                    
                    loss.backward()
                    optimizer.step()
                    train_loss+=loss.item()
                    if steps % print_every==0:
                        model.eval()
                        validation_loss=0
                        accuracy=0
                        with torch.no_grad():
                            for inputs,labels in dataloaders['valid']:
                                inputs,labels=inputs.to(device),
            labels.to(device)
                  
                                outputs=model(images)
                                loss=criterion(outputs,dim=1)
                                validation_loss+=loss.item()
                                ps=torch.softmax(outputs,dim=1)
                                top_p,top_class=ps.topk(1,dim=1)
                                equals=top_class==labels.view(*top_class.shape)
                                accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()
                    print(f"Epoch: {epoch+1}/{epochs}"
            f"Step:{steps}"
            f"Training Loss: {train_loss/print_every:.3f}"
             f"Test Loss: {validation_loss/len(dataloaders['valid']):.3f}"
             f"Accuracy: {accuracy/len(dataloaders['valid']):.3f}")
            train_loss=0
            model.train()
            if validation_loss<best_validation_loss:
                best_validation_loss=validation_loss
                patience_counter=0
            else:
                patience_counter+=1
                print(f"patience counter:{patience_counter}/{patience}")
                if patience_counter >=patience:
                    print("early stopping!")
                    return model
    return model
 def save_checkpoint(model,save_dir,arch,class_to_idx,hidden_units):
            checkpoints={
                'architecture':arch,
                'classifier':model.classifier,
                'state_dict':model.state_dict(),
                'class_to_idx':class_to_idx,
                'hidden_units':hidden_units}
            save_path=os.path.join(save_dir,'checkpoint.pth')
            torch.save(checkpoint,save_path)
            print(f"model saving path {save_path}")
def main():
            args=get_input_args()
            dataloaders,class_to_idx=load_data(args.data_dir)
            device=torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
            if args.arch=="vgg16":
                model=models.vgg16(pretrained=True)
                input_size=25088
            for param in model.parameters():
                param.requires_grad=False
                model.classifier=create_classifier(input_size=input_size,hidden_units=args.hidden_units,output_size=102)
 model.class_to_idx=class_to_idx
                model.to(device)
                criterion=nn.CrossEntropyLoss()
                optimizer=optim.Adam(model.classifier.parameters(),lr=args.learning_rate)
            
            model=train_model(model,criterian,optimizer,dataloaders,device,epochs=args.epochs)
            save_checkpoint(model,args.save_dir,args.arch,class_to_idx,args.hidden_units)
            
            if __name__==__'main'__:
                main()
            
 
            
            
            
                 
            
            
            
            
    
    
    