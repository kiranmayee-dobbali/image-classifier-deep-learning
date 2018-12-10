import torch
from torchvision import datasets, models, transforms
from torch import optim, nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

import json
from PIL import Image
import time
from collections import OrderedDict
import numpy as np
import glob
import random
import pickle


class ImageClassifer:
    def __init__(self, data_directory, save_directory, arch="vgg11",
                 learning_rate=0.003, hidden_layers=512, epochs=4, machine=None):
        self.arch = arch
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.print_freq = 40
        if machine is None:
            self.machine = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            if machine == "gpu":
                self.machine = "cuda:0"
            else:
                self.machine = "cpu"
        self.hidden_layer = hidden_layers
        self.criterion = nn.NLLLoss()
        self.data_directory = data_directory
        self.save_directory = save_directory

        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        test_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

        # TODO: Load the datasets with ImageFolder
        self.train_data = datasets.ImageFolder(self.data_directory + '/train', transform=train_transforms)
        self.test_data = datasets.ImageFolder(self.data_directory + '/test', transform=test_transforms)
        self.valid_data = datasets.ImageFolder(self.data_directory + '/valid', transform=test_transforms)

        # TODO: Using the image datasets and the trainforms, define the dataloaders

        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size=32)
        self.validloader = torch.utils.data.DataLoader(self.valid_data, batch_size=32)
        self.output_size = len(self.train_data.classes)


    def build_network(self):
        model_select = {"vgg13": models.vgg13(pretrained=True), "vgg11": models.vgg11(pretrained=True)}
        model = model_select[self.arch]
        print(model)

        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features, self.hidden_layer)
                                   , nn.ReLU()
                                   , nn.Dropout(0.05)
                                   , nn.Linear(self.hidden_layer, self.output_size)
                                   , nn.LogSoftmax(dim=1))
        model.classifier = classifier
        filename = 'model_base.pkl'
        pickle.dump(model, open(filename, 'wb'))
        return model, classifier

    def validation(self, model, validloader, criterion):
        valid_loss = 0
        accuracy = 0

        model.to(self.machine)

        for _, (inputs, labels) in enumerate(validloader):
            inputs = inputs.to(self.machine)
            labels = labels.to(self.machine)
            output = model.forward(inputs)
            valid_loss += criterion(output, labels).item()

            prob = torch.exp(output)

            equality = (labels.data == prob.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return valid_loss, accuracy

    def train(self, model):

        optimizer = optim.Adam(model.classifier.parameters(), lr=self.learning_rate)
        model.to(self.machine)
        print("Running script on {}..".format(self.machine))
        print("---------------------------------------------------------")
        start_time = time.time()
        counter = 0
        for e in range(self.epochs):
            running_loss = 0
            print("Epoch : {}".format(e + 1))

            for _, (inputs, labels) in enumerate(self.trainloader):
                counter += 1
                inputs = inputs.to(self.machine)
                labels = labels.to(self.machine)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if counter % self.print_freq == 0:
                    model.eval()
                    with torch.no_grad():
                        valid_loss, accuracy = self.validation(model, self.validloader, self.criterion)

                    training_loss = round(running_loss / self.print_freq, 3)
                    validation_loss = round(valid_loss / len(self.validloader), 3)
                    accuracy = round(float(accuracy / len(self.validloader)), 3)
                    print("              Training Loss :{} | Val Loss: {} | Val Accuracy : {}".format(training_loss,
                                                                                                      validation_loss,
                                                                                                      accuracy))

                    running_loss = 0
                    model.train()

            run_time = time.time() - start_time
            print('\nTime Elaspsed so far: {} min {} sec\n'.format(int(run_time / 60), round(run_time % 60, 1)))
            print("--------------------------------------------")

        return model, optimizer

    def save_model_checkpoint(self, model, optimizer):
        # TODO: Save the checkpoint

        model_state = {
            'class_to_idx': self.train_data.class_to_idx,
            'epoch': self.epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'classifier': model.classifier
        }
        torch.save(model_state, self.save_directory)



    def load_checkpoint(self,  model_base, filepath = None,):
        if not filepath:
            filepath = self.save_directory

        checkpoint = torch.load(filepath)
        model_base.load_state_dict(checkpoint['state_dict'])
        model_base.classifier = checkpoint['classifier']
        model_base.class_to_idx = checkpoint['class_to_idx']
        return model_base
    
    def process_image(self, image_path):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        image = Image.open(image_path)
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),  # Cropping
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

        return np.array(transform(image))

    def imshow(self, image, ax=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()

        image = image.transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        ax.imshow(image)

        return ax


    def mapping(self, categories):
        mapping = {}
        for label, num in self.train_data.class_to_idx.items():
            for cat_label, cat_name in categories.items():
                if cat_label == label:
                    mapping[num] = cat_name
        return mapping


    def predict(self, image_path, model_checkpoint, cat_to_name, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # TODO: Implement the code to predict the class from an image file
        
        model = self.load_checkpoint(model_checkpoint).cpu()
        img = self.process_image(image_path)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        img = img.unsqueeze_(0)
        model.eval()
        log_prob = model.forward(img)
        prob = torch.exp(log_prob)
        top_probs, top_classes = prob.topk(topk)[0], prob.topk(topk)[1]
        top_probs = top_probs.detach().numpy()[0]
        top_probs = [round(prob, 3) for prob in top_probs.tolist()]
        top_classes = top_classes.detach().numpy()[0]
        
        with open(cat_to_name, 'r') as f:
            categories = json.load(f)
        indx = self.mapping(categories)
        top_classes = [indx[x] for x in top_classes.tolist()]
        return top_probs, top_classes

    # TODO: Display an image along with the top 5 classes

    @staticmethod
    def format_result(probs, classes, image):
        fig, (ax1, ax2) = plt.subplots(figsize=(15, 4.5), ncols=2, dpi=100, nrows=1, sharey=False)
        ax1.imshow(image)
        ax1.set_title(classes[0])
        ax1.axis('off')

        class_num = np.arange(len(classes))
        ax2.barh(class_num, probs, color='green')
        ax2.set_title("Top 5 predictions with Prob")
        ax2.set_yticks(class_num)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.set_yticklabels(classes)
        ax2.invert_yaxis()
        return

    
if __name__ == "__main__":
    ic = ImageClassifer("flowers/","model_checkpoint.pth")
    print("Total number of classes : {}".format(ic.output_size))
    model_base, _ = ic.build_network()
    model, optimizer = ic.train(model_base)
    ic.save_model_checkpoint(model, optimizer)
