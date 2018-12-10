from image_classifer import ImageClassifer
import argparse

# Parsing Arguments
parser = argparse.ArgumentParser()
parser.add_argument("data_directory",help = "Add Datasets path", default = "flowers/")
parser.add_argument("--save_dir",help = "Add path to save the trained model", default = "model_checkpoint.pth")
parser.add_argument("--arch",help = "Select vgg11 or vgg13 or any vgg architecture", default = "vgg11")
parser.add_argument("--hidden_units", help ="Total hidden layers",default = 512)
parser.add_argument("--learning_rate",help = "Choose learning rate", default = 0.003)
parser.add_argument("--epochs",help = "Choose epochs", default = 4)
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()
data_directory = args.data_directory
save_directory =args.save_dir 
hidden_layers = int(args.hidden_units)
arch = args.arch
learning_rate = float(args.learning_rate)
epochs = int(args.epochs)
if args.gpu:
    machine = args.gpu
    print("Using GPU..")
else: 
    print("Using CPU..")
    machine = "cpu"

ic = ImageClassifer(data_directory,save_directory, learning_rate = learning_rate, arch = arch,
                    hidden_layers = hidden_layers, epochs = epochs, machine = machine )

print("Total number of classes : {}".format(ic.output_size))
model_base, _ = ic.build_network()
model, optimizer = ic.train(model_base)
ic.save_model_checkpoint(model, optimizer)   
print("Model is done training!!")