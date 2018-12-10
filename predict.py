from image_classifer import ImageClassifer
import argparse
import pickle
from operator import itemgetter

parser = argparse.ArgumentParser()
parser.add_argument("image_path",help = "Enter path to the image", default = "/home/workspace/aipnd-project/flowers/test/102/image_08023.jpg")
parser.add_argument("model_checkpoint",help = "Enter path to the image", default = "model_checkpoint.pth")
parser.add_argument("--k",help = "Top K predictions", default = 3)
parser.add_argument("--category_names",help = "Top K predictions", default = "cat_to_name.json")
parser.add_argument('--gpu', action='store_true')

args = parser.parse_args()
checkpoint_path = args.model_checkpoint
image_path = args.image_path
k = int(args.k)
categories = args.category_names
if args.gpu:
    machine = args.gpu
    print("Using GPU to predict..")
else: 
    print("Using CPU to predict..")
    machine = "cpu"

ic = ImageClassifer("flowers/",save_directory = checkpoint_path, machine = machine)
model_base = pickle.load(open('model_base.pkl', 'rb'))
model = ic.load_checkpoint(model_base, checkpoint_path)
prob, classes = ic.predict(image_path, model, categories, topk = k)
result = list(zip(prob, classes))
prob = max(result, key = itemgetter(0))[0]
class_label =  max(result, key = itemgetter(0))[1]
print("Predicted class is {} at {} probablity".format(class_label, prob))
print("Other top {} values are : {}".format(args.k, result))
