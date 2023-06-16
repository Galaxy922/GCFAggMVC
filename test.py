import torch
from network import GCFAggMVC
from metric import valid
import argparse
from dataloader import load_data
# Synthetic3d
# Prokaryotic
# CCV
# MNIST-USPS
# Hdigit
# YouTubeFace
# Cifar10
# Cifar10
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
Dataname = 'Hdigit'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--rec_epochs", default=200)
parser.add_argument("--fine_tune_epochs", default=100)
parser.add_argument("--low_feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset, dims, view, data_size, class_num = load_data(args.dataset)
model = GCFAggMVC(view, dims, args.low_feature_dim, args.high_feature_dim, device)
model = model.to(device)
checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)
print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading models...")
valid(model, device, dataset, view, data_size, class_num)
