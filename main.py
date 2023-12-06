from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, vgg16
from torchvision.datasets import ImageFolder
from utils import *

torch.manual_seed(0)
mkdir('figures')
mkdir('checkpoint')

parser = argparse.ArgumentParser(description="Final Project")
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=0.003, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--model', default="vgg16", type=str)
parser.add_argument('--tl_mode', default="no_pretrain", type=str)
args = parser.parse_args()

device = getDevice()
num_epochs = args.epochs
lr = args.lr
batch_size = args.batch_size

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

image_size = 64
dataset = ImageFolder(root='./data/PlantVillage', transform=transform)
print(f"Dataset total size: {len(dataset)}.")
print(f"Classes: {dataset.classes}.")

# split dataset into trainset and testset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

Test_Loss = {}
Test_Acc = {}
print('\n----- start training -----')
tl_modes = ['no_pretrain', 'linear_probe', 'finetune']
# tl_modes = ['linear_probe', 'finetune']
for tl_mode in tl_modes:
    if tl_mode != "no_pretrain":
        if args.model == "resnet18":
            model = resnet18(weights="DEFAULT")
        elif args.model == "vgg16":
            model = vgg16(weights="DEFAULT")

        if tl_mode == "linear_probe":
            for name, param in model.named_parameters():
                if "bn" not in name:
                        param.requires_grad = False

            # in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
                nn.Linear(512*7*7, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, 10)
        )
    else:
        if args.model == "resnet18":
            model = resnet18()
        elif args.model == "vgg16":
            model = vgg16()
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    train_total_loss, train_total_acc, test_total_loss, test_total_acc = [], [], [], []

    best_acc = 0.0
    state = {
        'model': [],
        'acc': 0,
        'epoch': 0,
    }

    for epoch in range(num_epochs):
        model.train()
        train_total = 0
        train_loss, train_correct = 0, 0
        for idx, (x, label) in enumerate(tqdm(train_loader)):
            x, label = x.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(x)

            predicted = torch.argmax(output.data, 1)
            train_total += label.size(0)
            train_correct += (predicted == label).sum().item()

            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_total_loss.append(train_loss / len(train_loader))
        train_total_acc.append(100 * train_correct / train_total)

        test_total = 0
        test_loss, test_correct = 0, 0
        with torch.no_grad():
            model.eval()
            for idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                outputs = model(data)

                predicted = torch.argmax(outputs.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()

                test_loss += loss_fn(outputs, target).item()
        test_total_loss.append(test_loss / len(test_loader))
        test_total_acc.append(100 * test_correct / test_total)

        print('Epoch: {}/{}'.format(epoch+1, num_epochs))
        print('[Train] loss: {:.5f}, acc: {:.2f}%'.format(train_total_loss[-1], train_total_acc[-1]))
        print('[Test]  loss: {:.5f}, acc: {:.2f}%'.format(test_total_loss[-1], test_total_acc[-1]))

        # save checkpoint
        if test_total_acc[-1] > best_acc:
            best_acc = test_total_acc[-1]
            state['model'] = model.state_dict()
            state['acc'] = test_total_acc[-1]
            state['epoch'] = epoch
            print('- New checkpoint -')

    Test_Loss[tl_mode] = test_total_loss
    Test_Acc[tl_mode] = test_total_acc
    ckpt_name = args.model + "_" + tl_mode
    torch.save(state, './checkpoint/' + ckpt_name + '.pth')
    print(f'Best accuracy [{ckpt_name}]: {best_acc:.2f}%')

    # evaluate on best model
    model.load_state_dict(state['model'])
    y_true, y_pred = evaluate(model, test_loader, device)
    plot_confusion_matrix(y_true, y_pred, 10, 'true', ckpt_name)

# plot figures
show_train_result(num_epochs, Test_Loss, Test_Acc)
