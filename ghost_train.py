import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import argparse
from ghostNet_cifar10 import GhostNet


if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    args = parser.parse_args()


    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    writerPath = '../dnn_result/record/{}-{}-{}'.format(num_epochs, batch_size, learning_rate)
    ckptPath = '../dnn_result/ckpt/{}-{}-{}.ckpt'.format(num_epochs, batch_size, learning_rate)

    print('num_epochs:{}'.format(num_epochs))
    print('batch_size:{}'.format(batch_size))
    print('learning_rate:{}'.format(learning_rate))
    print('writerPath:{}'.format(writerPath))
    print('ckptPaht:{}'.format(ckptPath))

    writer = SummaryWriter(writerPath)

    # zhq
    PATH = None
    # /zhq


    # Image preprocessing modules
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])

    # CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                                 train=True,
                                                 transform=transform,
                                                 download=False)

    test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                                train=False,
                                                transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)



    model = GhostNet()
    model.to(device)

    # zhq
    if PATH:
        model.load_state_dict(torch.load(PATH))
    # /zhq


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # For updating learning rate


    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                writer.add_scalar('loss', loss.item(), global_step=i+epoch*total_step)

        # zhq
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            acc = correct / total
            print("Epoch:{}, test accuracy:{}".format(epoch, acc))
            writer.add_scalar('test accuracy', acc, global_step=epoch)
            # /zhq

            # Decay learning rate
            if (epoch+1) % 20 == 0:
                curr_lr /= 3
                update_lr(optimizer, curr_lr)

    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), '{}'.format(ckptPath))