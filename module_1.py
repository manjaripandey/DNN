import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def get_loaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # CIFAR-10H data
    cifar10h_probs = np.load('data/cifar10h-probs.npy')
    
    return trainloader, testloader, cifar10h_probs

def calculate_entropy(probs):
    # Add a small epsilon to prevent log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log2(probs), axis=1)

def plot_entropy_distribution(cifar10h_probs):
    entropy = calculate_entropy(cifar10h_probs)
    plt.figure(figsize=(10, 6))
    plt.hist(entropy, bins=50, density=True)
    plt.title('Entropy Distribution of CIFAR-10H')
    plt.xlabel('Entropy')
    plt.ylabel('Density')
    plt.savefig('entropy_distribution.png')

if __name__ == '__main__':
    trainloader, testloader, cifar10h_probs = get_loaders()
    
    # --- Entropy Calculation and Visualization ---
    plot_entropy_distribution(cifar10h_probs)

    # --- Data Visualization ---
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig('cifar10_sample.png')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    plt.figure(figsize=(16, 8))
    imshow(torchvision.utils.make_grid(images[:8]))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(8)))
