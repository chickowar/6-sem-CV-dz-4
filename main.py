from src.datasets import get_mnist, get_svhn
from src.models import SimpleCNN
from src.train_eval import train, evaluate



if __name__ == '__main__':
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    mnist_loader = get_mnist(train=True)
    mnist_test = get_mnist(train=False)
    svhn_test = get_svhn()
    model = SimpleCNN()

    train(model, mnist_loader, device, epochs=10)

    print("Evaluation on MNIST (same domain):")
    print(evaluate(model, mnist_test, device))

    print("Evaluation on SVHN (cross-domain):")
    print(evaluate(model, svhn_test, device))