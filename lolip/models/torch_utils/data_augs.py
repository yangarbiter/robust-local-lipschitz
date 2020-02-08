from torchvision import datasets, transforms


def aug01():
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = None
    return transform_train, transform_test

def aug02():
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = None
    return transform_train, transform_test
