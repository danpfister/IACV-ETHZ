""" PyTorch interface to our dataset """
import torchvision.transforms as transforms


def get_transforms_train():
    """Return the transformations applied to images during training.
    
    See https://pytorch.org/vision/stable/transforms.html for a full list of 
    available transforms.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # convert image to a PyTorch Tensor
            transforms.RandomRotation(degrees=15),  # Random rotation within the range of -15 to +15 degrees
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Slight color changes
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize

        ]
    )
    return transform


def get_transforms_val():
    """Return the transformations applied to images during validation.

    Note: You do not need to change this function 
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # convert image to a PyTorch Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
        ]
    )
    return transform