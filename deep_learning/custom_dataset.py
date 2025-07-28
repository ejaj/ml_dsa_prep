from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
labels = [0, 1, 0]

image_dir = "dataset"  # path to your image folder
dataset = CustomImageDataset(image_dir=image_dir, labels=labels, transform=transform)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for images, labels in dataloader:
    print(images.shape)  # e.g., torch.Size([2, 3, 128, 128])
    print(labels)        # e.g., tensor([0, 1])
    break  # Just show first batch