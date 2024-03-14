def load_ImageNet(ImageNet_PATH, batch_size=128, workers=8, pin_memory=True): 
    
    traindir = os.path.join(ImageNet_PATH, 'train1')
    valdir   = os.path.join(ImageNet_PATH, 'val2')
    print('traindir = ',traindir)
    print('valdir = ',valdir)
    
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalizer
        ])
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalizer
        ])
    )
    print('train_dataset = ',len(train_dataset))
    print('val_dataset   = ',len(val_dataset))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, train_dataset, val_dataset

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.JPEG')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      img_name = self.image_files[idx]
      image = Image.open(img_name)

      # Convert image to RGB if it's not already in RGB format
      if image.mode != 'RGB':
          image = image.convert('RGB')

      if self.transform:
          image = self.transform(image)

      # Assuming labels are derived from the filename or some other logic
      # Replace this with your actual label extraction logic
      label = 0  # Placeholder for label

      return image, label