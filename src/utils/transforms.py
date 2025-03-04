from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
        # These will always be applied
        transforms.Resize((256, 256)),
        
        # These will be randomly applied with 50% probability
        transforms.RandomApply([
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.2, 
                    hue=0.1
                ),
            ])
        ], p=0.5),
        
        # These will always be applied
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]) 