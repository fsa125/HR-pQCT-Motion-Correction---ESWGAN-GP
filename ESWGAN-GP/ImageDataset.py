from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get and Load Image Datasets
class ImageDataset(Dataset):
    def __init__(self, files_hr,files_lr, hr_shape, device='cuda'):
        hr_height, hr_width = hr_shape
        mean = np.array([0.5])
        std = np.array([0.5])
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC), #BICUBIC
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.files_hr = files_hr
        self.files_lr = files_lr
        
    
    def __getitem__(self, index):
        img_hr = Image.open(self.files_hr[index % len(self.files_hr)]).convert('L')
        img_lr = Image.open(self.files_lr[index % len(self.files_lr)]).convert('L')

        img_lr = self.hr_transform(img_lr)
        img_hr = self.hr_transform(img_hr)
        img_lr = img_lr.to(device)
        img_hr = img_hr.to(device)
        
            
        return {"lr": img_lr, "hr": img_hr}
    
    def __len__(self):
        return len(self.files_hr)
    

class ImageDataset_test(Dataset):
    def __init__(self, files_hr, hr_shape, device='cuda'):
        hr_height, hr_width = hr_shape
        mean = np.array([0.5])
        std = np.array([0.5])
     
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC), #BICUBIC
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.files_hr = files_hr
        self.files_lr = files_hr
        
    
    def __getitem__(self, index):
        img_hr = Image.open(self.files_hr[index % len(self.files_hr)]).convert('L')
        img_lr = Image.open(self.files_lr[index % len(self.files_lr)]).convert('L')

        img_lr = self.hr_transform(img_lr)
        img_hr = self.hr_transform(img_hr)
        img_lr = img_lr.to(device)
        img_hr = img_hr.to(device)
        
            
        return {"lr": img_lr, "hr": img_hr}
    
    def __len__(self):
        return len(self.files_hr)