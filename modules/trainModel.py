try:
    import os
    import sys
    import time
    import torch
    import torchvision
    import numpy as np
    import pandas as pd
    from PIL import Image
    import torch.nn as nn
    import torch.optim as optim
    from torch.autograd import Variable
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
except ImportError as e:
    print("{} MODEL TRAINING MSSG: Fail Import Module: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e))
    sys.exit(1)

class Dataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_name = pd.read_csv(csv_file)
        self.len = self.data_name.shape[0] 
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        img_name = self.data_dir + self.data_name.iloc[idx, 1]
        image = Image.open(img_name).convert('RGB')
        y = self.data_name.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        return image, y

class Discriminator(nn.Module):
    def __init__(self, img_channels, features_d):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d, features_d*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d*2, features_d*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d*4, features_d*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d*8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, noise_channels, img_channels, features_g):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_channels, features_g*16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(features_g*16),
            nn.ReLU(),
            nn.ConvTranspose2d(features_g*16, features_g*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g*8),
            nn.ReLU(),
            nn.ConvTranspose2d(features_g*8, features_g*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g*4),
            nn.ReLU(),
            nn.ConvTranspose2d(features_g*4, features_g*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g*2),
            nn.ReLU(),
            nn.ConvTranspose2d(features_g*2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)

class Train():
    def __init__(self, config):
        self._config = config
        self._path_train_csvFile = self._config['path_train_csvFile']
        self._path_train_data = self._config['path_train_data']
        self._lr = self._config['lr']
        self._batch_size = self._config['batch_size']
        self._image_size = self._config['image_size']
        self._img_channels = self._config['img_channels']
        self._noise_channels = self._config['noise_channels']
        self._features_d = self._config['features_d']
        self._features_g = self._config['features_g']
        self._num_epochs = self._config['num_epochs']
    
    def trainer(self):
        if not os.path.exists('ckpt'):
            os.makedirs('ckpt')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transformations = transforms.Compose([
                                            transforms.Resize((self._image_size, self._image_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,),(0.5,)),
                                            ])
        dataset = Dataset(
                            csv_file = self._path_train_csvFile,
                            data_dir = self._path_train_data,
                            transform=transformations
                            )
        criterion = nn.BCELoss()
        dataloader = DataLoader(dataset=dataset, batch_size=self._batch_size, shuffle=True)

        netD = Discriminator(self._img_channels, self._features_d).to(device)
        netG = Generator(self._noise_channels, self._img_channels, self._features_g).to(device)

        optimizerD = optim.Adam(netD.parameters(), lr=self._lr)
        optimizerG = optim.Adam(netG.parameters(), lr=self._lr)

        netD.train()
        netG.train()

        print('MODEL MSSG: Starting Trainig...')
        loss_ref = 1
        t_s = time.time()
        for epoch in range(self._num_epochs):
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(device)
                batch_size = data.shape[0]

                netD.zero_grad()
                label = (torch.ones(batch_size)*0.9).to(device)
                output = netD(data).reshape(-1)
                lossD_real = criterion(output, label)
                D_x = output.mean().item()

                noise = torch.randn(batch_size, self._noise_channels, 1, 1).to(device)
                fake = netG(noise)
                label = (torch.ones(batch_size)*0.1).to(device)
                output = netD(fake.detach()).reshape(-1)
                lossD_fake = criterion(output, label)

                lossD = lossD_real + lossD_fake
                lossD.backward()
                optimizerD.step()

                netG.zero_grad()
                label = torch.ones(batch_size).to(device)
                output = netD(fake).reshape(-1)
                lossG = criterion(output, label)
                lossG.backward()
                optimizerG.step()

                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch}/{self._num_epochs}] Batch {batch_idx}/{len(dataloader)}  \
                        Loss D: {lossD:.4f}, Loss G: {lossG:.4f} D(x): {D_x:.4f}')
                    tensorboardLogs = {'lossG': lossG, 'lossD': lossD}
                    print(tensorboardLogs)
                state = {
                        'epoch': epoch+1,
                        'D':netD.state_dict(),
                        'G':netG.state_dict(),
                        'd_optim': optimizerD.state_dict(),
                        'g_optim' : optimizerG.state_dict()
                            }
                if lossD_real <= loss_ref:
                    loss_ref = lossD_real
                    torch.save(state, 'ckpt/ganAn{:06d}.pth'.format(epoch))
        torch.save(state, 'ckpt/ganAn{:06d}.pth'.format(self._num_epochs))        
        final_time= time.time() - t_s
        print('MSSG: Training complete: {:.0f}h {:.0f}m {:.0f}s'.format((final_time//3600), 
                                                                (final_time // 60)- (final_time // 3600)*60,
                                                                (final_time%60)))
class Predictions():
    def __init__(self, config):
        self._config = config
        self._path_model = self._config['path_model']
        self._path_test = self._config['path_test']
        self._model = self._config['model_name']
        self._image_size = self._config['image_size']

    def predictions(self, image):
        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'
        modelD = Discriminator(3, 16)
        state = torch.load(self._path_model+self._model, map_location=map_location)
        modelD.load_state_dict(state['D'])
        devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        modelD = modelD.to(devices)
        modelD.eval()
        transformations = transforms.Compose([
                                            transforms.Resize((self._image_size, self._image_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,),(0.5,)),
                                            ])
        print('image', image)
        img = Image.open(image)
        img = np.array(img)
        tensor_to_image = transforms.ToPILImage()
        img = tensor_to_image(img)
        image_tensor = transformations(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(devices)
        output = modelD(input)
        predict = round(modelD(input).item())
        return output, predict
    
    def test(self):
        list_img = [img.path for img in os.scandir(self._path_test) if img.name.endswith('.jpg')]
        print('lista', list_img)

        for img in list_img:
            output, predict = self.predictions(img)
            prob = torch.sigmoid(output).cpu().detach().numpy().item()
            print('PREDICT***', predict)
            print('PROB', prob*100)
