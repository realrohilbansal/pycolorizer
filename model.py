import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, bn=True, dropout=False):
        super(UNetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False) if down \
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.dropout = nn.Dropout(0.5) if dropout else None
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        return F.relu(x) if self.down else F.relu(x, inplace=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = UNetBlock(1, 64, bn=False)  # Input is L channel (1 channel)
        self.down2 = UNetBlock(64, 128)
        self.down3 = UNetBlock(128, 256)
        self.down4 = UNetBlock(256, 512)
        self.down5 = UNetBlock(512, 512)
        self.down6 = UNetBlock(512, 512)
        self.down7 = UNetBlock(512, 512)
        self.down8 = UNetBlock(512, 512, bn=False)
        
        self.up1 = UNetBlock(512, 512, down=False, dropout=True)
        self.up2 = UNetBlock(1024, 512, down=False, dropout=True)
        self.up3 = UNetBlock(1024, 512, down=False, dropout=True)
        self.up4 = UNetBlock(1024, 512, down=False)
        self.up5 = UNetBlock(1024, 256, down=False)
        self.up6 = UNetBlock(512, 128, down=False)
        self.up7 = UNetBlock(256, 64, down=False)
        self.up8 = nn.ConvTranspose2d(128, 2, 4, 2, 1)  # Output is AB channels (2 channels)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        return torch.tanh(self.up8(torch.cat([u7, d1], 1)))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # Input is L+AB (3 channels)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

def init_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def create_models():
    try:
        print("Creating Generator...")
        generator = Generator()
        generator.apply(init_weights)
        print("Generator created successfully.")

        print("Creating Discriminator...")
        discriminator = Discriminator()
        discriminator.apply(init_weights)
        print("Discriminator created successfully.")

        return generator, discriminator
    except Exception as e:
        print(f"Error in creating models: {str(e)}")
        return None, None

def test_models():
    print("Testing models...")
    try:
        generator, discriminator = create_models()
        if generator is None or discriminator is None:
            raise Exception("Model creation failed")
        
        test_input_g = torch.randn(1, 1, 256, 256)
        test_output_g = generator(test_input_g)
        if test_output_g.shape != torch.Size([1, 2, 256, 256]):
            raise Exception(f"Unexpected generator output shape: {test_output_g.shape}")
        
        test_input_d = torch.randn(1, 3, 256, 256)
        test_output_d = discriminator(test_input_d)
        if test_output_d.shape != torch.Size([1, 1, 30, 30]):
            raise Exception(f"Unexpected discriminator output shape: {test_output_d.shape}")
        
        print("Model test passed.")
        return True
    except Exception as e:
        print(f"Model test failed: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        print("Initializing models...")
        generator, discriminator = create_models()

        if generator is None or discriminator is None:
            raise Exception("Failed to create models")

        if not test_models():
            raise Exception("Model testing failed")

        print("Model creation and testing completed successfully.")
    except Exception as e:
        print(f"Critical error in main execution: {str(e)}")