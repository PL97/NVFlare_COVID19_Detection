from torchvision.models import densenet121
import torch.nn as nn
import torch
import numpy as np

class Des121(nn.Module):
    def __init__(self, trunc=-1, num_classes=2):
        super(Des121, self).__init__()
        net = densenet121(pretrained=True)
        if trunc > 0:
            self.backbone=nn.Sequential(*list(net.children())[:4+trunc*2])
        else:
            self.backbone = net.features
            
        ## contruct classification head
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dim = np.prod(self.backbone(test_input).shape)

        self.clf_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )
            
    def forward(self, x):
        x = self.backbone(x)
        print(x.shape)
        x = self.clf_head(x)
        return x
        
        
        
        
if __name__ == "__main__":

    pass
    
    # ## test case
    # net = Des121(trunc=-1)
    # x = torch.randn(1, 3, 224, 224)
    # y = net(x)
    # print(x.shape, y.shape)
