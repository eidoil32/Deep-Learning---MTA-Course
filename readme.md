# Final project:

## Paint the circle net

This project identifies and paints circles given in RGB images. Specifically, each input image contains a rectangle, a triangle and a circle. The image is fed into a neural netork, whose outputs are the center and radius of the circle within the image. The domain considered is "[0,1]x[0,1] "box". PyTorch is the platform used. The goal is to paint the circle.

## Graph
![Training and Validation loss](https://raw.githubusercontent.com/eidoil32/Deep-Learning---MTA-Course/master/graph.jpg)

### Training ephocs 
```
Train Epoch: 1 		Loss: 0.561110 	Accuracy: 0.5150
Train Epoch: 2 		Loss: 0.509501 	Accuracy: 0.4494
Train Epoch: 3 		Loss: 0.517159 	Accuracy: 0.6758
Train Epoch: 4 		Loss: 0.514399 	Accuracy: 0.3988
Train Epoch: 5 		Loss: 0.510192 	Accuracy: 0.4412
Train Epoch: 6 		Loss: 0.508128 	Accuracy: 0.4198
Train Epoch: 7 		Loss: 0.429290 	Accuracy: 0.6662
Train Epoch: 8 		Loss: 0.283076 	Accuracy: 0.7945
Train Epoch: 9 		Loss: 0.165525 	Accuracy: 0.8205
Train Epoch: 10 	Loss: 0.131755 	Accuracy: 0.9299
Train Epoch: 11 	Loss: 0.099998 	Accuracy: 0.9238
Train Epoch: 12 	Loss: 0.080777 	Accuracy: 0.9290
Train Epoch: 13 	Loss: 0.073665 	Accuracy: 0.9411
Train Epoch: 14 	Loss: 0.068458 	Accuracy: 0.9597
Train Epoch: 15 	Loss: 0.059250 	Accuracy: 0.9463
Train Epoch: 16 	Loss: 0.055652 	Accuracy: 0.9333
Train Epoch: 17 	Loss: 0.059634 	Accuracy: 0.9532
Train Epoch: 18 	Loss: 0.051383 	Accuracy: 0.9713
Train Epoch: 19 	Loss: 0.046399 	Accuracy: 0.9655
Train Epoch: 20 	Loss: 0.051303 	Accuracy: 0.9710
Train Epoch: 21 	Loss: 0.040987 	Accuracy: 0.9737
Train Epoch: 22 	Loss: 0.047692 	Accuracy: 0.9621
Train Epoch: 23 	Loss: 0.038526 	Accuracy: 0.9753
Train Epoch: 24 	Loss: 0.034701 	Accuracy: 0.9777
Train Epoch: 25 	Loss: 0.032484 	Accuracy: 0.9687
```

## CircleNet
We choose an three layers network:
```
class CircleNet(nn.Module): 
    def __init__(self):
        super(CircleNet, self).__init__()
		
        self.conv1_1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3)
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=4)
        self.fc1 = nn.Linear(12 * 12 * 64, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 3)
                
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 12 * 12 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x) 
        x = F.relu(x)
        out = x
       
        return out                
```

### Results
![Example](https://raw.githubusercontent.com/eidoil32/Deep-Learning---MTA-Course/master/example.png)
