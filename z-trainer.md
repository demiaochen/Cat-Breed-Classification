# report

## notes

* model 3 training seems good, loss decrease steadily, but testing accuracy improves hard around 75%

* kernel size 5 3 3 3 works better than any other size tested 

* maxpool performs better than avgpool

* lr with 0.0003 ~ 0.0005 can train well towards minimum loss


**TODO**：find technique to improve CNN, or try another architecture

## brief

**bold** means it has detail in the following section

model 1: kernel size 5 3 3 3 -> 71%

model 2: change kernel size to 3 2 2 2 -> 69%

**model 3**: kernel size 5 3 3 3 and add batchnorm -> 78%

**model 4**: kernel size 5 3 2 2, increase lr from 0.0005 to 0.001 -> 76%

**model 5**: model 3 with lr 0.0003 -> 78%  200 epochs loss: 19

model 6: model 5 but avgpool -> 72%

**model 7**: model 3 modified: no dropout before output -> 75%

**model 8**: model 3 modified: decreasing dropout -> 79%

**model 9**: model 3 modified: decreasing dropout with higher dropout and fc paras -> 78%

## details

### model 3

```shell
torch.Size([3, 80, 80])
batch: 64
learning_rate: 0.0005
train_val_split: 0.8
epochs: 100
Compose(
    RandomResizedCrop(size=(80, 80), scale=(0.75, 1.0), ratio=(0.75, 1.3), interpolation=bilinear)
    RandomHorizontalFlip(p=0.5)
    RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0)
    ColorJitter(brightness=[0.6, 1.4], contrast=[0.7, 1.3], saturation=[0.7, 1.3], hue=[-0.2, 0.2])
    RandomPosterize(bits=3,p=0.4)
    RandomEqualize(p=0.1)
    RandomGrayscale(p=0.1)
    RandomPerspective(p=0.1)
    ToTensor()
)
SimpleCNN(
  (conv_layers): Sequential(
    (0): Conv2d(3, 30, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(30, 120, kernel_size=(3, 3), stride=(1, 1))
    (5): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(120, 360, kernel_size=(3, 3), stride=(1, 1))
    (9): BatchNorm2d(360, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
    (11): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (12): Conv2d(360, 540, kernel_size=(3, 3), stride=(1, 1))
    (13): BatchNorm2d(540, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU()
    (15): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (fc_layers): Sequential(
    (0): Dropout(p=0.4, inplace=False)
    (1): Linear(in_features=4860, out_features=1000, bias=True)
    (2): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Dropout(p=0.4, inplace=False)
    (5): Linear(in_features=1000, out_features=1000, bias=True)
    (6): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): ReLU()
    (8): Dropout(p=0.4, inplace=False)
    (9): Linear(in_features=1000, out_features=8, bias=True)
  )
)
+-----------------------+------------+
|        Modules        | Parameters |
+-----------------------+------------+
|  conv_layers.0.weight |    2250    |
|   conv_layers.0.bias  |     30     |
|  conv_layers.1.weight |     30     |
|   conv_layers.1.bias  |     30     |
|  conv_layers.4.weight |   32400    |
|   conv_layers.4.bias  |    120     |
|  conv_layers.5.weight |    120     |
|   conv_layers.5.bias  |    120     |
|  conv_layers.8.weight |   388800   |
|   conv_layers.8.bias  |    360     |
|  conv_layers.9.weight |    360     |
|   conv_layers.9.bias  |    360     |
| conv_layers.12.weight |  1749600   |
|  conv_layers.12.bias  |    540     |
| conv_layers.13.weight |    540     |
|  conv_layers.13.bias  |    540     |
|   fc_layers.1.weight  |  4860000   |
|    fc_layers.1.bias   |    1000    |
|   fc_layers.2.weight  |    1000    |
|    fc_layers.2.bias   |    1000    |
|   fc_layers.5.weight  |  1000000   |
|    fc_layers.5.bias   |    1000    |
|   fc_layers.6.weight  |    1000    |
|    fc_layers.6.bias   |    1000    |
|   fc_layers.9.weight  |    8000    |
|    fc_layers.9.bias   |     8      |
+-----------------------+------------+
Total Trainable Params: 8050208
Start training...
ep 1, loss: 178.43, 6400 train 31.84%, 1600 test 47.00%
ep 2, loss: 162.57, 6400 train 38.94%, 1600 test 44.44%
ep 3, loss: 152.61, 6400 train 42.28%, 1600 test 47.56%
ep 4, loss: 145.92, 6400 train 44.97%, 1600 test 52.69%
ep 5, loss: 139.22, 6400 train 48.83%, 1600 test 56.19%
ep 6, loss: 132.59, 6400 train 51.22%, 1600 test 54.19%
ep 7, loss: 129.17, 6400 train 52.41%, 1600 test 58.81%
ep 8, loss: 125.60, 6400 train 53.92%, 1600 test 53.06%
ep 9, loss: 120.11, 6400 train 56.52%, 1600 test 52.94%
ep 10, loss: 117.25, 6400 train 58.25%, 1600 test 58.50%
[[ 97.   1.   5.  60.   7.   2.   8.  16.]
 [  0.  59.  28.   3.  44.   7.  12.  10.]
 [  5.  13. 137.   9.  29.   6.   3.   7.]
 [  2.   2.  21. 135.  61.   5.   0.   3.]
 [  0.   4.  57.   5. 127.   2.   0.   1.]
 [  0.   1.  33.  15.   4. 134.   2.   1.]
 [ 10.  10.  13.  16.  10.  16. 113.   6.]
 [  5.  29.   9.  14.  18.   1.  13. 134.]]
   Model saved to checkModel.pth
ep 11, loss: 115.05, 6400 train 57.70%, 1600 test 58.00%
ep 12, loss: 112.12, 6400 train 59.48%, 1600 test 48.94%
ep 13, loss: 109.04, 6400 train 60.64%, 1600 test 58.94%
ep 14, loss: 104.65, 6400 train 62.50%, 1600 test 65.19%
ep 15, loss: 102.83, 6400 train 63.19%, 1600 test 63.25%
ep 16, loss: 102.72, 6400 train 63.28%, 1600 test 66.75%
ep 17, loss: 98.45, 6400 train 64.61%, 1600 test 62.12%
ep 18, loss: 97.74, 6400 train 65.27%, 1600 test 67.81%
ep 19, loss: 96.38, 6400 train 65.08%, 1600 test 68.81%
ep 20, loss: 92.64, 6400 train 67.11%, 1600 test 70.00%
[[148.   3.   2.   9.   4.   1.  13.  16.]
 [  1.  91.  11.   0.  26.   8.  12.  14.]
 [  8.  21. 104.  11.  47.   6.   3.   9.]
 [ 17.   2.  13. 149.  35.   4.   4.   5.]
 [  1.   9.   9.   2. 168.   1.   2.   4.]
 [  2.   7.   5.   6.   3. 156.   8.   3.]
 [ 18.  11.   3.   1.   6.   8. 141.   6.]
 [ 13.  18.   5.   3.   6.   1.  14. 163.]]
   Model saved to checkModel.pth
ep 21, loss: 91.77, 6400 train 67.86%, 1600 test 69.50%
ep 22, loss: 88.13, 6400 train 68.73%, 1600 test 71.56%
ep 23, loss: 89.11, 6400 train 68.02%, 1600 test 68.19%
ep 24, loss: 85.08, 6400 train 69.22%, 1600 test 70.62%
ep 25, loss: 84.56, 6400 train 70.78%, 1600 test 69.75%
ep 26, loss: 84.66, 6400 train 70.16%, 1600 test 73.25%
ep 27, loss: 81.47, 6400 train 71.48%, 1600 test 72.88%
ep 28, loss: 80.29, 6400 train 71.27%, 1600 test 67.88%
ep 29, loss: 79.64, 6400 train 71.86%, 1600 test 72.69%
ep 30, loss: 78.41, 6400 train 72.06%, 1600 test 74.44%
[[151.   0.   6.  14.   3.   2.   7.  13.]
 [  0.  92.   9.   5.  16.  12.  10.  19.]
 [  3.  13. 130.  11.  28.  10.   3.  11.]
 [ 14.   0.  10. 181.  17.   3.   2.   2.]
 [  2.   8.   9.  10. 165.   0.   1.   1.]
 [  2.   3.   8.   8.   3. 161.   2.   3.]
 [ 14.   9.   8.   9.   3.  10. 130.  11.]
 [ 12.  13.   3.   3.   3.   0.   8. 181.]]
   Model saved to checkModel.pth
ep 31, loss: 77.13, 6400 train 72.52%, 1600 test 72.81%
ep 32, loss: 76.50, 6400 train 72.73%, 1600 test 71.94%
ep 33, loss: 75.74, 6400 train 73.08%, 1600 test 73.88%
ep 34, loss: 72.09, 6400 train 74.53%, 1600 test 68.62%
ep 35, loss: 71.87, 6400 train 74.31%, 1600 test 73.75%
ep 36, loss: 70.49, 6400 train 74.81%, 1600 test 71.25%
ep 37, loss: 71.36, 6400 train 74.80%, 1600 test 73.25%
ep 38, loss: 70.31, 6400 train 74.80%, 1600 test 72.50%
ep 39, loss: 66.56, 6400 train 75.67%, 1600 test 70.75%
ep 40, loss: 67.23, 6400 train 76.03%, 1600 test 74.25%
[[154.   0.   3.   9.   4.   0.   6.  20.]
 [  0.  72.  13.   7.  24.   7.  18.  22.]
 [  4.   5. 138.  11.  24.   9.  10.   8.]
 [ 21.   0.   6. 172.  19.   1.   5.   5.]
 [  2.   1.  16.  10. 162.   0.   3.   2.]
 [  1.   6.   6.  11.   3. 144.  18.   1.]
 [ 13.   6.   1.   5.   1.   2. 158.   8.]
 [  9.   6.   6.   2.   4.   0.   8. 188.]]
   Model saved to checkModel.pth
ep 41, loss: 65.04, 6400 train 76.88%, 1600 test 72.31%
ep 42, loss: 62.93, 6400 train 76.83%, 1600 test 74.56%
ep 43, loss: 64.22, 6400 train 77.84%, 1600 test 75.69%
ep 44, loss: 63.06, 6400 train 77.50%, 1600 test 73.06%
ep 45, loss: 62.65, 6400 train 77.41%, 1600 test 73.56%
ep 46, loss: 59.86, 6400 train 78.62%, 1600 test 75.69%
ep 47, loss: 59.62, 6400 train 79.56%, 1600 test 75.12%
ep 48, loss: 60.95, 6400 train 78.86%, 1600 test 76.25%
ep 49, loss: 58.40, 6400 train 79.02%, 1600 test 71.88%
ep 50, loss: 57.27, 6400 train 79.77%, 1600 test 74.12%
[[179.   1.   0.   3.   0.   0.   0.  13.]
 [  1. 101.  14.   6.   8.   7.   4.  22.]
 [  6.  23. 143.   8.  12.   7.   0.  10.]
 [ 27.   3.   6. 167.  10.   1.   1.  14.]
 [  4.  22.  20.   3. 144.   0.   0.   3.]
 [  1.   6.   9.   5.   2. 156.   3.   8.]
 [ 36.  13.   7.   7.   2.   5. 110.  14.]
 [ 14.  13.   2.   1.   6.   1.   0. 186.]]
   Model saved to checkModel.pth
ep 51, loss: 55.74, 6400 train 80.23%, 1600 test 73.38%
ep 52, loss: 55.58, 6400 train 80.09%, 1600 test 75.75%
ep 53, loss: 56.05, 6400 train 79.98%, 1600 test 73.25%
ep 54, loss: 54.40, 6400 train 80.52%, 1600 test 72.44%
ep 55, loss: 53.52, 6400 train 81.06%, 1600 test 73.38%
ep 56, loss: 51.92, 6400 train 81.14%, 1600 test 74.12%
ep 57, loss: 54.67, 6400 train 81.12%, 1600 test 74.31%
ep 58, loss: 53.21, 6400 train 80.69%, 1600 test 74.62%
ep 59, loss: 52.29, 6400 train 81.17%, 1600 test 75.12%
ep 60, loss: 50.65, 6400 train 81.94%, 1600 test 73.75%
[[135.   0.   0.   9.   3.   3.   3.  43.]
 [  0.  61.  18.   7.  17.  19.   5.  36.]
 [  4.   5. 144.  10.  14.  15.   2.  15.]
 [ 12.   0.   5. 182.   5.   6.   2.  17.]
 [  1.   5.  17.  11. 155.   1.   1.   5.]
 [  1.   2.   4.   5.   0. 170.   1.   7.]
 [  9.   6.   8.   7.   2.  11. 126.  25.]
 [  3.   4.   3.   2.   2.   1.   1. 207.]]
   Model saved to checkModel.pth
ep 61, loss: 48.50, 6400 train 82.84%, 1600 test 74.25%
ep 62, loss: 49.79, 6400 train 82.52%, 1600 test 74.56%
ep 63, loss: 47.90, 6400 train 82.28%, 1600 test 75.56%
ep 64, loss: 47.63, 6400 train 83.12%, 1600 test 75.00%
ep 65, loss: 47.75, 6400 train 83.39%, 1600 test 75.56%
ep 66, loss: 46.25, 6400 train 83.64%, 1600 test 75.56%
ep 67, loss: 44.73, 6400 train 84.38%, 1600 test 71.56%
ep 68, loss: 44.69, 6400 train 84.08%, 1600 test 70.44%
ep 69, loss: 44.52, 6400 train 84.36%, 1600 test 73.12%
ep 70, loss: 46.48, 6400 train 83.78%, 1600 test 75.31%
[[182.   1.   0.   5.   1.   0.   3.   4.]
 [  3.  70.  17.   7.  24.  13.  15.  14.]
 [  4.   6. 146.  12.  14.  14.   3.  10.]
 [ 21.   0.   4. 190.   9.   2.   1.   2.]
 [  3.   3.  10.   8. 168.   1.   2.   1.]
 [  2.   3.   6.   6.   1. 165.   6.   1.]
 [ 33.   9.   4.   8.   1.   8. 124.   7.]
 [ 33.  10.   5.   5.   5.   1.   4. 160.]]
   Model saved to checkModel.pth
ep 71, loss: 43.96, 6400 train 84.83%, 1600 test 75.12%
ep 72, loss: 42.71, 6400 train 84.70%, 1600 test 73.62%
ep 73, loss: 42.38, 6400 train 85.27%, 1600 test 75.44%
ep 74, loss: 42.06, 6400 train 85.42%, 1600 test 74.81%
ep 75, loss: 41.51, 6400 train 85.16%, 1600 test 70.31%
ep 76, loss: 39.48, 6400 train 85.97%, 1600 test 75.75%
ep 77, loss: 42.09, 6400 train 84.58%, 1600 test 75.06%
ep 78, loss: 40.83, 6400 train 85.70%, 1600 test 76.50%
ep 79, loss: 38.34, 6400 train 86.81%, 1600 test 75.56%
ep 80, loss: 38.21, 6400 train 86.45%, 1600 test 75.31%
[[158.   0.   0.   7.   3.   1.   0.  27.]
 [  0.  81.   5.   3.  31.  12.   7.  24.]
 [  6.   8. 115.  10.  33.  18.   5.  14.]
 [ 13.   1.   1. 175.  17.   7.   1.  14.]
 [  2.   5.   2.   4. 177.   2.   1.   3.]
 [  1.   4.   1.   5.   3. 170.   1.   5.]
 [ 14.   8.   3.   7.   3.   9. 130.  20.]
 [  8.   7.   2.   2.   4.   1.   0. 199.]]
   Model saved to checkModel.pth
ep 81, loss: 37.81, 6400 train 86.77%, 1600 test 75.69%
ep 82, loss: 39.23, 6400 train 86.09%, 1600 test 75.06%
ep 83, loss: 38.53, 6400 train 86.56%, 1600 test 74.00%
ep 84, loss: 37.00, 6400 train 86.98%, 1600 test 76.44%
ep 85, loss: 36.16, 6400 train 87.22%, 1600 test 74.00%
ep 86, loss: 38.94, 6400 train 86.58%, 1600 test 75.81%
ep 87, loss: 34.76, 6400 train 87.73%, 1600 test 76.75%
ep 88, loss: 35.26, 6400 train 87.38%, 1600 test 76.81%
ep 89, loss: 36.41, 6400 train 87.52%, 1600 test 74.44%
ep 90, loss: 35.09, 6400 train 87.33%, 1600 test 76.12%
[[155.   1.   3.  19.   3.   3.   2.  10.]
 [  0.  75.  14.   8.  26.  21.   3.  16.]
 [  3.   8. 153.  10.  12.  15.   1.   7.]
 [  8.   0.   4. 194.  17.   5.   0.   1.]
 [  2.   3.   9.   3. 176.   2.   1.   0.]
 [  1.   3.   3.   5.   1. 174.   1.   2.]
 [ 12.   9.   7.  15.   6.  19. 116.  10.]
 [ 10.  10.   9.   3.  11.   3.   2. 175.]]
   Model saved to checkModel.pth
ep 91, loss: 33.69, 6400 train 88.19%, 1600 test 76.12%
ep 92, loss: 34.59, 6400 train 87.34%, 1600 test 75.88%
ep 93, loss: 35.44, 6400 train 87.77%, 1600 test 75.50%
ep 94, loss: 33.21, 6400 train 88.61%, 1600 test 77.31%
ep 95, loss: 32.52, 6400 train 88.50%, 1600 test 77.38%
ep 96, loss: 32.27, 6400 train 88.73%, 1600 test 77.75%
ep 97, loss: 33.07, 6400 train 88.61%, 1600 test 75.19%
ep 98, loss: 33.02, 6400 train 88.55%, 1600 test 77.88%
ep 99, loss: 31.64, 6400 train 88.38%, 1600 test 72.81%
ep 100, loss: 31.48, 6400 train 88.72%, 1600 test 78.56%
[[163.   0.   5.  11.   0.   1.   3.  13.]
 [  0.  90.  13.   8.  11.  16.   9.  16.]
 [  3.   6. 153.  11.  11.  10.   3.  12.]
 [ 10.   0.   7. 200.   5.   4.   1.   2.]
 [  1.  10.  14.   5. 160.   3.   1.   2.]
 [  2.   1.   5.   7.   0. 169.   4.   2.]
 [ 12.  11.   5.   9.   0.   5. 140.  12.]
 [ 11.  12.   4.   5.   4.   1.   4. 182.]]
   Model saved to checkModel.pth
   Model saved to savedModel.pth
```
### model 4

 ``` shell
torch.Size([3, 80, 80])
batch size: 64
learning rate: 0.001
train_val_split: 0.8
epochs: 100
Compose(
    RandomResizedCrop(size=(80, 80), scale=(0.75, 1.0), ratio=(0.75, 1.3), interpolation=bilinear)
    RandomHorizontalFlip(p=0.5)
    RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0)
    ColorJitter(brightness=[0.6, 1.4], contrast=[0.7, 1.3], saturation=[0.7, 1.3], hue=[-0.2, 0.2])
    RandomPosterize(bits=3,p=0.4)
    RandomEqualize(p=0.1)
    RandomGrayscale(p=0.1)
    RandomPerspective(p=0.1)
    ToTensor()
)
SimpleCNN(
  (conv_layers): Sequential(
    (0): Conv2d(3, 30, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(30, 120, kernel_size=(3, 3), stride=(1, 1))
    (5): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(120, 360, kernel_size=(2, 2), stride=(1, 1))
    (9): BatchNorm2d(360, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
    (11): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (12): Conv2d(360, 540, kernel_size=(2, 2), stride=(1, 1))
    (13): BatchNorm2d(540, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU()
    (15): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (fc_layers): Sequential(
    (0): Dropout(p=0.4, inplace=False)
    (1): Linear(in_features=8640, out_features=1000, bias=True)
    (2): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Dropout(p=0.4, inplace=False)
    (5): Linear(in_features=1000, out_features=1000, bias=True)
    (6): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): ReLU()
    (8): Dropout(p=0.4, inplace=False)
    (9): Linear(in_features=1000, out_features=8, bias=True)
  )
)
+-----------------------+------------+
|        Modules        | Parameters |
+-----------------------+------------+
|  conv_layers.0.weight |    2250    |
|   conv_layers.0.bias  |     30     |
|  conv_layers.1.weight |     30     |
|   conv_layers.1.bias  |     30     |
|  conv_layers.4.weight |   32400    |
|   conv_layers.4.bias  |    120     |
|  conv_layers.5.weight |    120     |
|   conv_layers.5.bias  |    120     |
|  conv_layers.8.weight |   172800   |
|   conv_layers.8.bias  |    360     |
|  conv_layers.9.weight |    360     |
|   conv_layers.9.bias  |    360     |
| conv_layers.12.weight |   777600   |
|  conv_layers.12.bias  |    540     |
| conv_layers.13.weight |    540     |
|  conv_layers.13.bias  |    540     |
|   fc_layers.1.weight  |  8640000   |
|    fc_layers.1.bias   |    1000    |
|   fc_layers.2.weight  |    1000    |
|    fc_layers.2.bias   |    1000    |
|   fc_layers.5.weight  |  1000000   |
|    fc_layers.5.bias   |    1000    |
|   fc_layers.6.weight  |    1000    |
|    fc_layers.6.bias   |    1000    |
|   fc_layers.9.weight  |    8000    |
|    fc_layers.9.bias   |     8      |
+-----------------------+------------+
Total Trainable Params: 10642208
Start training...
ep 1, loss: 180.50, 6400 train 31.36%, 1600 test 35.94%
ep 2, loss: 164.65, 6400 train 38.38%, 1600 test 30.75%
ep 3, loss: 155.41, 6400 train 41.98%, 1600 test 37.56%
ep 4, loss: 147.44, 6400 train 44.81%, 1600 test 46.81%
ep 5, loss: 142.81, 6400 train 47.33%, 1600 test 48.88%
ep 6, loss: 135.37, 6400 train 50.08%, 1600 test 53.75%
ep 7, loss: 132.94, 6400 train 51.59%, 1600 test 54.81%
ep 8, loss: 127.54, 6400 train 53.58%, 1600 test 57.12%
ep 9, loss: 123.87, 6400 train 54.66%, 1600 test 50.44%
ep 10, loss: 120.83, 6400 train 55.88%, 1600 test 55.06%
[[147.   0.   4.   7.   3.   1.  10.  11.]
 [  1.  56.  41.   5.  23.  22.  36.  29.]
 [ 10.   7. 143.   5.  14.  25.   4.   4.]
 [ 17.   1.  43. 110.   6.  34.   7.   0.]
 [  5.  22. 104.   8.  55.  10.   2.   2.]
 [  3.   2.  13.   8.   3. 153.   4.   0.]
 [ 20.   5.  23.   6.   6.  23. 105.   7.]
 [ 32.  15.  10.   4.   4.   0.   8. 112.]]
   Model saved to checkModel.pth
ep 11, loss: 116.15, 6400 train 57.25%, 1600 test 49.31%
ep 12, loss: 114.02, 6400 train 59.23%, 1600 test 53.44%
ep 13, loss: 110.60, 6400 train 59.59%, 1600 test 53.00%
ep 14, loss: 109.38, 6400 train 60.72%, 1600 test 48.94%
ep 15, loss: 105.27, 6400 train 62.11%, 1600 test 61.50%
ep 16, loss: 102.76, 6400 train 63.53%, 1600 test 51.12%
ep 17, loss: 100.13, 6400 train 64.61%, 1600 test 56.38%
ep 18, loss: 97.93, 6400 train 65.36%, 1600 test 62.31%
ep 19, loss: 95.71, 6400 train 65.78%, 1600 test 57.44%
ep 20, loss: 94.37, 6400 train 65.36%, 1600 test 57.31%
[[137.   0.  17.  16.   3.   0.   4.   6.]
 [  2.  61.  49.   3.  54.  16.   9.  19.]
 [  7.   1. 150.   3.  38.  11.   0.   2.]
 [  4.   0.  49.  97.  49.  17.   2.   0.]
 [  0.   1.  52.   1. 151.   2.   0.   1.]
 [  0.   3.  26.   7.  10. 140.   0.   0.]
 [ 17.   5.  42.  16.   6.  31.  76.   2.]
 [ 29.  18.  10.   7.  13.   2.   1. 105.]]
   Model saved to checkModel.pth
ep 21, loss: 92.08, 6400 train 66.94%, 1600 test 55.75%
ep 22, loss: 90.24, 6400 train 67.50%, 1600 test 57.94%
ep 23, loss: 88.35, 6400 train 68.36%, 1600 test 61.44%
ep 24, loss: 86.39, 6400 train 68.86%, 1600 test 59.00%
ep 25, loss: 84.16, 6400 train 70.41%, 1600 test 65.25%
ep 26, loss: 82.34, 6400 train 70.17%, 1600 test 61.44%
ep 27, loss: 84.66, 6400 train 69.91%, 1600 test 61.50%
ep 28, loss: 78.76, 6400 train 71.45%, 1600 test 56.44%
ep 29, loss: 77.92, 6400 train 72.44%, 1600 test 66.62%
ep 30, loss: 79.32, 6400 train 72.58%, 1600 test 64.25%
[[159.   0.   4.   9.   0.   1.   5.   5.]
 [  4.  68.  31.  10.  10.  38.  18.  34.]
 [ 10.   4. 157.  10.   6.  19.   0.   6.]
 [ 18.   0.  13. 158.   5.  21.   1.   2.]
 [  5.   6.  65.  18. 102.  10.   0.   2.]
 [  2.   1.   7.   7.   2. 166.   1.   0.]
 [ 21.   2.  12.  15.   2.  46.  93.   4.]
 [ 31.   9.   6.   7.   1.   4.   2. 125.]]
   Model saved to checkModel.pth
ep 31, loss: 74.94, 6400 train 72.80%, 1600 test 65.31%
ep 32, loss: 74.98, 6400 train 72.78%, 1600 test 65.31%
ep 33, loss: 72.93, 6400 train 74.03%, 1600 test 58.63%
ep 34, loss: 73.04, 6400 train 74.25%, 1600 test 66.88%
ep 35, loss: 70.74, 6400 train 74.81%, 1600 test 63.81%
ep 36, loss: 68.38, 6400 train 75.84%, 1600 test 60.62%
ep 37, loss: 70.70, 6400 train 74.86%, 1600 test 70.31%
ep 38, loss: 65.65, 6400 train 76.12%, 1600 test 69.06%
ep 39, loss: 64.50, 6400 train 77.28%, 1600 test 70.12%
ep 40, loss: 65.59, 6400 train 76.84%, 1600 test 69.88%
[[118.   3.   4.  26.   2.   1.  13.  16.]
 [  0. 137.   8.   6.  16.  17.  10.  19.]
 [  8.  20. 111.  13.  31.  19.   4.   6.]
 [  2.   2.   5. 173.   9.  22.   5.   0.]
 [  0.  20.   9.  15. 154.   6.   0.   4.]
 [  0.   5.   1.  12.   5. 162.   1.   0.]
 [  3.  21.   5.   9.   4.  21. 124.   8.]
 [  8.  22.   4.   7.   1.   3.   1. 139.]]
   Model saved to checkModel.pth
ep 41, loss: 63.13, 6400 train 77.30%, 1600 test 66.94%
ep 42, loss: 62.15, 6400 train 78.03%, 1600 test 68.44%
ep 43, loss: 60.15, 6400 train 78.56%, 1600 test 64.81%
ep 44, loss: 60.60, 6400 train 78.66%, 1600 test 69.44%
ep 45, loss: 57.91, 6400 train 79.75%, 1600 test 71.56%
ep 46, loss: 56.36, 6400 train 79.97%, 1600 test 66.62%
ep 47, loss: 56.15, 6400 train 80.11%, 1600 test 71.00%
ep 48, loss: 56.97, 6400 train 79.77%, 1600 test 69.25%
ep 49, loss: 54.69, 6400 train 80.47%, 1600 test 70.56%
ep 50, loss: 55.62, 6400 train 80.61%, 1600 test 64.62%
[[141.   0.   2.  20.   0.   2.  15.   3.]
 [  1.  76.   2.  13.  13.  63.  36.   9.]
 [  9.   1.  94.  26.  32.  36.   9.   5.]
 [  5.   0.   1. 175.   2.  26.   9.   0.]
 [  1.   6.   2.  39. 133.  23.   4.   0.]
 [  1.   0.   0.   7.   0. 177.   1.   0.]
 [  8.   0.   1.  10.   1.  44. 131.   0.]
 [ 22.  19.   1.  17.   1.   8.  10. 107.]]
   Model saved to checkModel.pth
ep 51, loss: 52.92, 6400 train 80.98%, 1600 test 71.19%
ep 52, loss: 51.59, 6400 train 81.89%, 1600 test 65.50%
ep 53, loss: 51.38, 6400 train 81.95%, 1600 test 68.25%
ep 54, loss: 49.89, 6400 train 82.52%, 1600 test 72.25%
ep 55, loss: 47.72, 6400 train 82.28%, 1600 test 71.50%
ep 56, loss: 51.08, 6400 train 81.94%, 1600 test 67.00%
ep 57, loss: 48.75, 6400 train 82.81%, 1600 test 69.88%
ep 58, loss: 46.90, 6400 train 83.52%, 1600 test 70.69%
ep 59, loss: 47.30, 6400 train 83.73%, 1600 test 71.75%
ep 60, loss: 44.64, 6400 train 84.17%, 1600 test 72.31%
[[141.   2.   3.   8.   0.   2.  13.  14.]
 [  0. 154.  12.   1.   4.  10.  10.  22.]
 [  7.  17. 135.   8.  19.  14.   5.   7.]
 [  9.   3.   1. 172.   4.  13.   9.   7.]
 [  1.  45.  16.  16. 119.   3.   1.   7.]
 [  2.  10.   3.   7.   0. 163.   1.   0.]
 [  9.  24.   7.   6.   0.  15. 125.   9.]
 [  8.  17.   3.   5.   0.   1.   3. 148.]]
   Model saved to checkModel.pth
ep 61, loss: 44.90, 6400 train 84.17%, 1600 test 71.00%
ep 62, loss: 45.92, 6400 train 84.08%, 1600 test 72.38%
ep 63, loss: 43.67, 6400 train 84.44%, 1600 test 69.56%
ep 64, loss: 42.72, 6400 train 84.78%, 1600 test 71.38%
ep 65, loss: 44.48, 6400 train 84.67%, 1600 test 72.56%
ep 66, loss: 43.75, 6400 train 85.05%, 1600 test 71.00%
ep 67, loss: 40.66, 6400 train 85.91%, 1600 test 74.19%
ep 68, loss: 40.45, 6400 train 86.19%, 1600 test 69.06%
ep 69, loss: 39.77, 6400 train 85.88%, 1600 test 72.19%
ep 70, loss: 40.19, 6400 train 85.86%, 1600 test 73.19%
[[146.   0.   1.  19.   0.   1.   8.   8.]
 [  2. 105.  14.   7.  21.  14.  19.  31.]
 [ 12.   9. 127.  16.  23.  14.   5.   6.]
 [  6.   1.   5. 192.   2.   5.   4.   3.]
 [  1.   7.  16.  18. 156.   4.   3.   3.]
 [  2.   3.   3.  15.   4. 156.   2.   1.]
 [ 10.   9.   3.  18.   0.  12. 138.   5.]
 [ 13.  10.   2.   5.   2.   0.   2. 151.]]
   Model saved to checkModel.pth
ep 71, loss: 40.59, 6400 train 85.97%, 1600 test 74.00%
ep 72, loss: 38.82, 6400 train 85.97%, 1600 test 71.12%
ep 73, loss: 37.79, 6400 train 86.45%, 1600 test 73.69%
ep 74, loss: 37.35, 6400 train 86.80%, 1600 test 73.12%
ep 75, loss: 37.16, 6400 train 87.14%, 1600 test 74.00%
ep 76, loss: 36.15, 6400 train 87.00%, 1600 test 73.75%
ep 77, loss: 36.98, 6400 train 87.03%, 1600 test 70.69%
ep 78, loss: 35.64, 6400 train 87.59%, 1600 test 71.12%
ep 79, loss: 35.45, 6400 train 87.77%, 1600 test 72.62%
ep 80, loss: 33.73, 6400 train 88.70%, 1600 test 73.62%
[[134.   0.   2.  22.   1.   2.  12.  10.]
 [  0. 132.   3.   8.  17.  13.  20.  20.]
 [  8.  10. 117.  13.  31.  19.   7.   7.]
 [  4.   1.   5. 191.   2.   8.   5.   2.]
 [  1.   7.   8.  23. 158.   7.   2.   2.]
 [  2.   5.   0.   9.   3. 166.   1.   0.]
 [  6.  17.   0.  11.   4.  15. 135.   7.]
 [ 10.  15.   2.  10.   0.   1.   2. 145.]]
   Model saved to checkModel.pth
ep 81, loss: 34.49, 6400 train 87.80%, 1600 test 73.94%
ep 82, loss: 33.36, 6400 train 88.55%, 1600 test 73.44%
ep 83, loss: 34.09, 6400 train 88.50%, 1600 test 72.75%
ep 84, loss: 33.54, 6400 train 88.42%, 1600 test 73.31%
ep 85, loss: 32.49, 6400 train 89.08%, 1600 test 73.12%
ep 86, loss: 32.87, 6400 train 89.16%, 1600 test 73.25%
ep 87, loss: 32.74, 6400 train 88.69%, 1600 test 74.38%
ep 88, loss: 34.05, 6400 train 88.23%, 1600 test 67.75%
ep 89, loss: 31.90, 6400 train 88.64%, 1600 test 72.50%
ep 90, loss: 32.03, 6400 train 88.94%, 1600 test 73.31%
[[144.   1.   2.  10.   1.   1.  20.   4.]
 [  0. 114.   1.  10.  29.  21.  27.  11.]
 [  8.   9. 115.   7.  38.  20.  11.   4.]
 [  9.   1.   2. 170.   4.  17.  15.   0.]
 [  1.   5.   5.  11. 174.   8.   4.   0.]
 [  2.   2.   1.   8.   3. 167.   3.   0.]
 [  6.  12.   3.   6.   2.  12. 154.   0.]
 [ 14.  15.   2.   9.   1.   0.   9. 135.]]
   Model saved to checkModel.pth
ep 91, loss: 30.68, 6400 train 89.25%, 1600 test 73.81%
ep 92, loss: 29.54, 6400 train 89.75%, 1600 test 72.44%
ep 93, loss: 32.08, 6400 train 88.67%, 1600 test 72.81%
ep 94, loss: 29.27, 6400 train 90.00%, 1600 test 71.75%
ep 95, loss: 28.69, 6400 train 90.19%, 1600 test 74.00%
ep 96, loss: 29.55, 6400 train 89.95%, 1600 test 75.25%
ep 97, loss: 29.57, 6400 train 89.92%, 1600 test 74.12%
ep 98, loss: 28.55, 6400 train 90.12%, 1600 test 72.44%
ep 99, loss: 28.13, 6400 train 90.36%, 1600 test 72.62%
ep 100, loss: 28.06, 6400 train 90.58%, 1600 test 74.19%
[[147.   0.   2.  12.   0.   2.  13.   7.]
 [  1. 115.  12.   9.  15.  13.  24.  24.]
 [  9.   7. 134.   9.  22.  17.   7.   7.]
 [  8.   1.   7. 177.   3.  12.   7.   3.]
 [  3.   6.  14.  14. 158.   6.   4.   3.]
 [  3.   2.   3.   7.   2. 167.   2.   0.]
 [ 11.   9.   4.   4.   0.  14. 147.   6.]
 [ 16.  13.   2.   6.   1.   1.   4. 142.]]
   Model saved to checkModel.pth
   Model saved to savedModel.pth
 ```

### model 5

```shell
torch.Size([3, 80, 80])
batch size: 64
learning rate: 0.0003
train_val_split: 0.8
epochs: 200
Compose(
    RandomResizedCrop(size=(80, 80), scale=(0.75, 1.0), ratio=(0.75, 1.3), interpolation=bilinear)
    RandomHorizontalFlip(p=0.5)
    RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0)
    ColorJitter(brightness=[0.6, 1.4], contrast=[0.7, 1.3], saturation=[0.7, 1.3], hue=[-0.2, 0.2])
    RandomPosterize(bits=3,p=0.4)
    RandomEqualize(p=0.1)
    RandomGrayscale(p=0.1)
    RandomPerspective(p=0.1)
    ToTensor()
)
SimpleCNN(
  (conv_layers): Sequential(
    (0): Conv2d(3, 30, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(30, 120, kernel_size=(3, 3), stride=(1, 1))
    (5): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(120, 360, kernel_size=(3, 3), stride=(1, 1))
    (9): BatchNorm2d(360, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
    (11): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (12): Conv2d(360, 540, kernel_size=(3, 3), stride=(1, 1))
    (13): BatchNorm2d(540, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU()
    (15): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (fc_layers): Sequential(
    (0): Dropout(p=0.4, inplace=False)
    (1): Linear(in_features=4860, out_features=1000, bias=True)
    (2): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Dropout(p=0.4, inplace=False)
    (5): Linear(in_features=1000, out_features=1000, bias=True)
    (6): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): ReLU()
    (8): Dropout(p=0.4, inplace=False)
    (9): Linear(in_features=1000, out_features=8, bias=True)
  )
)
+-----------------------+------------+
|        Modules        | Parameters |
+-----------------------+------------+
|  conv_layers.0.weight |    2250    |
|   conv_layers.0.bias  |     30     |
|  conv_layers.1.weight |     30     |
|   conv_layers.1.bias  |     30     |
|  conv_layers.4.weight |   32400    |
|   conv_layers.4.bias  |    120     |
|  conv_layers.5.weight |    120     |
|   conv_layers.5.bias  |    120     |
|  conv_layers.8.weight |   388800   |
|   conv_layers.8.bias  |    360     |
|  conv_layers.9.weight |    360     |
|   conv_layers.9.bias  |    360     |
| conv_layers.12.weight |  1749600   |
|  conv_layers.12.bias  |    540     |
| conv_layers.13.weight |    540     |
|  conv_layers.13.bias  |    540     |
|   fc_layers.1.weight  |  4860000   |
|    fc_layers.1.bias   |    1000    |
|   fc_layers.2.weight  |    1000    |
|    fc_layers.2.bias   |    1000    |
|   fc_layers.5.weight  |  1000000   |
|    fc_layers.5.bias   |    1000    |
|   fc_layers.6.weight  |    1000    |
|    fc_layers.6.bias   |    1000    |
|   fc_layers.9.weight  |    8000    |
|    fc_layers.9.bias   |     8      |
+-----------------------+------------+
Total Trainable Params: 8050208
Start training...
ep 1, loss: 183.12, 6400 train 29.52%, 1600 test 44.00%
ep 2, loss: 168.70, 6400 train 36.50%, 1600 test 45.81%
ep 3, loss: 157.33, 6400 train 41.33%, 1600 test 43.44%
ep 4, loss: 150.14, 6400 train 43.59%, 1600 test 41.31%
ep 5, loss: 144.24, 6400 train 46.80%, 1600 test 51.94%
ep 6, loss: 138.78, 6400 train 48.64%, 1600 test 51.94%
ep 7, loss: 133.27, 6400 train 50.58%, 1600 test 57.50%
ep 8, loss: 129.80, 6400 train 51.75%, 1600 test 46.00%
ep 9, loss: 126.32, 6400 train 53.45%, 1600 test 54.44%
ep 10, loss: 120.48, 6400 train 55.83%, 1600 test 59.81%
[[135.   1.   3.  17.   1.   2.  51.  10.]
 [  0. 102.  15.   9.  15.  22.  31.  10.]
 [  7.  18.  88.  20.  20.  25.  19.   7.]
 [  4.   1.   2. 136.   2.  21.  17.   2.]
 [  0.  33.  40.  19.  85.   7.   9.   3.]
 [  0.   4.  10.   4.   1. 169.  14.   2.]
 [  2.   9.   5.   6.   1.  25. 139.   1.]
 [ 14.  23.   2.   2.   4.   3.  48. 103.]]
   Model saved to checkModel.pth
ep 11, loss: 118.29, 6400 train 56.39%, 1600 test 44.69%
ep 12, loss: 116.63, 6400 train 58.11%, 1600 test 57.00%
ep 13, loss: 111.65, 6400 train 59.16%, 1600 test 58.44%
ep 14, loss: 108.77, 6400 train 60.20%, 1600 test 61.50%
ep 15, loss: 108.15, 6400 train 61.12%, 1600 test 63.75%
ep 16, loss: 106.37, 6400 train 61.56%, 1600 test 65.25%
ep 17, loss: 103.60, 6400 train 63.41%, 1600 test 62.38%
ep 18, loss: 101.90, 6400 train 63.41%, 1600 test 65.88%
ep 19, loss: 100.31, 6400 train 63.62%, 1600 test 65.62%
ep 20, loss: 96.22, 6400 train 65.45%, 1600 test 65.62%
[[152.   1.   2.  15.   2.   2.   3.  43.]
 [  1.  89.   6.   4.  23.  41.   7.  33.]
 [  6.   9.  96.  15.  28.  30.   2.  18.]
 [  8.   0.   3. 133.   2.  22.   2.  15.]
 [  0.   8.  25.  14. 134.   7.   1.   7.]
 [  0.   2.   3.   4.   1. 185.   2.   7.]
 [ 16.   7.   3.   8.   1.  48.  82.  23.]
 [  5.   5.   0.   1.   6.   2.   1. 179.]]
   Model saved to checkModel.pth
ep 21, loss: 95.24, 6400 train 66.22%, 1600 test 64.88%
ep 22, loss: 93.96, 6400 train 66.91%, 1600 test 64.62%
ep 23, loss: 90.18, 6400 train 68.03%, 1600 test 65.94%
ep 24, loss: 90.94, 6400 train 67.14%, 1600 test 61.31%
ep 25, loss: 88.92, 6400 train 68.30%, 1600 test 58.81%
ep 26, loss: 88.58, 6400 train 67.95%, 1600 test 66.88%
ep 27, loss: 85.30, 6400 train 69.45%, 1600 test 65.00%
ep 28, loss: 84.47, 6400 train 69.16%, 1600 test 67.00%
ep 29, loss: 82.95, 6400 train 70.89%, 1600 test 65.94%
ep 30, loss: 81.45, 6400 train 70.44%, 1600 test 62.19%
[[148.   1.   3.  11.   0.   2.   1.  54.]
 [  0.  85.   2.   1.   7.  22.   3.  84.]
 [  7.   7. 107.  13.  10.  21.   0.  39.]
 [ 15.   0.   1. 128.   1.   9.   3.  28.]
 [  2.  22.  27.  15.  90.   6.   1.  33.]
 [  1.   2.   3.   4.   1. 172.   1.  20.]
 [ 11.   7.   3.   4.   0.  33.  76.  54.]
 [  3.   3.   2.   2.   0.   0.   0. 189.]]
   Model saved to checkModel.pth
ep 31, loss: 82.55, 6400 train 71.12%, 1600 test 68.88%
ep 32, loss: 80.56, 6400 train 71.42%, 1600 test 67.31%
ep 33, loss: 79.59, 6400 train 71.14%, 1600 test 67.88%
ep 34, loss: 76.39, 6400 train 73.14%, 1600 test 71.88%
ep 35, loss: 77.10, 6400 train 72.33%, 1600 test 71.75%
ep 36, loss: 76.10, 6400 train 72.98%, 1600 test 71.94%
ep 37, loss: 76.51, 6400 train 72.61%, 1600 test 70.31%
ep 38, loss: 74.66, 6400 train 74.00%, 1600 test 70.12%
ep 39, loss: 71.50, 6400 train 74.25%, 1600 test 71.00%
ep 40, loss: 71.36, 6400 train 74.38%, 1600 test 60.88%
[[148.   0.   3.  33.   0.  11.   6.  19.]
 [  0.  59.   0.  12.  14.  96.  10.  13.]
 [  7.   2.  71.  17.  30.  62.   1.  14.]
 [  4.   0.   1. 154.   0.  24.   0.   2.]
 [  0.   2.   8.  31. 125.  22.   1.   7.]
 [  0.   0.   0.   2.   0. 200.   1.   1.]
 [  6.   3.   0.  11.   1. 105.  57.   5.]
 [  4.   6.   0.  12.   1.  13.   3. 160.]]
   Model saved to checkModel.pth
ep 41, loss: 69.31, 6400 train 75.38%, 1600 test 70.00%
ep 42, loss: 71.14, 6400 train 74.95%, 1600 test 72.62%
ep 43, loss: 68.18, 6400 train 76.34%, 1600 test 73.75%
ep 44, loss: 70.58, 6400 train 75.06%, 1600 test 70.06%
ep 45, loss: 67.23, 6400 train 76.30%, 1600 test 73.25%
ep 46, loss: 65.29, 6400 train 76.67%, 1600 test 71.81%
ep 47, loss: 64.79, 6400 train 78.19%, 1600 test 69.69%
ep 48, loss: 65.78, 6400 train 76.22%, 1600 test 75.00%
ep 49, loss: 62.78, 6400 train 78.12%, 1600 test 71.38%
ep 50, loss: 63.07, 6400 train 78.09%, 1600 test 72.19%
[[176.   3.   3.  11.   1.   4.   5.  17.]
 [  0. 117.   8.   0.  12.  29.   8.  30.]
 [  5.  10. 127.   5.  21.  21.   2.  13.]
 [ 15.   0.   6. 130.   9.  17.   1.   7.]
 [  1.   8.  20.   6. 149.   5.   0.   7.]
 [  1.   2.   5.   1.   1. 189.   1.   4.]
 [ 10.   9.  11.   4.   1.  44.  95.  14.]
 [  8.   9.   2.   2.   2.   2.   2. 172.]]
   Model saved to checkModel.pth
ep 51, loss: 61.16, 6400 train 78.36%, 1600 test 73.19%
ep 52, loss: 62.44, 6400 train 77.77%, 1600 test 69.12%
ep 53, loss: 61.41, 6400 train 77.83%, 1600 test 68.38%
ep 54, loss: 59.87, 6400 train 78.75%, 1600 test 72.44%
ep 55, loss: 60.35, 6400 train 78.41%, 1600 test 70.44%
ep 56, loss: 58.73, 6400 train 78.86%, 1600 test 68.50%
ep 57, loss: 57.43, 6400 train 79.34%, 1600 test 69.69%
ep 58, loss: 55.80, 6400 train 80.84%, 1600 test 72.88%
ep 59, loss: 57.98, 6400 train 79.05%, 1600 test 69.50%
ep 60, loss: 57.24, 6400 train 79.48%, 1600 test 73.69%
[[196.   2.   1.   4.   1.   1.   6.   9.]
 [  1. 136.   4.   2.  13.  16.  21.  11.]
 [ 11.  16. 108.   7.  33.  15.   9.   5.]
 [ 34.   1.   2. 125.   7.  12.   4.   0.]
 [  4.   9.   7.   6. 158.   2.   5.   5.]
 [  3.   3.   1.   0.   2. 187.   6.   2.]
 [ 21.  13.   2.   3.   1.  24. 118.   6.]
 [ 19.  18.   2.   2.   1.   0.   6. 151.]]
   Model saved to checkModel.pth
ep 61, loss: 54.27, 6400 train 80.70%, 1600 test 71.75%
ep 62, loss: 53.13, 6400 train 81.06%, 1600 test 67.50%
ep 63, loss: 54.84, 6400 train 80.58%, 1600 test 73.06%
ep 64, loss: 52.21, 6400 train 81.73%, 1600 test 72.94%
ep 65, loss: 50.20, 6400 train 81.88%, 1600 test 72.69%
ep 66, loss: 54.52, 6400 train 80.33%, 1600 test 71.94%
ep 67, loss: 51.69, 6400 train 81.56%, 1600 test 71.25%
ep 68, loss: 50.02, 6400 train 82.14%, 1600 test 69.75%
ep 69, loss: 49.67, 6400 train 82.48%, 1600 test 73.06%
ep 70, loss: 48.99, 6400 train 82.77%, 1600 test 73.56%
[[194.   2.   3.   4.   0.   1.   4.  12.]
 [  2. 126.   1.   6.  14.  23.  14.  18.]
 [ 10.  16.  93.  12.  32.  23.   7.  11.]
 [ 23.   0.   1. 141.   4.  10.   4.   2.]
 [  3.   8.   4.  10. 157.   6.   3.   5.]
 [  3.   4.   0.   5.   2. 182.   6.   2.]
 [ 22.   8.   2.   6.   1.  24. 120.   5.]
 [ 13.  12.   0.   1.   2.   0.   7. 164.]]
   Model saved to checkModel.pth
ep 71, loss: 49.82, 6400 train 82.45%, 1600 test 65.81%
ep 72, loss: 48.09, 6400 train 83.42%, 1600 test 73.88%
ep 73, loss: 49.37, 6400 train 82.16%, 1600 test 72.56%
ep 74, loss: 48.21, 6400 train 82.62%, 1600 test 72.56%
ep 75, loss: 47.69, 6400 train 83.38%, 1600 test 73.81%
ep 76, loss: 47.02, 6400 train 83.42%, 1600 test 73.62%
ep 77, loss: 46.73, 6400 train 83.41%, 1600 test 72.44%
ep 78, loss: 44.95, 6400 train 84.58%, 1600 test 70.81%
ep 79, loss: 44.82, 6400 train 84.36%, 1600 test 73.69%
ep 80, loss: 45.74, 6400 train 83.61%, 1600 test 72.88%
[[174.   2.   0.  23.   2.   3.   3.  13.]
 [  1. 108.   2.   7.  17.  39.  17.  13.]
 [ 11.   8.  92.  13.  37.  31.   4.   8.]
 [  7.   0.   2. 164.   2.   9.   0.   1.]
 [  1.   1.   3.  14. 164.   6.   2.   5.]
 [  3.   1.   0.   3.   3. 189.   3.   2.]
 [  9.   9.   3.  13.   1.  35. 113.   5.]
 [ 12.  13.   0.   3.   3.   4.   2. 162.]]
   Model saved to checkModel.pth
ep 81, loss: 44.65, 6400 train 83.95%, 1600 test 73.75%
ep 82, loss: 42.55, 6400 train 84.91%, 1600 test 73.88%
ep 83, loss: 43.72, 6400 train 84.72%, 1600 test 74.81%
ep 84, loss: 42.64, 6400 train 85.53%, 1600 test 75.06%
ep 85, loss: 42.31, 6400 train 85.02%, 1600 test 72.81%
ep 86, loss: 42.37, 6400 train 85.27%, 1600 test 70.50%
ep 87, loss: 41.65, 6400 train 85.28%, 1600 test 72.94%
ep 88, loss: 39.55, 6400 train 86.44%, 1600 test 72.88%
ep 89, loss: 40.92, 6400 train 85.86%, 1600 test 75.19%
ep 90, loss: 38.18, 6400 train 86.33%, 1600 test 75.44%
[[187.   1.   1.   6.   1.   1.   5.  18.]
 [  2. 125.   1.   5.  13.  19.  17.  22.]
 [ 10.  10. 102.  10.  30.  20.   8.  14.]
 [ 17.   0.   2. 148.   0.   5.   4.   9.]
 [  1.   7.   4.  16. 159.   4.   1.   4.]
 [  4.   4.   2.   4.   1. 181.   5.   3.]
 [ 14.   7.   3.   4.   1.  16. 134.   9.]
 [  3.  14.   1.   2.   2.   0.   6. 171.]]
   Model saved to checkModel.pth
ep 91, loss: 40.17, 6400 train 86.11%, 1600 test 73.00%
ep 92, loss: 37.51, 6400 train 86.67%, 1600 test 75.94%
ep 93, loss: 38.31, 6400 train 86.66%, 1600 test 71.00%
ep 94, loss: 38.60, 6400 train 86.31%, 1600 test 71.06%
ep 95, loss: 38.58, 6400 train 86.20%, 1600 test 68.62%
ep 96, loss: 39.32, 6400 train 86.41%, 1600 test 74.00%
ep 97, loss: 37.36, 6400 train 87.09%, 1600 test 68.31%
ep 98, loss: 38.08, 6400 train 86.66%, 1600 test 74.50%
ep 99, loss: 34.85, 6400 train 87.59%, 1600 test 75.75%
ep 100, loss: 36.24, 6400 train 87.58%, 1600 test 73.69%
[[158.   2.   4.   9.   3.   1.  29.  14.]
 [  0. 149.   3.   2.   8.  16.  19.   7.]
 [  8.  18. 102.   6.  27.  22.  10.  11.]
 [ 11.   2.   2. 136.   0.  19.   9.   6.]
 [  0.  18.   4.   6. 157.   5.   4.   2.]
 [  1.   4.   0.   1.   2. 187.   7.   2.]
 [  6.  20.   4.   4.   1.  19. 133.   1.]
 [  3.  22.   1.   1.   2.   0.  13. 157.]]
   Model saved to checkModel.pth
ep 101, loss: 37.68, 6400 train 87.00%, 1600 test 75.81%
ep 102, loss: 34.68, 6400 train 87.89%, 1600 test 73.00%
ep 103, loss: 36.38, 6400 train 87.23%, 1600 test 73.44%
ep 104, loss: 33.77, 6400 train 88.34%, 1600 test 74.19%
ep 105, loss: 36.08, 6400 train 87.11%, 1600 test 72.38%
ep 106, loss: 33.24, 6400 train 88.02%, 1600 test 71.81%
ep 107, loss: 35.54, 6400 train 87.58%, 1600 test 74.75%
ep 108, loss: 34.32, 6400 train 88.27%, 1600 test 75.19%
ep 109, loss: 33.17, 6400 train 88.38%, 1600 test 73.25%
ep 110, loss: 33.10, 6400 train 88.66%, 1600 test 74.44%
[[190.   0.   1.  11.   2.   2.   3.  11.]
 [  3. 133.   3.   7.   9.  18.  11.  20.]
 [  8.  11. 103.  18.  28.  21.   3.  12.]
 [  7.   0.   3. 164.   1.   8.   0.   2.]
 [  1.  10.   6.  21. 143.   7.   1.   7.]
 [  1.   2.   0.   2.   2. 188.   4.   5.]
 [ 14.   9.   6.  19.   0.  32. 102.   6.]
 [  7.  14.   0.   5.   1.   1.   3. 168.]]
   Model saved to checkModel.pth
ep 111, loss: 33.17, 6400 train 89.09%, 1600 test 74.19%
ep 112, loss: 33.14, 6400 train 87.70%, 1600 test 71.62%
ep 113, loss: 33.52, 6400 train 88.25%, 1600 test 73.69%
ep 114, loss: 31.99, 6400 train 89.17%, 1600 test 75.94%
ep 115, loss: 30.89, 6400 train 89.27%, 1600 test 74.31%
ep 116, loss: 31.42, 6400 train 89.22%, 1600 test 73.62%
ep 117, loss: 31.29, 6400 train 88.98%, 1600 test 71.44%
ep 118, loss: 30.62, 6400 train 89.08%, 1600 test 74.44%
ep 119, loss: 30.28, 6400 train 89.17%, 1600 test 73.62%
ep 120, loss: 31.70, 6400 train 88.72%, 1600 test 71.94%
[[175.   2.   2.   7.   3.   8.  16.   7.]
 [  0. 111.   5.   1.  14.  53.  13.   7.]
 [  8.   6. 106.   5.  27.  38.   6.   8.]
 [  7.   1.   1. 135.   1.  32.   4.   4.]
 [  0.   3.   4.  13. 155.  15.   3.   3.]
 [  0.   0.   1.   0.   0. 201.   1.   1.]
 [  4.   6.   3.   3.   1.  60. 110.   1.]
 [  5.  17.   1.   3.   2.   9.   4. 158.]]
   Model saved to checkModel.pth
ep 121, loss: 32.53, 6400 train 89.12%, 1600 test 72.38%
ep 122, loss: 30.66, 6400 train 89.69%, 1600 test 72.94%
ep 123, loss: 29.06, 6400 train 90.08%, 1600 test 74.62%
ep 124, loss: 30.65, 6400 train 89.48%, 1600 test 72.88%
ep 125, loss: 30.49, 6400 train 89.47%, 1600 test 77.56%
ep 126, loss: 30.14, 6400 train 89.73%, 1600 test 75.00%
ep 127, loss: 28.85, 6400 train 90.09%, 1600 test 77.69%
ep 128, loss: 29.46, 6400 train 89.36%, 1600 test 75.94%
ep 129, loss: 29.51, 6400 train 89.88%, 1600 test 74.44%
ep 130, loss: 27.87, 6400 train 90.45%, 1600 test 76.19%
[[195.   0.   2.   4.   3.   3.   5.   8.]
 [  3. 113.   8.   3.  16.  19.  34.   8.]
 [  9.   8. 133.   7.   9.  22.  12.   4.]
 [ 14.   1.   4. 149.   1.  12.   3.   1.]
 [  0.   3.  16.  10. 148.  10.   6.   3.]
 [  3.   2.   0.   1.   2. 189.   6.   1.]
 [  9.   6.   7.   3.   0.  18. 143.   2.]
 [ 14.  15.   0.   4.   2.   4.  11. 149.]]
   Model saved to checkModel.pth
ep 131, loss: 27.94, 6400 train 89.86%, 1600 test 75.38%
ep 132, loss: 27.65, 6400 train 90.25%, 1600 test 74.69%
ep 133, loss: 26.95, 6400 train 90.38%, 1600 test 74.81%
ep 134, loss: 27.28, 6400 train 90.48%, 1600 test 76.62%
ep 135, loss: 27.03, 6400 train 90.80%, 1600 test 75.25%
ep 136, loss: 29.17, 6400 train 89.72%, 1600 test 75.56%
ep 137, loss: 28.11, 6400 train 90.23%, 1600 test 77.25%
ep 138, loss: 28.49, 6400 train 90.02%, 1600 test 73.69%
ep 139, loss: 25.92, 6400 train 90.61%, 1600 test 74.25%
ep 140, loss: 27.92, 6400 train 90.81%, 1600 test 76.19%
[[186.   1.   0.  15.   3.   2.   3.  10.]
 [  1. 131.   2.   7.  18.  14.  13.  18.]
 [  8.  13.  97.  11.  37.  20.   7.  11.]
 [  6.   0.   1. 160.   4.   8.   2.   4.]
 [  0.   1.   1.  11. 171.   6.   2.   4.]
 [  2.   2.   0.   7.   2. 180.   8.   3.]
 [ 15.  15.   4.   6.   1.  11. 129.   7.]
 [  4.  16.   1.   4.   6.   0.   3. 165.]]
   Model saved to checkModel.pth
ep 141, loss: 26.39, 6400 train 90.83%, 1600 test 75.88%
ep 142, loss: 25.18, 6400 train 91.64%, 1600 test 75.38%
ep 143, loss: 25.92, 6400 train 91.06%, 1600 test 74.44%
ep 144, loss: 27.61, 6400 train 90.86%, 1600 test 75.44%
ep 145, loss: 24.76, 6400 train 91.38%, 1600 test 75.44%
ep 146, loss: 24.13, 6400 train 91.61%, 1600 test 75.88%
ep 147, loss: 24.22, 6400 train 91.67%, 1600 test 76.38%
ep 148, loss: 22.84, 6400 train 91.84%, 1600 test 74.94%
ep 149, loss: 25.40, 6400 train 91.30%, 1600 test 74.75%
ep 150, loss: 22.65, 6400 train 92.08%, 1600 test 75.62%
[[178.   1.   0.  11.   3.   2.   0.  25.]
 [  1. 132.   1.   4.  12.  19.   8.  27.]
 [  8.  15. 101.  10.  26.  25.   4.  15.]
 [  9.   0.   2. 148.   4.   9.   1.  12.]
 [  0.   3.   3.  10. 167.   7.   0.   6.]
 [  2.   2.   0.   2.   2. 188.   4.   4.]
 [ 11.  11.   2.   7.   1.  17. 123.  16.]
 [  3.  12.   1.   2.   5.   0.   3. 173.]]
   Model saved to checkModel.pth
ep 151, loss: 23.30, 6400 train 92.00%, 1600 test 76.25%
ep 152, loss: 24.69, 6400 train 91.34%, 1600 test 75.94%
ep 153, loss: 24.17, 6400 train 91.80%, 1600 test 75.38%
ep 154, loss: 24.77, 6400 train 91.36%, 1600 test 76.19%
ep 155, loss: 23.65, 6400 train 91.84%, 1600 test 73.81%
ep 156, loss: 24.58, 6400 train 91.62%, 1600 test 74.00%
ep 157, loss: 24.64, 6400 train 91.33%, 1600 test 76.12%
ep 158, loss: 23.18, 6400 train 92.22%, 1600 test 75.25%
ep 159, loss: 24.82, 6400 train 91.36%, 1600 test 76.75%
ep 160, loss: 23.32, 6400 train 92.27%, 1600 test 76.25%
[[184.   1.   3.   5.   4.   1.   5.  17.]
 [  1. 148.   4.   1.   7.  16.   9.  18.]
 [  7.  15. 131.   3.  12.  21.   3.  12.]
 [ 14.   0.   4. 134.   6.  13.   4.  10.]
 [  0.  10.  13.   3. 159.   5.   1.   5.]
 [  3.   6.   3.   1.   2. 182.   3.   4.]
 [ 15.  20.   7.   5.   0.  22. 112.   7.]
 [  4.  14.   3.   1.   0.   0.   7. 170.]]
   Model saved to checkModel.pth
ep 161, loss: 22.36, 6400 train 92.23%, 1600 test 75.12%
ep 162, loss: 23.62, 6400 train 91.84%, 1600 test 77.00%
ep 163, loss: 22.02, 6400 train 92.48%, 1600 test 73.94%
ep 164, loss: 23.18, 6400 train 91.95%, 1600 test 75.00%
ep 165, loss: 22.17, 6400 train 92.05%, 1600 test 72.38%
ep 166, loss: 21.53, 6400 train 92.62%, 1600 test 77.25%
ep 167, loss: 21.40, 6400 train 92.33%, 1600 test 75.62%
ep 168, loss: 21.81, 6400 train 92.47%, 1600 test 75.19%
ep 169, loss: 20.98, 6400 train 92.80%, 1600 test 76.06%
ep 170, loss: 21.32, 6400 train 92.61%, 1600 test 75.44%
[[183.   0.   0.   6.   2.   3.   9.  17.]
 [  1. 120.   4.   5.   8.  34.  13.  19.]
 [  8.  11. 122.   4.  13.  30.   4.  12.]
 [  7.   1.   2. 140.   1.  21.   4.   9.]
 [  0.   7.  10.  10. 152.  12.   0.   5.]
 [  1.   1.   0.   1.   1. 193.   5.   2.]
 [ 11.  10.   5.   4.   0.  28. 125.   5.]
 [  3.  15.   2.   1.   1.   0.   5. 172.]]
   Model saved to checkModel.pth
ep 171, loss: 22.40, 6400 train 92.72%, 1600 test 76.19%
ep 172, loss: 20.67, 6400 train 92.73%, 1600 test 74.88%
ep 173, loss: 20.93, 6400 train 92.44%, 1600 test 75.94%
ep 174, loss: 22.71, 6400 train 92.30%, 1600 test 72.44%
ep 175, loss: 20.91, 6400 train 93.19%, 1600 test 74.56%
ep 176, loss: 20.05, 6400 train 93.17%, 1600 test 73.81%
ep 177, loss: 20.93, 6400 train 93.00%, 1600 test 75.94%
ep 178, loss: 21.80, 6400 train 92.81%, 1600 test 75.25%
ep 179, loss: 18.57, 6400 train 93.88%, 1600 test 74.94%
ep 180, loss: 19.46, 6400 train 93.14%, 1600 test 73.44%
[[179.   0.   6.   6.   1.   3.   8.  17.]
 [  0. 103.  10.   5.  10.  35.  18.  23.]
 [  8.   6. 123.   7.  17.  27.   5.  11.]
 [ 13.   0.   2. 146.   1.  15.   3.   5.]
 [  1.   2.  17.  16. 138.  13.   3.   6.]
 [  3.   1.   2.   2.   0. 192.   2.   2.]
 [  7.   9.   5.   4.   0.  29. 125.   9.]
 [  7.  10.   1.   2.   1.   2.   7. 169.]]
   Model saved to checkModel.pth
ep 181, loss: 21.32, 6400 train 92.73%, 1600 test 75.94%
ep 182, loss: 20.57, 6400 train 92.97%, 1600 test 74.31%
ep 183, loss: 20.03, 6400 train 93.33%, 1600 test 74.50%
ep 184, loss: 19.63, 6400 train 93.42%, 1600 test 76.00%
ep 185, loss: 19.30, 6400 train 93.39%, 1600 test 77.25%
ep 186, loss: 20.30, 6400 train 92.86%, 1600 test 76.06%
ep 187, loss: 19.67, 6400 train 93.09%, 1600 test 75.44%
ep 188, loss: 20.97, 6400 train 92.34%, 1600 test 76.38%
ep 189, loss: 20.55, 6400 train 93.02%, 1600 test 76.06%
ep 190, loss: 18.18, 6400 train 93.59%, 1600 test 77.31%
[[196.   1.   1.   7.   1.   0.   4.  10.]
 [  2. 126.   2.   6.  11.  18.  23.  16.]
 [ 12.  16. 112.   9.  22.  17.   5.  11.]
 [ 15.   1.   2. 154.   2.   5.   2.   4.]
 [  0.   5.   5.   7. 167.   5.   3.   4.]
 [  3.   4.   2.   6.   2. 178.   4.   5.]
 [ 13.   7.   5.   7.   0.  16. 137.   3.]
 [  9.  12.   1.   1.   3.   0.   6. 167.]]
   Model saved to checkModel.pth
ep 191, loss: 19.53, 6400 train 93.38%, 1600 test 77.19%
ep 192, loss: 19.77, 6400 train 93.23%, 1600 test 78.00%
ep 193, loss: 19.63, 6400 train 93.47%, 1600 test 77.00%
ep 194, loss: 19.22, 6400 train 93.39%, 1600 test 75.69%
ep 195, loss: 18.90, 6400 train 93.77%, 1600 test 76.00%
ep 196, loss: 19.14, 6400 train 93.58%, 1600 test 77.00%
ep 197, loss: 21.75, 6400 train 92.50%, 1600 test 75.44%
ep 198, loss: 18.91, 6400 train 93.33%, 1600 test 76.06%
ep 199, loss: 18.91, 6400 train 93.36%, 1600 test 75.69%
ep 200, loss: 19.55, 6400 train 93.44%, 1600 test 76.25%
[[187.   1.   1.   4.   1.   0.   4.  22.]
 [  1. 114.   5.   5.  15.  22.  12.  30.]
 [ 10.   8. 126.   8.  20.  16.   5.  11.]
 [ 16.   0.   3. 140.   4.  10.   3.   9.]
 [  1.   0.  11.   7. 162.   6.   1.   8.]
 [  2.   1.   0.   4.   2. 187.   2.   6.]
 [ 14.   5.   6.   6.   0.  17. 128.  12.]
 [  2.  12.   2.   1.   3.   0.   3. 176.]]
   Model saved to checkModel.pth
   Model saved to savedModel.pth
```

### model 7
```shell
torch.Size([3, 80, 80])
batch size: 64
learning rate: 0.0005
train_val_split: 0.8
epochs: 350
Compose(
    RandomResizedCrop(size=(80, 80), scale=(0.75, 1.0), ratio=(0.75, 1.3), interpolation=bilinear)
    RandomHorizontalFlip(p=0.5)
    RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0)
    ColorJitter(brightness=[0.6, 1.4], contrast=[0.7, 1.3], saturation=[0.7, 1.3], hue=[-0.2, 0.2])
    RandomPosterize(bits=3,p=0.4)
    RandomEqualize(p=0.1)
    RandomGrayscale(p=0.1)
    RandomPerspective(p=0.1)
    ToTensor()
)
SimpleCNN(
  (conv_layers): Sequential(
    (0): Conv2d(3, 30, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(30, 120, kernel_size=(3, 3), stride=(1, 1))
    (5): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(120, 360, kernel_size=(3, 3), stride=(1, 1))
    (9): BatchNorm2d(360, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
    (11): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (12): Conv2d(360, 540, kernel_size=(3, 3), stride=(1, 1))
    (13): BatchNorm2d(540, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU()
    (15): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (fc_layers): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=4860, out_features=1000, bias=True)
    (2): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Dropout(p=0.4, inplace=False)
    (5): Linear(in_features=1000, out_features=1000, bias=True)
    (6): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): ReLU()
    (8): Linear(in_features=1000, out_features=8, bias=True)
  )
)
+-----------------------+------------+
|        Modules        | Parameters |
+-----------------------+------------+
|  conv_layers.0.weight |    2250    |
|   conv_layers.0.bias  |     30     |
|  conv_layers.1.weight |     30     |
|   conv_layers.1.bias  |     30     |
|  conv_layers.4.weight |   32400    |
|   conv_layers.4.bias  |    120     |
|  conv_layers.5.weight |    120     |
|   conv_layers.5.bias  |    120     |
|  conv_layers.8.weight |   388800   |
|   conv_layers.8.bias  |    360     |
|  conv_layers.9.weight |    360     |
|   conv_layers.9.bias  |    360     |
| conv_layers.12.weight |  1749600   |
|  conv_layers.12.bias  |    540     |
| conv_layers.13.weight |    540     |
|  conv_layers.13.bias  |    540     |
|   fc_layers.1.weight  |  4860000   |
|    fc_layers.1.bias   |    1000    |
|   fc_layers.2.weight  |    1000    |
|    fc_layers.2.bias   |    1000    |
|   fc_layers.5.weight  |  1000000   |
|    fc_layers.5.bias   |    1000    |
|   fc_layers.6.weight  |    1000    |
|    fc_layers.6.bias   |    1000    |
|   fc_layers.8.weight  |    8000    |
|    fc_layers.8.bias   |     8      |
+-----------------------+------------+
Total Trainable Params: 8050208
Start training...
ep 1, loss: 177.50, 6400 train 31.78%, 1600 test 40.44%
ep 2, loss: 158.68, 6400 train 39.77%, 1600 test 42.75%
ep 3, loss: 149.24, 6400 train 44.27%, 1600 test 46.62%
ep 4, loss: 144.27, 6400 train 46.28%, 1600 test 45.62%
ep 5, loss: 138.61, 6400 train 48.77%, 1600 test 49.19%
ep 6, loss: 133.89, 6400 train 50.70%, 1600 test 56.00%
ep 7, loss: 129.13, 6400 train 52.41%, 1600 test 55.38%
ep 8, loss: 124.42, 6400 train 53.95%, 1600 test 48.25%
ep 9, loss: 120.95, 6400 train 56.22%, 1600 test 39.75%
ep 10, loss: 117.38, 6400 train 57.41%, 1600 test 61.12%
[[124.   1.   0.  15.   0.   2.  23.  29.]
 [  0.  95.   5.   7.  11.  37.  21.  25.]
 [  7.  11.  92.  21.  20.  43.  12.   6.]
 [ 16.   5.   3. 121.   1.  38.   6.   5.]
 [  3.  20.  33.  17. 101.  14.   0.  11.]
 [  1.   3.   2.  10.   0. 160.  11.   3.]
 [ 12.  14.   1.   9.   0.  27. 131.  17.]
 [  8.  26.   1.   1.   2.   1.   5. 154.]]
   Model saved to checkModel.pth
ep 11, loss: 114.58, 6400 train 58.17%, 1600 test 56.62%
ep 12, loss: 112.37, 6400 train 59.33%, 1600 test 60.25%
ep 13, loss: 110.37, 6400 train 59.92%, 1600 test 51.06%
ep 14, loss: 108.33, 6400 train 61.56%, 1600 test 61.88%
ep 15, loss: 104.42, 6400 train 61.95%, 1600 test 57.88%
ep 16, loss: 102.89, 6400 train 62.58%, 1600 test 50.38%
ep 17, loss: 99.47, 6400 train 63.91%, 1600 test 53.00%
ep 18, loss: 98.25, 6400 train 64.72%, 1600 test 54.31%
ep 19, loss: 96.41, 6400 train 64.81%, 1600 test 62.06%
ep 20, loss: 96.56, 6400 train 65.14%, 1600 test 68.19%
[[150.   1.   1.  14.   0.   0.  15.  13.]
 [  1.  89.  11.   6.  20.  28.  23.  23.]
 [ 11.   7. 129.  16.  17.  19.   9.   4.]
 [ 14.   4.  10. 140.   6.  14.   6.   1.]
 [  2.   9.  40.  10. 125.   4.   2.   7.]
 [  2.   1.   7.  11.   1. 155.  11.   2.]
 [ 11.  11.   5.  10.   0.  17. 146.  11.]
 [ 16.  14.   2.   2.   2.   1.   4. 157.]]
   Model saved to checkModel.pth
ep 21, loss: 93.48, 6400 train 66.50%, 1600 test 61.50%
ep 22, loss: 90.45, 6400 train 67.59%, 1600 test 59.06%
ep 23, loss: 89.29, 6400 train 67.47%, 1600 test 63.88%
ep 24, loss: 88.15, 6400 train 68.31%, 1600 test 60.75%
ep 25, loss: 87.02, 6400 train 68.30%, 1600 test 65.19%
ep 26, loss: 87.21, 6400 train 68.34%, 1600 test 51.75%
ep 27, loss: 84.64, 6400 train 69.64%, 1600 test 62.88%
ep 28, loss: 83.02, 6400 train 70.61%, 1600 test 67.94%
ep 29, loss: 81.93, 6400 train 70.16%, 1600 test 66.81%
ep 30, loss: 79.74, 6400 train 71.25%, 1600 test 63.06%
[[ 97.   5.   0.  30.   1.  11.   7.  43.]
 [  0. 123.   2.   2.   5.  38.   7.  24.]
 [  4.  14. 120.   8.  15.  37.   4.  10.]
 [  3.   2.   6. 126.   6.  47.   0.   5.]
 [  0.  36.  28.   9. 110.  10.   0.   6.]
 [  1.   3.   1.   4.   0. 177.   1.   3.]
 [  4.  17.   7.  12.   1.  71.  83.  16.]
 [  2.  12.   2.   5.   3.   1.   0. 173.]]
   Model saved to checkModel.pth
ep 31, loss: 78.48, 6400 train 71.66%, 1600 test 68.19%
ep 32, loss: 77.16, 6400 train 72.16%, 1600 test 66.50%
ep 33, loss: 77.54, 6400 train 72.34%, 1600 test 56.31%
ep 34, loss: 76.35, 6400 train 72.48%, 1600 test 62.69%
ep 35, loss: 73.79, 6400 train 73.41%, 1600 test 69.44%
ep 36, loss: 73.47, 6400 train 74.09%, 1600 test 65.56%
ep 37, loss: 72.61, 6400 train 73.97%, 1600 test 65.56%
ep 38, loss: 72.28, 6400 train 73.77%, 1600 test 68.69%
ep 39, loss: 70.68, 6400 train 74.16%, 1600 test 68.12%
ep 40, loss: 68.76, 6400 train 75.80%, 1600 test 69.12%
[[117.   0.   0.  29.   1.   5.  18.  24.]
 [  0.  80.   7.  10.  29.  32.  24.  19.]
 [  5.   2. 111.  27.  37.  17.   6.   7.]
 [  4.   0.   2. 172.   2.  12.   0.   3.]
 [  0.   2.   9.  18. 159.   5.   2.   4.]
 [  1.   0.   2.  15.   0. 164.   5.   3.]
 [  4.  11.   3.  17.   1.  23. 139.  13.]
 [  3.  12.   3.   7.   6.   0.   3. 164.]]
   Model saved to checkModel.pth
ep 41, loss: 68.18, 6400 train 75.61%, 1600 test 60.62%
ep 42, loss: 68.02, 6400 train 75.86%, 1600 test 62.25%
ep 43, loss: 66.17, 6400 train 76.30%, 1600 test 65.00%
ep 44, loss: 65.56, 6400 train 76.62%, 1600 test 65.12%
ep 45, loss: 64.85, 6400 train 77.22%, 1600 test 68.88%
ep 46, loss: 62.38, 6400 train 76.98%, 1600 test 66.56%
ep 47, loss: 61.26, 6400 train 77.61%, 1600 test 69.44%
ep 48, loss: 63.45, 6400 train 77.62%, 1600 test 71.62%
ep 49, loss: 61.72, 6400 train 77.81%, 1600 test 69.12%
ep 50, loss: 61.13, 6400 train 78.05%, 1600 test 70.06%
[[134.   3.   1.  31.   4.   0.   5.  16.]
 [  0. 126.   4.   8.  17.  11.  11.  24.]
 [  8.   8.  96.  28.  46.  16.   4.   6.]
 [  7.   1.   1. 177.   3.   4.   0.   2.]
 [  0.  14.   5.  10. 160.   6.   1.   3.]
 [  1.   8.   1.  19.   0. 155.   4.   2.]
 [ 17.  20.   3.  40.   4.  17.  97.  13.]
 [  5.   8.   1.   6.   1.   0.   1. 176.]]
   Model saved to checkModel.pth
ep 51, loss: 59.91, 6400 train 78.41%, 1600 test 65.69%
ep 52, loss: 58.55, 6400 train 79.00%, 1600 test 67.94%
ep 53, loss: 57.53, 6400 train 79.62%, 1600 test 70.31%
ep 54, loss: 57.20, 6400 train 79.38%, 1600 test 64.81%
ep 55, loss: 55.77, 6400 train 80.33%, 1600 test 71.62%
ep 56, loss: 56.91, 6400 train 79.28%, 1600 test 62.56%
ep 57, loss: 55.04, 6400 train 80.44%, 1600 test 65.25%
ep 58, loss: 52.89, 6400 train 81.20%, 1600 test 69.44%
ep 59, loss: 51.73, 6400 train 81.48%, 1600 test 62.62%
ep 60, loss: 52.40, 6400 train 80.86%, 1600 test 69.12%
[[127.   0.   0.  41.   3.   4.   8.  11.]
 [  0.  97.   3.   4.  26.  39.  15.  17.]
 [  4.   3. 121.  26.  26.  26.   1.   5.]
 [  4.   0.   2. 175.   3.  10.   0.   1.]
 [  0.   7.  15.  11. 158.   6.   1.   1.]
 [  0.   1.   1.  13.   2. 170.   2.   1.]
 [ 10.  13.   6.  35.   5.  38.  97.   7.]
 [  5.  17.   3.   8.   3.   1.   0. 161.]]
   Model saved to checkModel.pth
ep 61, loss: 52.15, 6400 train 81.02%, 1600 test 71.25%
ep 62, loss: 51.51, 6400 train 81.78%, 1600 test 71.69%
ep 63, loss: 50.52, 6400 train 81.89%, 1600 test 70.50%
ep 64, loss: 49.18, 6400 train 82.42%, 1600 test 71.31%
ep 65, loss: 48.59, 6400 train 82.89%, 1600 test 72.50%
ep 66, loss: 47.93, 6400 train 82.80%, 1600 test 72.25%
ep 67, loss: 49.84, 6400 train 81.86%, 1600 test 71.44%
ep 68, loss: 47.02, 6400 train 82.89%, 1600 test 68.75%
ep 69, loss: 46.46, 6400 train 83.88%, 1600 test 68.62%
ep 70, loss: 45.97, 6400 train 83.33%, 1600 test 71.94%
[[148.   5.   0.  22.   2.   2.   8.   7.]
 [  0. 144.   2.   2.  14.  17.  12.  10.]
 [  8.  25. 104.  14.  32.  21.   7.   1.]
 [  6.   2.   3. 166.   6.  10.   2.   0.]
 [  0.  22.   6.   7. 154.   8.   1.   1.]
 [  3.   8.   1.   6.   0. 167.   5.   0.]
 [ 19.  23.   5.  14.   1.  19. 129.   1.]
 [  7.  41.   1.   8.   0.   1.   1. 139.]]
   Model saved to checkModel.pth
ep 71, loss: 45.62, 6400 train 83.97%, 1600 test 72.81%
ep 72, loss: 47.28, 6400 train 83.34%, 1600 test 72.00%
ep 73, loss: 43.94, 6400 train 84.20%, 1600 test 66.12%
ep 74, loss: 43.05, 6400 train 84.91%, 1600 test 74.06%
ep 75, loss: 41.83, 6400 train 85.11%, 1600 test 74.00%
ep 76, loss: 42.80, 6400 train 84.78%, 1600 test 73.75%
ep 77, loss: 43.30, 6400 train 85.00%, 1600 test 69.00%
ep 78, loss: 41.52, 6400 train 85.28%, 1600 test 67.12%
ep 79, loss: 43.48, 6400 train 84.44%, 1600 test 67.31%
ep 80, loss: 41.19, 6400 train 85.31%, 1600 test 68.50%
[[143.   0.   0.  42.   1.   1.   3.   4.]
 [  2. 109.   1.  21.  26.  24.   7.  11.]
 [  6.   4. 126.  32.  22.  17.   1.   4.]
 [  5.   1.   1. 184.   2.   2.   0.   0.]
 [  0.   9.   8.  18. 158.   5.   0.   1.]
 [  1.   1.   2.  28.   1. 154.   2.   1.]
 [ 24.  12.   3.  69.   2.  19.  76.   6.]
 [  7.  14.   3.  26.   0.   2.   0. 146.]]
   Model saved to checkModel.pth
ep 81, loss: 40.17, 6400 train 85.89%, 1600 test 72.38%
ep 82, loss: 39.27, 6400 train 85.91%, 1600 test 72.25%
ep 83, loss: 40.24, 6400 train 85.55%, 1600 test 74.81%
ep 84, loss: 39.71, 6400 train 86.22%, 1600 test 74.38%
ep 85, loss: 39.81, 6400 train 86.25%, 1600 test 75.62%
ep 86, loss: 37.84, 6400 train 86.00%, 1600 test 71.81%
ep 87, loss: 38.91, 6400 train 86.22%, 1600 test 74.81%
ep 88, loss: 37.31, 6400 train 86.84%, 1600 test 73.06%
ep 89, loss: 35.53, 6400 train 87.84%, 1600 test 71.50%
ep 90, loss: 35.91, 6400 train 87.66%, 1600 test 70.81%
[[127.   4.   2.  40.   3.   4.   2.  12.]
 [  0. 116.   4.   8.  28.  20.   2.  23.]
 [  4.   4. 129.  24.  28.  16.   3.   4.]
 [  6.   1.   3. 172.   7.   5.   0.   1.]
 [  1.   5.   6.   7. 176.   2.   0.   2.]
 [  1.   1.   4.  10.   2. 166.   2.   4.]
 [ 10.  18.   6.  52.   6.  30.  74.  15.]
 [  3.  11.   1.   7.   2.   1.   0. 173.]]
   Model saved to checkModel.pth
ep 91, loss: 37.18, 6400 train 86.94%, 1600 test 72.88%
ep 92, loss: 34.35, 6400 train 87.83%, 1600 test 71.00%
ep 93, loss: 35.88, 6400 train 87.73%, 1600 test 72.75%
ep 94, loss: 36.62, 6400 train 86.91%, 1600 test 73.12%
ep 95, loss: 36.64, 6400 train 86.97%, 1600 test 73.44%
```

### model 8

```shell
torch.Size([3, 80, 80])
batch size: 64
learning rate: 0.0004
train_val_split: 0.8
epochs: 350
Compose(
    RandomResizedCrop(size=(80, 80), scale=(0.75, 1.0), ratio=(0.75, 1.3), interpolation=bilinear)
    RandomHorizontalFlip(p=0.5)
    RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0)
    ColorJitter(brightness=[0.6, 1.4], contrast=[0.7, 1.3], saturation=[0.7, 1.3], hue=[-0.2, 0.2])
    RandomPosterize(bits=3,p=0.4)
    RandomEqualize(p=0.1)
    RandomGrayscale(p=0.1)
    RandomPerspective(p=0.1)
    ToTensor()
)
SimpleCNN(
  (conv_layers): Sequential(
    (0): Conv2d(3, 30, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(30, 120, kernel_size=(3, 3), stride=(1, 1))
    (5): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(120, 360, kernel_size=(3, 3), stride=(1, 1))
    (9): BatchNorm2d(360, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
    (11): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (12): Conv2d(360, 540, kernel_size=(3, 3), stride=(1, 1))
    (13): BatchNorm2d(540, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU()
    (15): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (fc_layers): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=4860, out_features=1000, bias=True)
    (2): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Dropout(p=0.4, inplace=False)
    (5): Linear(in_features=1000, out_features=1000, bias=True)
    (6): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): ReLU()
    (8): Dropout(p=0.3, inplace=False)
    (9): Linear(in_features=1000, out_features=8, bias=True)
  )
)
+-----------------------+------------+
|        Modules        | Parameters |
+-----------------------+------------+
|  conv_layers.0.weight |    2250    |
|   conv_layers.0.bias  |     30     |
|  conv_layers.1.weight |     30     |
|   conv_layers.1.bias  |     30     |
|  conv_layers.4.weight |   32400    |
|   conv_layers.4.bias  |    120     |
|  conv_layers.5.weight |    120     |
|   conv_layers.5.bias  |    120     |
|  conv_layers.8.weight |   388800   |
|   conv_layers.8.bias  |    360     |
|  conv_layers.9.weight |    360     |
|   conv_layers.9.bias  |    360     |
| conv_layers.12.weight |  1749600   |
|  conv_layers.12.bias  |    540     |
| conv_layers.13.weight |    540     |
|  conv_layers.13.bias  |    540     |
|   fc_layers.1.weight  |  4860000   |
|    fc_layers.1.bias   |    1000    |
|   fc_layers.2.weight  |    1000    |
|    fc_layers.2.bias   |    1000    |
|   fc_layers.5.weight  |  1000000   |
|    fc_layers.5.bias   |    1000    |
|   fc_layers.6.weight  |    1000    |
|    fc_layers.6.bias   |    1000    |
|   fc_layers.9.weight  |    8000    |
|    fc_layers.9.bias   |     8      |
+-----------------------+------------+
Total Trainable Params: 8050208
Start training...
ep 1, loss: 183.87, 6400 train 29.92%, 1600 test 39.00%
ep 2, loss: 164.85, 6400 train 38.42%, 1600 test 48.75%
ep 3, loss: 156.15, 6400 train 41.03%, 1600 test 47.25%
ep 4, loss: 148.46, 6400 train 44.06%, 1600 test 52.19%
ep 5, loss: 141.96, 6400 train 47.08%, 1600 test 54.69%
ep 6, loss: 136.21, 6400 train 49.97%, 1600 test 55.75%
ep 7, loss: 131.30, 6400 train 52.09%, 1600 test 52.75%
ep 8, loss: 127.41, 6400 train 53.41%, 1600 test 56.00%
ep 9, loss: 122.96, 6400 train 55.17%, 1600 test 63.38%
ep 10, loss: 121.12, 6400 train 55.80%, 1600 test 54.56%
[[178.   0.   0.   7.   0.   3.  20.   9.]
 [  2.  23.   0.  19.  22.  74.  25.  37.]
 [ 13.   2.  40.  32.  13.  56.  14.  10.]
 [ 26.   0.   0. 123.   1.  37.  15.   5.]
 [  3.   2.   9.  69.  83.  14.   7.   9.]
 [  2.   0.   0.   5.   0. 190.   2.   2.]
 [ 22.   1.   0.   6.   1.  54.  97.   5.]
 [ 30.   2.   1.   8.   5.  12.  14. 139.]]
   Model saved to checkModel.pth
ep 11, loss: 115.81, 6400 train 57.95%, 1600 test 51.94%
ep 12, loss: 114.48, 6400 train 58.95%, 1600 test 59.06%
ep 13, loss: 112.10, 6400 train 59.62%, 1600 test 59.31%
ep 14, loss: 107.41, 6400 train 61.30%, 1600 test 61.25%
ep 15, loss: 106.36, 6400 train 61.72%, 1600 test 67.00%
ep 16, loss: 104.41, 6400 train 62.58%, 1600 test 58.94%
ep 17, loss: 101.69, 6400 train 63.47%, 1600 test 43.56%
ep 18, loss: 100.00, 6400 train 63.80%, 1600 test 60.06%
ep 19, loss: 98.95, 6400 train 64.31%, 1600 test 57.00%
ep 20, loss: 97.22, 6400 train 64.88%, 1600 test 64.75%
[[138.   1.   2.  36.   3.  10.  10.  17.]
 [  0.  92.   5.   9.  14.  58.  10.  14.]
 [  5.   9.  87.  15.  13.  30.  11.  10.]
 [  2.   3.   9. 165.   2.  19.   1.   6.]
 [  1.   8.  21.  17. 128.   9.   2.  10.]
 [  1.   0.   4.   7.   2. 184.   0.   3.]
 [  6.  13.   3.   8.   2.  54.  97.   3.]
 [  6.  21.   1.  10.   6.  14.   8. 145.]]
   Model saved to checkModel.pth
ep 21, loss: 94.55, 6400 train 66.28%, 1600 test 65.81%
ep 22, loss: 94.85, 6400 train 66.03%, 1600 test 65.81%
ep 23, loss: 92.20, 6400 train 67.38%, 1600 test 65.94%
ep 24, loss: 92.27, 6400 train 66.69%, 1600 test 68.88%
ep 25, loss: 90.14, 6400 train 67.84%, 1600 test 68.81%
ep 26, loss: 87.73, 6400 train 67.89%, 1600 test 55.06%
ep 27, loss: 86.32, 6400 train 68.84%, 1600 test 66.19%
ep 28, loss: 85.21, 6400 train 68.98%, 1600 test 66.56%
ep 29, loss: 83.49, 6400 train 70.09%, 1600 test 68.31%
ep 30, loss: 84.07, 6400 train 70.28%, 1600 test 59.50%
[[125.   0.   2.  30.   1.  19.  33.   7.]
 [  0.  60.   2.   8.  17.  67.  39.   9.]
 [  5.   1.  62.  11.  19.  56.  17.   9.]
 [  2.   0.   2. 123.   1.  66.  10.   3.]
 [  1.   4.   3.  34. 116.  18.  15.   5.]
 [  1.   0.   1.   2.   0. 194.   1.   2.]
 [  0.   3.   0.   4.   0.  49. 129.   1.]
 [  6.  14.   0.   8.   4.  18.  18. 143.]]
   Model saved to checkModel.pth
ep 31, loss: 80.49, 6400 train 71.22%, 1600 test 70.50%
ep 32, loss: 80.49, 6400 train 71.42%, 1600 test 66.94%
ep 33, loss: 78.82, 6400 train 71.86%, 1600 test 68.00%
ep 34, loss: 78.30, 6400 train 72.22%, 1600 test 62.06%
ep 35, loss: 76.27, 6400 train 72.58%, 1600 test 56.75%
ep 36, loss: 74.33, 6400 train 73.19%, 1600 test 68.44%
ep 37, loss: 76.70, 6400 train 72.28%, 1600 test 59.25%
ep 38, loss: 74.66, 6400 train 73.22%, 1600 test 69.12%
ep 39, loss: 71.46, 6400 train 74.97%, 1600 test 71.19%
ep 40, loss: 72.70, 6400 train 74.08%, 1600 test 65.19%
[[112.   2.   3.  11.   3.  12.  53.  21.]
 [  0. 113.   8.   0.   8.  33.  26.  14.]
 [  2.   5. 119.   2.   9.  24.  11.   8.]
 [  4.   3.   8. 100.   2.  61.  24.   5.]
 [  0.  15.  26.   4. 125.  11.   6.   9.]
 [  0.   2.   4.   0.   1. 188.   4.   2.]
 [  0.  12.   5.   1.   0.  44. 123.   1.]
 [  3.  19.   1.   1.   3.   8.  13. 163.]]
   Model saved to checkModel.pth
ep 41, loss: 74.20, 6400 train 73.41%, 1600 test 69.94%
ep 42, loss: 69.65, 6400 train 75.03%, 1600 test 72.75%
ep 43, loss: 69.17, 6400 train 75.19%, 1600 test 72.00%
ep 44, loss: 70.89, 6400 train 74.16%, 1600 test 67.94%
ep 45, loss: 68.44, 6400 train 75.36%, 1600 test 67.94%
ep 46, loss: 67.34, 6400 train 75.73%, 1600 test 69.88%
ep 47, loss: 66.79, 6400 train 76.56%, 1600 test 69.19%
ep 48, loss: 66.45, 6400 train 76.73%, 1600 test 60.94%
ep 49, loss: 65.26, 6400 train 76.78%, 1600 test 73.94%
ep 50, loss: 63.53, 6400 train 77.09%, 1600 test 70.88%
[[153.   0.   5.  18.   3.   9.   9.  20.]
 [  3. 115.  12.   0.   9.  35.  13.  15.]
 [  5.   6. 127.   7.   9.  16.   2.   8.]
 [  8.   1.  12. 139.   8.  30.   4.   5.]
 [  0.   8.  23.   3. 145.   8.   1.   8.]
 [  1.   1.   7.   3.   0. 186.   0.   3.]
 [  8.  19.  10.   6.   2.  44.  90.   7.]
 [  6.  16.   2.   3.   3.   2.   0. 179.]]
   Model saved to checkModel.pth
ep 51, loss: 62.54, 6400 train 77.75%, 1600 test 68.12%
ep 52, loss: 61.73, 6400 train 78.08%, 1600 test 72.00%
ep 53, loss: 63.70, 6400 train 76.73%, 1600 test 70.06%
ep 54, loss: 62.24, 6400 train 78.31%, 1600 test 71.81%
ep 55, loss: 60.57, 6400 train 78.41%, 1600 test 72.00%
ep 56, loss: 60.06, 6400 train 78.27%, 1600 test 69.31%
ep 57, loss: 59.30, 6400 train 78.45%, 1600 test 65.00%
ep 58, loss: 57.51, 6400 train 79.31%, 1600 test 65.62%
ep 59, loss: 57.12, 6400 train 79.22%, 1600 test 68.75%
ep 60, loss: 58.12, 6400 train 79.20%, 1600 test 69.12%
[[182.   0.   1.   7.   2.   1.  24.   0.]
 [  3.  64.   0.  16.  24.  34.  44.  17.]
 [  8.   3.  79.  26.  25.  13.  17.   9.]
 [ 19.   0.   2. 173.   2.   5.   5.   1.]
 [  2.   2.   2.  36. 139.   6.   4.   5.]
 [  2.   0.   1.  16.   0. 176.   4.   2.]
 [ 11.   6.   1.   8.   2.  24. 132.   2.]
 [ 22.   5.   0.   8.   4.   3.   8. 161.]]
   Model saved to checkModel.pth
ep 61, loss: 57.80, 6400 train 79.72%, 1600 test 72.31%
ep 62, loss: 55.49, 6400 train 80.61%, 1600 test 71.00%
ep 63, loss: 54.04, 6400 train 81.16%, 1600 test 72.81%
ep 64, loss: 53.11, 6400 train 81.38%, 1600 test 74.44%
ep 65, loss: 53.96, 6400 train 80.89%, 1600 test 74.88%
ep 66, loss: 52.56, 6400 train 81.33%, 1600 test 74.81%
ep 67, loss: 52.86, 6400 train 81.89%, 1600 test 72.00%
ep 68, loss: 51.80, 6400 train 81.95%, 1600 test 71.56%
ep 69, loss: 49.45, 6400 train 81.83%, 1600 test 72.12%
ep 70, loss: 51.60, 6400 train 81.86%, 1600 test 73.44%
[[179.   0.   3.  24.   2.   1.   3.   5.]
 [  2.  86.  11.  13.  14.  35.   7.  34.]
 [  5.   4. 121.  15.   9.  16.   3.   7.]
 [  7.   0.   5. 182.   2.   9.   0.   2.]
 [  0.   1.  15.  15. 149.   8.   1.   7.]
 [  1.   0.   3.  11.   0. 182.   1.   3.]
 [ 14.  12.   7.  16.   1.  28. 101.   7.]
 [ 14.   7.   3.   7.   4.   1.   0. 175.]]
   Model saved to checkModel.pth
ep 71, loss: 50.77, 6400 train 81.64%, 1600 test 65.75%
ep 72, loss: 50.07, 6400 train 82.55%, 1600 test 73.50%
ep 73, loss: 49.94, 6400 train 82.17%, 1600 test 73.00%
ep 74, loss: 48.77, 6400 train 82.67%, 1600 test 76.62%
ep 75, loss: 48.44, 6400 train 82.72%, 1600 test 71.19%
ep 76, loss: 47.35, 6400 train 83.16%, 1600 test 72.50%
ep 77, loss: 47.98, 6400 train 83.23%, 1600 test 72.62%
ep 78, loss: 47.53, 6400 train 82.72%, 1600 test 74.94%
ep 79, loss: 47.35, 6400 train 82.84%, 1600 test 75.44%
ep 80, loss: 46.11, 6400 train 83.78%, 1600 test 72.12%
[[182.   0.   1.   3.   1.   3.  22.   5.]
 [  3. 131.   2.   0.  10.  19.  28.   9.]
 [  7.  10.  85.   7.  21.  27.  18.   5.]
 [ 22.   1.   1. 117.   3.  39.  19.   5.]
 [  1.  14.   0.   3. 153.  11.   9.   5.]
 [  1.   4.   1.   0.   1. 187.   5.   2.]
 [  7.  15.   2.   0.   2.  20. 138.   2.]
 [ 14.  24.   1.   2.   1.   2.   6. 161.]]
   Model saved to checkModel.pth
ep 81, loss: 45.17, 6400 train 83.64%, 1600 test 73.62%
ep 82, loss: 45.34, 6400 train 83.83%, 1600 test 74.56%
ep 83, loss: 42.23, 6400 train 85.41%, 1600 test 67.81%
ep 84, loss: 44.45, 6400 train 84.00%, 1600 test 72.56%
ep 85, loss: 44.13, 6400 train 84.16%, 1600 test 73.44%
ep 86, loss: 42.40, 6400 train 84.98%, 1600 test 72.62%
ep 87, loss: 43.99, 6400 train 84.19%, 1600 test 71.00%
ep 88, loss: 41.87, 6400 train 85.30%, 1600 test 72.94%
ep 89, loss: 42.42, 6400 train 85.25%, 1600 test 70.69%
ep 90, loss: 40.75, 6400 train 85.69%, 1600 test 75.06%
[[187.   0.   1.  14.   3.   4.   6.   2.]
 [  2. 109.   2.   3.  13.  29.  22.  22.]
 [  6.   9. 102.  10.  21.  22.   5.   5.]
 [ 12.   0.   2. 171.   3.  17.   0.   2.]
 [  1.   7.   6.   9. 156.  11.   1.   5.]
 [  1.   1.   0.   4.   2. 189.   2.   2.]
 [ 19.  12.   6.   5.   1.  27. 113.   3.]
 [ 11.  13.   1.   4.   3.   3.   2. 174.]]
   Model saved to checkModel.pth
ep 91, loss: 40.84, 6400 train 85.48%, 1600 test 71.50%
ep 92, loss: 41.86, 6400 train 85.14%, 1600 test 75.75%
ep 93, loss: 40.02, 6400 train 85.67%, 1600 test 75.12%
ep 94, loss: 39.44, 6400 train 86.53%, 1600 test 74.12%
ep 95, loss: 40.13, 6400 train 86.14%, 1600 test 71.88%
ep 96, loss: 39.98, 6400 train 85.67%, 1600 test 71.75%
ep 97, loss: 38.95, 6400 train 86.56%, 1600 test 75.81%
ep 98, loss: 38.61, 6400 train 86.59%, 1600 test 76.44%
ep 99, loss: 38.06, 6400 train 86.64%, 1600 test 73.75%
ep 100, loss: 37.44, 6400 train 86.89%, 1600 test 76.12%
[[184.   0.   2.  16.   3.   4.   8.   0.]
 [  2. 123.   2.   4.  12.  25.  21.  13.]
 [  9.  12.  88.  12.  24.  20.   9.   6.]
 [ 10.   1.   1. 183.   2.   7.   2.   1.]
 [  1.   5.   2.  10. 166.   7.   2.   3.]
 [  2.   1.   2.   3.   2. 187.   2.   2.]
 [  9.  12.   1.  12.   3.  19. 129.   1.]
 [ 16.  19.   1.   8.   4.   2.   3. 158.]]
   Model saved to checkModel.pth
ep 101, loss: 38.47, 6400 train 86.64%, 1600 test 77.06%
ep 102, loss: 36.67, 6400 train 86.94%, 1600 test 75.50%
ep 103, loss: 35.71, 6400 train 86.84%, 1600 test 76.00%
ep 104, loss: 37.06, 6400 train 87.64%, 1600 test 77.12%
ep 105, loss: 36.20, 6400 train 87.61%, 1600 test 76.44%
ep 106, loss: 35.54, 6400 train 87.91%, 1600 test 72.00%
ep 107, loss: 34.98, 6400 train 87.44%, 1600 test 71.19%
ep 108, loss: 33.46, 6400 train 88.61%, 1600 test 77.50%
ep 109, loss: 35.22, 6400 train 87.61%, 1600 test 73.62%
ep 110, loss: 34.77, 6400 train 88.20%, 1600 test 71.38%
[[177.   1.   4.  10.   2.  12.  10.   1.]
 [  3.  96.   2.   6.  20.  44.  23.   8.]
 [  6.   7.  94.  10.  18.  32.   8.   5.]
 [  8.   0.   1. 166.   2.  27.   2.   1.]
 [  1.   3.   5.  19. 153.  11.   1.   3.]
 [  1.   0.   0.   3.   1. 193.   1.   2.]
 [  7.   9.   1.   6.   2.  46. 114.   1.]
 [ 11.  18.   1.   8.   6.  10.   8. 149.]]
   Model saved to checkModel.pth
ep 111, loss: 33.44, 6400 train 88.19%, 1600 test 73.81%
ep 112, loss: 33.62, 6400 train 87.91%, 1600 test 74.06%
ep 113, loss: 33.80, 6400 train 88.38%, 1600 test 76.44%
ep 114, loss: 33.36, 6400 train 88.28%, 1600 test 74.12%
ep 115, loss: 32.19, 6400 train 88.89%, 1600 test 77.31%
ep 116, loss: 33.54, 6400 train 88.62%, 1600 test 75.44%
ep 117, loss: 33.65, 6400 train 88.33%, 1600 test 72.75%
ep 118, loss: 32.62, 6400 train 88.41%, 1600 test 76.94%
ep 119, loss: 31.41, 6400 train 89.23%, 1600 test 73.75%
ep 120, loss: 33.56, 6400 train 87.95%, 1600 test 73.56%
[[156.   0.   5.  34.   3.   8.   4.   7.]
 [  2. 132.   5.   3.  13.  28.   5.  14.]
 [  6.  11. 104.   6.  19.  28.   0.   6.]
 [  7.   2.   1. 171.   6.  19.   0.   1.]
 [  0.   7.   9.   2. 168.   6.   0.   4.]
 [  1.   1.   3.   2.   2. 190.   0.   2.]
 [  8.  18.   5.   9.   4.  50.  83.   9.]
 [  6.  18.   2.   6.   4.   2.   0. 173.]]
   Model saved to checkModel.pth
ep 121, loss: 31.06, 6400 train 89.48%, 1600 test 77.19%
ep 122, loss: 30.75, 6400 train 89.14%, 1600 test 77.44%
ep 123, loss: 30.88, 6400 train 89.12%, 1600 test 76.94%
ep 124, loss: 29.98, 6400 train 89.20%, 1600 test 77.06%
ep 125, loss: 29.22, 6400 train 89.86%, 1600 test 73.81%
ep 126, loss: 32.21, 6400 train 88.92%, 1600 test 76.31%
ep 127, loss: 32.18, 6400 train 88.59%, 1600 test 76.00%
ep 128, loss: 29.44, 6400 train 89.77%, 1600 test 76.31%
ep 129, loss: 29.57, 6400 train 90.00%, 1600 test 75.25%
ep 130, loss: 30.69, 6400 train 89.48%, 1600 test 74.38%
[[173.   1.   2.  31.   0.   4.   2.   4.]
 [  2. 117.   2.  12.  12.  20.  14.  23.]
 [  6.   8. 110.  22.   9.  18.   1.   6.]
 [  8.   0.   0. 190.   1.   6.   0.   2.]
 [  0.  10.  10.  13. 151.   5.   1.   6.]
 [  2.   1.   1.  12.   1. 181.   0.   3.]
 [ 13.  11.   5.  26.   2.  25.  97.   7.]
 [ 11.  11.   2.  12.   2.   0.   2. 171.]]
   Model saved to checkModel.pth
ep 131, loss: 30.01, 6400 train 89.42%, 1600 test 75.06%
ep 132, loss: 29.47, 6400 train 89.81%, 1600 test 76.69%
ep 133, loss: 28.33, 6400 train 90.38%, 1600 test 73.81%
ep 134, loss: 29.17, 6400 train 89.53%, 1600 test 76.88%
ep 135, loss: 28.60, 6400 train 89.92%, 1600 test 75.38%
ep 136, loss: 26.68, 6400 train 90.67%, 1600 test 76.56%
ep 137, loss: 26.58, 6400 train 90.70%, 1600 test 77.75%
ep 138, loss: 27.64, 6400 train 90.02%, 1600 test 76.75%
ep 139, loss: 27.10, 6400 train 91.02%, 1600 test 76.69%
ep 140, loss: 28.94, 6400 train 90.02%, 1600 test 73.81%
[[183.   1.   7.   4.   1.   6.   9.   6.]
 [  1. 100.   5.   4.  10.  44.  24.  14.]
 [  4.   4. 126.   4.   5.  29.   2.   6.]
 [ 20.   1.   2. 155.   1.  27.   0.   1.]
 [  1.   8.  11.  12. 142.  16.   1.   5.]
 [  1.   0.   3.   0.   0. 193.   2.   2.]
 [ 11.  13.   4.   5.   0.  38. 113.   2.]
 [  8.  17.   3.   1.   3.   6.   4. 169.]]
   Model saved to checkModel.pth
ep 141, loss: 26.77, 6400 train 90.91%, 1600 test 75.69%
ep 142, loss: 24.90, 6400 train 91.47%, 1600 test 76.31%
ep 143, loss: 27.34, 6400 train 90.28%, 1600 test 77.81%
ep 144, loss: 28.01, 6400 train 90.55%, 1600 test 74.44%
ep 145, loss: 27.52, 6400 train 90.83%, 1600 test 75.06%
ep 146, loss: 25.33, 6400 train 91.30%, 1600 test 75.56%
ep 147, loss: 26.60, 6400 train 90.98%, 1600 test 76.81%
ep 148, loss: 26.22, 6400 train 90.98%, 1600 test 73.06%
ep 149, loss: 26.16, 6400 train 90.83%, 1600 test 76.88%
ep 150, loss: 26.95, 6400 train 90.70%, 1600 test 73.88%
[[184.   1.   1.  16.   5.   2.   6.   2.]
 [  2.  78.   8.  12.  22.  45.  26.   9.]
 [  5.   2. 123.   7.  12.  21.   4.   6.]
 [ 10.   0.   2. 182.   3.   9.   0.   1.]
 [  1.   3.   9.   6. 162.  11.   1.   3.]
 [  2.   1.   4.   3.   0. 189.   0.   2.]
 [ 18.   6.   7.  11.   0.  36. 107.   1.]
 [ 16.   8.   4.  13.   5.   5.   3. 157.]]
   Model saved to checkModel.pth
ep 151, loss: 27.01, 6400 train 90.88%, 1600 test 76.75%
ep 152, loss: 24.39, 6400 train 91.48%, 1600 test 74.00%
ep 153, loss: 25.08, 6400 train 91.66%, 1600 test 76.88%
ep 154, loss: 25.06, 6400 train 91.06%, 1600 test 74.44%
ep 155, loss: 23.59, 6400 train 91.88%, 1600 test 73.75%
ep 156, loss: 25.82, 6400 train 91.02%, 1600 test 77.56%
ep 157, loss: 24.59, 6400 train 91.66%, 1600 test 76.62%
ep 158, loss: 22.00, 6400 train 92.56%, 1600 test 77.38%
ep 159, loss: 22.36, 6400 train 92.12%, 1600 test 78.06%
ep 160, loss: 23.94, 6400 train 91.94%, 1600 test 77.38%
[[183.   0.   0.  23.   4.   2.   3.   2.]
 [  3. 107.   6.   4.  19.  29.  18.  16.]
 [  7.   6. 119.  12.  11.  15.   5.   5.]
 [  9.   1.   1. 189.   4.   2.   0.   1.]
 [  0.   1.  10.   5. 172.   3.   1.   4.]
 [  2.   1.   0.   8.   2. 183.   3.   2.]
 [ 18.  10.   3.  13.   2.  23. 113.   4.]
 [ 14.  11.   3.   6.   5.   0.   0. 172.]]
   Model saved to checkModel.pth
ep 161, loss: 23.90, 6400 train 91.56%, 1600 test 72.50%
ep 162, loss: 23.47, 6400 train 91.83%, 1600 test 75.12%
ep 163, loss: 25.33, 6400 train 91.09%, 1600 test 74.38%
ep 164, loss: 25.11, 6400 train 91.41%, 1600 test 78.06%
ep 165, loss: 23.51, 6400 train 91.98%, 1600 test 76.31%
ep 166, loss: 23.87, 6400 train 91.80%, 1600 test 74.31%
ep 167, loss: 23.16, 6400 train 91.98%, 1600 test 75.81%
ep 168, loss: 23.80, 6400 train 91.94%, 1600 test 74.44%
ep 169, loss: 23.51, 6400 train 91.97%, 1600 test 77.62%
ep 170, loss: 21.95, 6400 train 92.50%, 1600 test 77.75%
[[185.   0.   2.   7.   5.   3.   3.  12.]
 [  1. 138.   1.   1.   9.  14.  14.  24.]
 [  5.  16. 114.   6.  12.  15.   3.   9.]
 [ 14.   1.   1. 170.   7.   7.   2.   5.]
 [  1.  11.   5.   6. 160.   6.   1.   6.]
 [  2.   7.   3.   4.   0. 181.   2.   2.]
 [ 11.  16.   2.  10.   2.  18. 119.   8.]
 [  7.  15.   1.   3.   4.   1.   3. 177.]]
   Model saved to checkModel.pth
ep 171, loss: 22.60, 6400 train 92.47%, 1600 test 77.56%
ep 172, loss: 21.39, 6400 train 92.56%, 1600 test 72.12%
ep 173, loss: 22.15, 6400 train 92.31%, 1600 test 74.88%
ep 174, loss: 22.81, 6400 train 92.06%, 1600 test 76.81%
ep 175, loss: 21.40, 6400 train 92.47%, 1600 test 78.44%
ep 176, loss: 21.27, 6400 train 93.06%, 1600 test 76.44%
ep 177, loss: 21.57, 6400 train 92.36%, 1600 test 77.69%
ep 178, loss: 22.66, 6400 train 92.27%, 1600 test 77.38%
ep 179, loss: 20.10, 6400 train 93.23%, 1600 test 75.75%
ep 180, loss: 22.29, 6400 train 92.44%, 1600 test 76.31%
[[168.   0.   3.  15.   6.   8.  10.   7.]
 [  1. 141.   3.   1.  12.  25.  11.   8.]
 [  6.  12. 114.   4.  16.  22.   1.   5.]
 [  6.   2.   0. 167.   8.  17.   3.   4.]
 [  0.  10.   3.   6. 167.   6.   1.   3.]
 [  2.   6.   1.   3.   2. 186.   0.   1.]
 [  3.  14.   4.   9.   4.  31. 120.   1.]
 [  5.  30.   1.   3.   6.   5.   3. 158.]]
   Model saved to checkModel.pth
ep 181, loss: 20.75, 6400 train 92.67%, 1600 test 76.94%
ep 182, loss: 21.41, 6400 train 92.66%, 1600 test 76.00%
ep 183, loss: 21.49, 6400 train 92.36%, 1600 test 73.38%
ep 184, loss: 20.99, 6400 train 92.80%, 1600 test 76.19%
ep 185, loss: 18.82, 6400 train 93.52%, 1600 test 77.94%
ep 186, loss: 19.87, 6400 train 93.12%, 1600 test 74.31%
ep 187, loss: 20.45, 6400 train 92.97%, 1600 test 77.56%
ep 188, loss: 21.80, 6400 train 92.77%, 1600 test 76.56%
ep 189, loss: 20.76, 6400 train 92.81%, 1600 test 74.25%
ep 190, loss: 18.87, 6400 train 93.28%, 1600 test 75.94%
[[162.   1.   2.  23.   2.   2.   8.  17.]
 [  1. 120.   0.   4.  13.  30.  13.  21.]
 [  4.  17.  99.  14.  13.  23.   2.   8.]
 [  5.   2.   0. 185.   3.   5.   1.   6.]
 [  0.   8.   0.  12. 165.   5.   1.   5.]
 [  1.   1.   0.   7.   2. 186.   2.   2.]
 [  8.  13.   0.   6.   3.  33. 116.   7.]
 [  2.  14.   1.   6.   3.   2.   1. 182.]]
   Model saved to checkModel.pth
ep 191, loss: 20.68, 6400 train 92.97%, 1600 test 75.94%
ep 192, loss: 19.65, 6400 train 93.14%, 1600 test 77.19%
ep 193, loss: 20.14, 6400 train 92.94%, 1600 test 77.19%
ep 194, loss: 19.72, 6400 train 93.16%, 1600 test 78.81%
ep 195, loss: 19.60, 6400 train 93.36%, 1600 test 74.81%
ep 196, loss: 19.60, 6400 train 93.19%, 1600 test 76.06%
ep 197, loss: 20.75, 6400 train 92.95%, 1600 test 75.75%
ep 198, loss: 17.84, 6400 train 93.98%, 1600 test 75.19%
ep 199, loss: 18.99, 6400 train 93.31%, 1600 test 78.19%
ep 200, loss: 19.88, 6400 train 93.50%, 1600 test 77.88%
[[183.   0.   3.   8.   2.   1.  12.   8.]
 [  1. 119.   2.   2.   9.  24.  24.  21.]
 [  6.  10. 119.   8.   6.  19.   6.   6.]
 [  9.   2.   0. 169.   2.  15.   8.   2.]
 [  0.   7.   9.   9. 154.  11.   2.   4.]
 [  2.   1.   3.   3.   0. 190.   0.   2.]
 [  5.  12.   2.   4.   0.  27. 135.   1.]
 [  5.  17.   2.   1.   2.   3.   4. 177.]]
   Model saved to checkModel.pth
ep 201, loss: 17.28, 6400 train 94.14%, 1600 test 79.25%
ep 202, loss: 19.12, 6400 train 93.53%, 1600 test 78.31%
ep 203, loss: 21.00, 6400 train 92.97%, 1600 test 76.12%
ep 204, loss: 18.10, 6400 train 93.83%, 1600 test 76.62%
ep 205, loss: 19.32, 6400 train 93.38%, 1600 test 78.44%
ep 206, loss: 18.46, 6400 train 93.80%, 1600 test 74.94%
ep 207, loss: 18.57, 6400 train 93.45%, 1600 test 76.75%
ep 208, loss: 18.85, 6400 train 93.69%, 1600 test 77.31%
ep 209, loss: 17.52, 6400 train 94.09%, 1600 test 77.62%
ep 210, loss: 18.43, 6400 train 93.75%, 1600 test 77.50%
[[179.   0.   2.  14.   3.   5.  10.   4.]
 [  1. 139.   6.   1.   8.  24.  13.  10.]
 [  4.  13. 118.   7.  10.  18.   3.   7.]
 [  5.   1.   2. 175.   4.  18.   1.   1.]
 [  0.  15.   6.   5. 155.  10.   1.   4.]
 [  1.   4.   3.   1.   1. 190.   1.   0.]
 [  9.  16.   7.   4.   1.  25. 120.   4.]
 [  6.  27.   3.   3.   1.   4.   3. 164.]]
   Model saved to checkModel.pth
ep 211, loss: 19.16, 6400 train 93.20%, 1600 test 75.56%
ep 212, loss: 19.58, 6400 train 93.05%, 1600 test 75.75%
ep 213, loss: 17.36, 6400 train 94.05%, 1600 test 76.88%
ep 214, loss: 18.02, 6400 train 93.64%, 1600 test 77.06%
ep 215, loss: 19.05, 6400 train 93.42%, 1600 test 77.50%
ep 216, loss: 17.59, 6400 train 94.19%, 1600 test 77.81%
ep 217, loss: 17.92, 6400 train 93.84%, 1600 test 76.19%
ep 218, loss: 18.17, 6400 train 93.73%, 1600 test 77.81%
ep 219, loss: 18.64, 6400 train 93.61%, 1600 test 76.94%
ep 220, loss: 18.73, 6400 train 93.45%, 1600 test 73.81%
[[195.   0.   0.  16.   1.   1.   2.   2.]
 [  4. 105.   2.  10.  12.  34.  25.  10.]
 [  7.  10. 111.  17.   9.  16.   5.   5.]
 [ 11.   0.   1. 187.   0.   4.   2.   2.]
 [  2.   8.   4.  23. 145.  10.   1.   3.]
 [  3.   3.   2.   5.   0. 187.   0.   1.]
 [ 23.  11.   1.  16.   0.  32. 102.   1.]
 [ 29.  18.   3.   5.   2.   5.   0. 149.]]
   Model saved to checkModel.pth
ep 221, loss: 17.47, 6400 train 93.75%, 1600 test 77.00%
ep 222, loss: 17.83, 6400 train 94.05%, 1600 test 78.00%
ep 223, loss: 17.55, 6400 train 94.00%, 1600 test 74.56%
ep 224, loss: 16.99, 6400 train 94.36%, 1600 test 77.75%
ep 225, loss: 16.76, 6400 train 94.33%, 1600 test 75.12%
ep 226, loss: 16.22, 6400 train 94.53%, 1600 test 77.88%
ep 227, loss: 16.83, 6400 train 94.53%, 1600 test 77.19%
ep 228, loss: 17.46, 6400 train 94.00%, 1600 test 77.44%
ep 229, loss: 16.48, 6400 train 94.23%, 1600 test 77.12%
ep 230, loss: 16.92, 6400 train 94.30%, 1600 test 75.44%
[[170.   0.   1.  20.   5.   1.   5.  15.]
 [  1. 112.   2.   3.  21.  23.  12.  28.]
 [  6.   9. 104.   6.  23.  21.   4.   7.]
 [  7.   0.   2. 175.   9.   8.   2.   4.]
 [  0.   3.   2.   7. 176.   4.   1.   3.]
 [  1.   2.   2.   5.   3. 183.   1.   4.]
 [ 12.  13.   0.  11.   4.  27. 107.  12.]
 [  6.   8.   1.   4.   8.   3.   1. 180.]]
   Model saved to checkModel.pth
ep 231, loss: 17.32, 6400 train 94.20%, 1600 test 76.12%
ep 232, loss: 17.34, 6400 train 94.28%, 1600 test 78.81%
ep 233, loss: 16.12, 6400 train 94.48%, 1600 test 77.81%
ep 234, loss: 17.00, 6400 train 94.09%, 1600 test 76.19%
ep 235, loss: 17.13, 6400 train 94.12%, 1600 test 76.38%
ep 236, loss: 16.48, 6400 train 94.28%, 1600 test 76.06%
ep 237, loss: 15.94, 6400 train 94.48%, 1600 test 77.38%
ep 238, loss: 15.92, 6400 train 94.78%, 1600 test 75.50%
ep 239, loss: 14.46, 6400 train 94.75%, 1600 test 77.88%
ep 240, loss: 16.64, 6400 train 94.50%, 1600 test 75.12%
[[176.   0.   0.  18.   1.   1.   9.  12.]
 [  1. 112.   1.   6.  10.  20.  28.  24.]
 [  7.  17.  92.  19.   9.  17.  10.   9.]
 [  6.   1.   1. 185.   1.   5.   5.   3.]
 [  1.  16.   4.  21. 135.   8.   3.   8.]
 [  1.   2.   0.  10.   0. 179.   6.   3.]
 [  9.  12.   0.   6.   0.  14. 142.   3.]
 [  9.  13.   1.   4.   0.   0.   3. 181.]]
   Model saved to checkModel.pth
ep 241, loss: 15.13, 6400 train 95.00%, 1600 test 77.94%
ep 242, loss: 15.80, 6400 train 94.58%, 1600 test 78.56%
ep 243, loss: 16.48, 6400 train 94.38%, 1600 test 78.25%
ep 244, loss: 15.99, 6400 train 94.62%, 1600 test 77.19%
ep 245, loss: 16.92, 6400 train 94.47%, 1600 test 77.81%
ep 246, loss: 16.59, 6400 train 94.48%, 1600 test 77.50%
ep 247, loss: 16.75, 6400 train 94.58%, 1600 test 78.88%
ep 248, loss: 15.10, 6400 train 94.91%, 1600 test 75.50%
ep 249, loss: 15.86, 6400 train 94.50%, 1600 test 75.88%
ep 250, loss: 14.80, 6400 train 94.84%, 1600 test 77.19%
[[186.   0.   3.  12.   2.   3.   5.   6.]
 [  1. 113.   4.   7.  18.  20.  18.  21.]
 [  6.   8. 114.  15.  13.  15.   3.   6.]
 [ 11.   1.   1. 185.   2.   5.   1.   1.]
 [  0.   3.   9.  11. 159.   7.   2.   5.]
 [  2.   3.   2.   7.   2. 180.   1.   4.]
 [ 15.  11.   4.  11.   0.  17. 123.   5.]
 [ 10.   9.   5.   4.   4.   2.   2. 175.]]
   Model saved to checkModel.pth
ep 251, loss: 15.64, 6400 train 94.36%, 1600 test 77.81%
ep 252, loss: 14.48, 6400 train 95.11%, 1600 test 77.69%
ep 253, loss: 15.31, 6400 train 94.80%, 1600 test 78.88%
ep 254, loss: 14.55, 6400 train 95.19%, 1600 test 78.38%
ep 255, loss: 14.16, 6400 train 95.31%, 1600 test 78.12%
ep 256, loss: 15.24, 6400 train 94.66%, 1600 test 76.75%
ep 257, loss: 13.77, 6400 train 95.44%, 1600 test 79.06%
ep 258, loss: 15.79, 6400 train 94.48%, 1600 test 77.38%
ep 259, loss: 14.34, 6400 train 95.27%, 1600 test 76.62%
ep 260, loss: 15.29, 6400 train 94.88%, 1600 test 75.88%
[[178.   0.   3.  20.   2.   1.   8.   5.]
 [  2.  97.   7.  10.  19.  19.  35.  13.]
 [  6.   4. 122.  12.  12.  10.   7.   7.]
 [  6.   0.   3. 188.   1.   4.   3.   2.]
 [  1.   1.  13.  18. 151.   7.   2.   3.]
 [  2.   3.   3.  11.   0. 175.   4.   3.]
 [  8.   9.   5.  14.   0.  12. 137.   1.]
 [ 10.  12.   5.   7.   4.   3.   4. 166.]]
   Model saved to checkModel.pth
ep 261, loss: 13.83, 6400 train 95.61%, 1600 test 75.50%
ep 262, loss: 15.56, 6400 train 94.78%, 1600 test 78.00%
ep 263, loss: 15.37, 6400 train 95.28%, 1600 test 78.31%
ep 264, loss: 14.42, 6400 train 95.08%, 1600 test 76.62%
ep 265, loss: 15.13, 6400 train 94.94%, 1600 test 78.19%
ep 266, loss: 13.87, 6400 train 95.25%, 1600 test 77.38%
ep 267, loss: 14.71, 6400 train 95.14%, 1600 test 78.00%
ep 268, loss: 13.58, 6400 train 95.45%, 1600 test 77.44%
ep 269, loss: 13.94, 6400 train 95.27%, 1600 test 76.06%
ep 270, loss: 14.07, 6400 train 95.28%, 1600 test 76.62%
[[173.   0.   1.  11.   3.   3.  23.   3.]
 [  1. 115.   1.   7.  14.  21.  31.  12.]
 [  6.   7. 101.  12.  17.  19.  10.   8.]
 [  7.   0.   0. 181.   2.   9.   7.   1.]
 [  0.   3.   5.  15. 162.   4.   4.   3.]
 [  1.   2.   0.   7.   1. 185.   2.   3.]
 [  3.  11.   2.   5.   2.  18. 144.   1.]
 [  6.  16.   1.   5.   4.   2.  12. 165.]]
   Model saved to checkModel.pth
ep 271, loss: 14.01, 6400 train 95.27%, 1600 test 78.38%
ep 272, loss: 13.91, 6400 train 95.33%, 1600 test 77.81%
ep 273, loss: 14.73, 6400 train 95.09%, 1600 test 77.75%
ep 274, loss: 12.75, 6400 train 95.59%, 1600 test 78.00%
ep 275, loss: 14.57, 6400 train 95.00%, 1600 test 77.38%
ep 276, loss: 14.81, 6400 train 94.92%, 1600 test 78.50%
ep 277, loss: 15.06, 6400 train 94.95%, 1600 test 78.25%
ep 278, loss: 13.99, 6400 train 95.30%, 1600 test 75.69%
ep 279, loss: 13.84, 6400 train 95.62%, 1600 test 77.31%
ep 280, loss: 13.53, 6400 train 95.34%, 1600 test 79.25%
[[178.   0.   2.  10.   2.   3.   7.  15.]
 [  1. 134.   3.   2.  10.  12.  18.  22.]
 [  5.  16. 118.   7.   8.  13.   5.   8.]
 [  6.   2.   1. 175.   4.   9.   3.   7.]
 [  1.  10.   7.   4. 163.   5.   3.   3.]
 [  2.   3.   3.   4.   1. 183.   2.   3.]
 [  9.  16.   5.   4.   1.  13. 134.   4.]
 [  4.  14.   3.   2.   2.   2.   1. 183.]]
   Model saved to checkModel.pth
ep 281, loss: 13.13, 6400 train 95.55%, 1600 test 77.50%
ep 282, loss: 14.22, 6400 train 95.27%, 1600 test 74.69%
ep 283, loss: 13.70, 6400 train 95.20%, 1600 test 78.12%
ep 284, loss: 14.40, 6400 train 95.16%, 1600 test 75.88%
ep 285, loss: 13.91, 6400 train 95.34%, 1600 test 77.56%
ep 286, loss: 14.03, 6400 train 95.11%, 1600 test 78.56%
ep 287, loss: 13.13, 6400 train 95.78%, 1600 test 77.12%
ep 288, loss: 13.77, 6400 train 95.33%, 1600 test 76.31%
ep 289, loss: 13.39, 6400 train 95.47%, 1600 test 77.12%
ep 290, loss: 13.71, 6400 train 95.25%, 1600 test 75.12%
[[176.   1.   3.  14.   4.   6.  10.   3.]
 [  2. 103.   7.   7.  10.  40.  24.   9.]
 [  6.   7. 130.   5.   4.  16.   6.   6.]
 [  9.   0.   4. 166.   0.  24.   3.   1.]
 [  0.   6.  11.  12. 153.  10.   1.   3.]
 [  1.   1.   3.   1.   0. 191.   2.   2.]
 [  5.  10.   6.   5.   1.  35. 123.   1.]
 [ 14.  11.   5.   5.   3.   8.   5. 160.]]
   Model saved to checkModel.pth
ep 291, loss: 12.89, 6400 train 95.67%, 1600 test 78.25%
ep 292, loss: 14.05, 6400 train 95.11%, 1600 test 77.88%
ep 293, loss: 12.45, 6400 train 95.80%, 1600 test 77.38%
ep 294, loss: 13.52, 6400 train 95.38%, 1600 test 78.75%
ep 295, loss: 13.93, 6400 train 95.25%, 1600 test 76.69%
ep 296, loss: 12.65, 6400 train 95.97%, 1600 test 78.62%
ep 297, loss: 12.52, 6400 train 95.95%, 1600 test 77.94%
ep 298, loss: 12.34, 6400 train 95.75%, 1600 test 77.12%
ep 299, loss: 13.37, 6400 train 95.56%, 1600 test 77.19%
ep 300, loss: 12.10, 6400 train 95.81%, 1600 test 76.06%
[[186.   0.   2.  18.   2.   1.   2.   6.]
 [  1. 104.   7.  13.  16.  23.  14.  24.]
 [  7.   8. 127.  10.   6.  14.   2.   6.]
 [  5.   0.   3. 184.   4.   7.   0.   4.]
 [  0.   1.   6.  13. 167.   5.   1.   3.]
 [  2.   2.   4.   9.   0. 181.   1.   2.]
 [ 20.   9.   4.  22.   4.  22. 102.   3.]
 [ 11.  14.   5.   6.   5.   4.   0. 166.]]
   Model saved to checkModel.pth
ep 301, loss: 13.12, 6400 train 95.67%, 1600 test 76.88%
ep 302, loss: 12.64, 6400 train 95.73%, 1600 test 78.81%
ep 303, loss: 14.20, 6400 train 95.36%, 1600 test 76.94%
ep 304, loss: 13.20, 6400 train 95.62%, 1600 test 76.88%
ep 305, loss: 11.79, 6400 train 95.97%, 1600 test 75.88%
ep 306, loss: 12.05, 6400 train 95.86%, 1600 test 77.44%
ep 307, loss: 12.76, 6400 train 95.66%, 1600 test 77.56%
ep 308, loss: 13.79, 6400 train 95.28%, 1600 test 76.94%
ep 309, loss: 12.41, 6400 train 95.88%, 1600 test 77.06%
ep 310, loss: 13.34, 6400 train 95.50%, 1600 test 76.19%
[[178.   3.   2.  19.   2.   1.   3.   9.]
 [  1. 142.   0.   2.  17.  16.  11.  13.]
 [  5.  16.  95.  13.  26.  15.   4.   6.]
 [  6.   1.   0. 182.   5.   9.   0.   4.]
 [  0.  12.   4.   6. 164.   5.   1.   4.]
 [  2.   8.   1.   6.   2. 180.   0.   2.]
 [ 13.  21.   3.  17.   1.  21. 103.   7.]
 [  5.  22.   1.   5.   2.   0.   1. 175.]]
   Model saved to checkModel.pth
ep 311, loss: 12.84, 6400 train 95.23%, 1600 test 76.88%
ep 312, loss: 11.76, 6400 train 95.95%, 1600 test 77.69%
ep 313, loss: 12.03, 6400 train 96.08%, 1600 test 78.56%
ep 314, loss: 12.79, 6400 train 95.80%, 1600 test 77.62%
ep 315, loss: 13.40, 6400 train 95.36%, 1600 test 78.31%
ep 316, loss: 13.58, 6400 train 95.39%, 1600 test 78.69%
ep 317, loss: 13.69, 6400 train 95.42%, 1600 test 78.38%
ep 318, loss: 12.18, 6400 train 96.05%, 1600 test 77.44%
ep 319, loss: 11.86, 6400 train 95.75%, 1600 test 77.81%
ep 320, loss: 11.89, 6400 train 96.06%, 1600 test 78.50%
[[186.   0.   6.  11.   3.   2.   4.   5.]
 [  1. 114.   7.   3.  21.  29.  12.  15.]
 [  4.   7. 133.   6.  12.  11.   1.   6.]
 [  7.   0.   4. 179.   4.   8.   4.   1.]
 [  0.   1.   9.   3. 174.   3.   1.   5.]
 [  2.   2.   4.   2.   2. 188.   0.   1.]
 [ 15.  12.   7.  10.   5.  21. 113.   3.]
 [  8.   8.   6.   6.   7.   4.   3. 169.]]
   Model saved to checkModel.pth
```

### model 9

``` shell
torch.Size([3, 80, 80])
batch size: 64
learning rate: 0.0004
train_val_split: 0.8
epochs: 1000
Compose(
    RandomResizedCrop(size=(80, 80), scale=(0.75, 1.0), ratio=(0.75, 1.3), interpolation=bilinear)
    RandomHorizontalFlip(p=0.5)
    RandomRotation(degrees=[-10.0, 10.0], interpolation=nearest, expand=False, fill=0)
    ColorJitter(brightness=[0.6, 1.4], contrast=[0.7, 1.3], saturation=[0.7, 1.3], hue=[-0.2, 0.2])
    RandomPosterize(bits=3,p=0.4)
    RandomEqualize(p=0.1)
    RandomGrayscale(p=0.1)
    RandomPerspective(p=0.1)
    ToTensor()
)
SimpleCNN(
  (conv_layers): Sequential(
    (0): Conv2d(3, 30, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): BatchNorm2d(30, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(30, 120, kernel_size=(3, 3), stride=(1, 1))
    (5): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (8): Conv2d(120, 360, kernel_size=(3, 3), stride=(1, 1))
    (9): BatchNorm2d(360, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): ReLU()
    (11): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (12): Conv2d(360, 540, kernel_size=(3, 3), stride=(1, 1))
    (13): BatchNorm2d(540, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (14): ReLU()
    (15): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (fc_layers): Sequential(
    (0): Dropout(p=0.7, inplace=False)
    (1): Linear(in_features=4860, out_features=1600, bias=True)
    (2): BatchNorm1d(1600, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Dropout(p=0.5, inplace=False)
    (5): Linear(in_features=1600, out_features=1000, bias=True)
    (6): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): ReLU()
    (8): Dropout(p=0.3, inplace=False)
    (9): Linear(in_features=1000, out_features=8, bias=True)
  )
)
+-----------------------+------------+
|        Modules        | Parameters |
+-----------------------+------------+
|  conv_layers.0.weight |    2250    |
|   conv_layers.0.bias  |     30     |
|  conv_layers.1.weight |     30     |
|   conv_layers.1.bias  |     30     |
|  conv_layers.4.weight |   32400    |
|   conv_layers.4.bias  |    120     |
|  conv_layers.5.weight |    120     |
|   conv_layers.5.bias  |    120     |
|  conv_layers.8.weight |   388800   |
|   conv_layers.8.bias  |    360     |
|  conv_layers.9.weight |    360     |
|   conv_layers.9.bias  |    360     |
| conv_layers.12.weight |  1749600   |
|  conv_layers.12.bias  |    540     |
| conv_layers.13.weight |    540     |
|  conv_layers.13.bias  |    540     |
|   fc_layers.1.weight  |  7776000   |
|    fc_layers.1.bias   |    1600    |
|   fc_layers.2.weight  |    1600    |
|    fc_layers.2.bias   |    1600    |
|   fc_layers.5.weight  |  1600000   |
|    fc_layers.5.bias   |    1000    |
|   fc_layers.6.weight  |    1000    |
|    fc_layers.6.bias   |    1000    |
|   fc_layers.9.weight  |    8000    |
|    fc_layers.9.bias   |     8      |
+-----------------------+------------+
Total Trainable Params: 11568008
Start training...
ep 1, loss: 189.35, 6400 train 27.16%, 1600 test 37.19%
ep 2, loss: 170.17, 6400 train 35.28%, 1600 test 45.06%
ep 3, loss: 158.05, 6400 train 40.03%, 1600 test 47.25%
ep 4, loss: 153.35, 6400 train 41.81%, 1600 test 30.06%
ep 5, loss: 145.52, 6400 train 46.22%, 1600 test 34.00%
ep 6, loss: 142.38, 6400 train 47.31%, 1600 test 43.19%
ep 7, loss: 138.55, 6400 train 48.98%, 1600 test 40.12%
ep 8, loss: 134.82, 6400 train 50.59%, 1600 test 43.31%
ep 9, loss: 131.74, 6400 train 51.88%, 1600 test 48.50%
ep 10, loss: 127.92, 6400 train 53.36%, 1600 test 53.25%
[[172.   0.   1.  11.   1.   2.   6.   6.]
 [  2.  54.   4.  22.   7.  55.  26.  39.]
 [ 16.   3.  56.  52.  18.  46.  11.   7.]
 [ 33.   2.   1. 140.   0.  14.   9.   2.]
 [  5.  12.  20.  66.  65.  14.   6.  12.]
 [  1.   0.   0.  25.   0. 147.   4.   2.]
 [ 27.   1.   0.  24.   0.  43. 109.   7.]
 [ 54.   8.   1.   9.   3.   2.   6. 109.]]
   Model saved to checkModel.pth
ep 11, loss: 127.13, 6400 train 52.67%, 1600 test 46.00%
ep 12, loss: 124.36, 6400 train 54.56%, 1600 test 42.06%
ep 13, loss: 123.35, 6400 train 54.95%, 1600 test 37.81%
ep 14, loss: 121.70, 6400 train 55.30%, 1600 test 53.94%
ep 15, loss: 117.92, 6400 train 57.58%, 1600 test 57.06%
ep 16, loss: 117.12, 6400 train 58.00%, 1600 test 50.31%
ep 17, loss: 114.62, 6400 train 58.17%, 1600 test 39.88%
ep 18, loss: 112.76, 6400 train 59.39%, 1600 test 56.00%
ep 19, loss: 110.30, 6400 train 60.50%, 1600 test 54.06%
ep 20, loss: 108.12, 6400 train 60.53%, 1600 test 57.00%
[[113.   0.   0.  59.   2.   2.   0.  23.]
 [  1.  87.  10.  22.  22.  31.   0.  36.]
 [ 10.   7.  95.  45.  30.  14.   1.   7.]
 [  4.   0.   2. 179.   6.   5.   0.   5.]
 [  0.   5.  20.  31. 130.   8.   0.   6.]
 [  0.   1.   2.  31.   3. 140.   0.   2.]
 [ 16.   4.   4.  82.   1.  66.  20.  18.]
 [  6.  12.   0.  18.   3.   5.   0. 148.]]
   Model saved to checkModel.pth
ep 21, loss: 107.56, 6400 train 61.55%, 1600 test 49.56%
ep 22, loss: 106.01, 6400 train 61.98%, 1600 test 55.62%
ep 23, loss: 105.54, 6400 train 61.88%, 1600 test 55.56%
ep 24, loss: 102.91, 6400 train 62.80%, 1600 test 54.81%
ep 25, loss: 102.41, 6400 train 63.31%, 1600 test 51.50%
ep 26, loss: 99.28, 6400 train 64.95%, 1600 test 58.81%
ep 27, loss: 101.70, 6400 train 63.34%, 1600 test 57.50%
ep 28, loss: 96.97, 6400 train 65.00%, 1600 test 62.19%
ep 29, loss: 95.50, 6400 train 65.22%, 1600 test 57.69%
ep 30, loss: 97.54, 6400 train 65.05%, 1600 test 60.38%
[[150.   0.   0.  35.   4.   2.   0.   8.]
 [  1.  60.   3.  19.  35.  65.   1.  25.]
 [ 11.   3.  90.  33.  34.  29.   2.   7.]
 [  5.   1.   3. 173.   5.  14.   0.   0.]
 [  0.   1.  14.  24. 146.  11.   1.   3.]
 [  1.   0.   0.  16.   4. 157.   0.   1.]
 [ 30.   5.   2.  36.   6.  77.  48.   7.]
 [ 15.   9.   0.  12.   9.   5.   0. 142.]]
   Model saved to checkModel.pth
ep 31, loss: 94.75, 6400 train 66.22%, 1600 test 54.94%
ep 32, loss: 93.40, 6400 train 67.48%, 1600 test 64.50%
ep 33, loss: 92.26, 6400 train 66.77%, 1600 test 65.50%
ep 34, loss: 89.96, 6400 train 67.92%, 1600 test 58.88%
ep 35, loss: 90.00, 6400 train 67.92%, 1600 test 61.19%
ep 36, loss: 88.45, 6400 train 68.64%, 1600 test 62.56%
ep 37, loss: 88.31, 6400 train 68.80%, 1600 test 59.88%
ep 38, loss: 86.74, 6400 train 69.41%, 1600 test 64.12%
ep 39, loss: 85.33, 6400 train 69.53%, 1600 test 55.06%
ep 40, loss: 86.63, 6400 train 69.20%, 1600 test 62.88%
[[157.   0.   0.  21.   2.   2.   0.  17.]
 [  0.  91.   1.  18.  14.  36.   1.  48.]
 [ 16.   7.  72.  44.  32.  25.   3.  10.]
 [ 10.   0.   0. 175.   3.   6.   0.   7.]
 [  1.   8.   8.  37. 127.  12.   1.   6.]
 [  1.   1.   0.  20.   2. 152.   0.   3.]
 [ 47.   7.   0.  34.   2.  39.  63.  19.]
 [  9.   7.   0.   2.   3.   2.   0. 169.]]
   Model saved to checkModel.pth
ep 41, loss: 84.52, 6400 train 70.72%, 1600 test 61.50%
ep 42, loss: 83.61, 6400 train 70.42%, 1600 test 64.25%
ep 43, loss: 84.08, 6400 train 70.59%, 1600 test 68.31%
ep 44, loss: 83.95, 6400 train 70.47%, 1600 test 64.50%
ep 45, loss: 79.47, 6400 train 71.88%, 1600 test 66.06%
ep 46, loss: 79.69, 6400 train 71.59%, 1600 test 68.19%
ep 47, loss: 79.80, 6400 train 71.70%, 1600 test 66.31%
ep 48, loss: 79.90, 6400 train 72.16%, 1600 test 69.38%
ep 49, loss: 77.30, 6400 train 72.70%, 1600 test 61.00%
ep 50, loss: 79.27, 6400 train 72.38%, 1600 test 57.75%
[[ 94.   0.   1.  71.   5.  10.   0.  18.]
 [  0.  54.   9.  16.  27.  80.   0.  23.]
 [  2.   1. 115.  38.  17.  27.   1.   8.]
 [  1.   0.   3. 178.   2.  17.   0.   0.]
 [  0.   1.  19.  27. 134.  15.   0.   4.]
 [  0.   0.   0.  12.   2. 164.   0.   1.]
 [  8.   4.   3.  56.   3. 103.  25.   9.]
 [  4.   5.   0.  14.   7.   2.   0. 160.]]
   Model saved to checkModel.pth
ep 51, loss: 76.52, 6400 train 72.45%, 1600 test 56.19%
ep 52, loss: 76.48, 6400 train 72.56%, 1600 test 62.69%
ep 53, loss: 75.86, 6400 train 73.25%, 1600 test 69.00%
ep 54, loss: 73.84, 6400 train 73.25%, 1600 test 69.69%
ep 55, loss: 75.13, 6400 train 73.05%, 1600 test 69.94%
ep 56, loss: 73.46, 6400 train 73.78%, 1600 test 67.19%
ep 57, loss: 71.04, 6400 train 74.19%, 1600 test 65.12%
ep 58, loss: 71.61, 6400 train 74.72%, 1600 test 65.31%
ep 59, loss: 70.65, 6400 train 74.88%, 1600 test 65.62%
ep 60, loss: 70.73, 6400 train 74.92%, 1600 test 70.69%
[[167.   0.   0.  16.   4.   4.   1.   7.]
 [  2. 115.   1.   8.   8.  41.  11.  23.]
 [ 13.   8.  91.  25.  24.  29.  10.   9.]
 [ 10.   1.   1. 167.   2.  15.   2.   3.]
 [  2.   8.   5.  17. 142.  17.   3.   6.]
 [  0.   0.   0.   8.   2. 164.   2.   3.]
 [ 25.   9.   0.  19.   1.  32. 124.   1.]
 [ 12.   9.   0.   6.   1.   2.   1. 161.]]
   Model saved to checkModel.pth
ep 61, loss: 71.00, 6400 train 75.08%, 1600 test 71.69%
ep 62, loss: 69.92, 6400 train 74.89%, 1600 test 70.69%
ep 63, loss: 70.67, 6400 train 74.97%, 1600 test 73.00%
ep 64, loss: 67.64, 6400 train 76.03%, 1600 test 60.31%
ep 65, loss: 66.08, 6400 train 76.95%, 1600 test 67.50%
ep 66, loss: 66.52, 6400 train 76.50%, 1600 test 71.62%
ep 67, loss: 66.71, 6400 train 76.31%, 1600 test 61.75%
ep 68, loss: 66.23, 6400 train 76.70%, 1600 test 66.19%
ep 69, loss: 63.50, 6400 train 77.22%, 1600 test 68.25%
ep 70, loss: 65.23, 6400 train 76.53%, 1600 test 69.88%
[[146.   0.   0.  19.   4.   2.   3.  25.]
 [  0.  97.   4.  14.   6.  27.   3.  58.]
 [  5.   3. 114.  30.  14.  24.   6.  13.]
 [  5.   1.   1. 172.   3.   9.   0.  10.]
 [  0.  10.   9.  25. 137.  10.   1.   8.]
 [  1.   1.   1.   5.   2. 163.   1.   5.]
 [ 13.   7.   0.  26.   0.  36. 113.  16.]
 [  6.   2.   0.   5.   1.   2.   0. 176.]]
   Model saved to checkModel.pth
ep 71, loss: 63.76, 6400 train 77.45%, 1600 test 71.88%
ep 72, loss: 64.82, 6400 train 77.55%, 1600 test 69.38%
ep 73, loss: 61.84, 6400 train 78.12%, 1600 test 64.56%
ep 74, loss: 63.36, 6400 train 77.77%, 1600 test 68.81%
ep 75, loss: 63.74, 6400 train 77.42%, 1600 test 68.38%
ep 76, loss: 62.50, 6400 train 78.00%, 1600 test 73.06%
ep 77, loss: 60.78, 6400 train 78.89%, 1600 test 69.62%
ep 78, loss: 61.13, 6400 train 78.48%, 1600 test 63.75%
ep 79, loss: 60.68, 6400 train 78.16%, 1600 test 68.88%
ep 80, loss: 59.85, 6400 train 78.66%, 1600 test 72.75%
[[170.   0.   1.  13.   1.   2.   9.   3.]
 [  1.  92.   1.  15.  16.  38.  30.  16.]
 [  8.   5. 115.  32.  10.  23.  10.   6.]
 [ 11.   0.   1. 173.   2.  10.   4.   0.]
 [  2.   3.   9.  30. 137.  15.   2.   2.]
 [  0.   0.   0.   8.   1. 166.   4.   0.]
 [ 12.   7.   0.  17.   1.  17. 156.   1.]
 [ 17.   6.   0.   8.   3.   2.   1. 155.]]
   Model saved to checkModel.pth
ep 81, loss: 58.46, 6400 train 79.45%, 1600 test 74.56%
ep 82, loss: 60.32, 6400 train 78.20%, 1600 test 72.00%
ep 83, loss: 58.42, 6400 train 79.62%, 1600 test 65.81%
ep 84, loss: 57.04, 6400 train 79.62%, 1600 test 73.00%
ep 85, loss: 56.90, 6400 train 79.58%, 1600 test 69.94%
ep 86, loss: 57.93, 6400 train 79.20%, 1600 test 67.25%
ep 87, loss: 56.89, 6400 train 79.39%, 1600 test 77.56%
ep 88, loss: 56.08, 6400 train 80.22%, 1600 test 71.25%
ep 89, loss: 55.88, 6400 train 79.89%, 1600 test 71.19%
ep 90, loss: 56.39, 6400 train 79.34%, 1600 test 70.06%
[[142.   1.   1.  34.   3.   6.   4.   8.]
 [  0. 121.   1.  10.  12.  42.   7.  16.]
 [ 10.   7. 105.  26.  12.  37.   4.   8.]
 [  2.   2.   0. 176.   2.  18.   0.   1.]
 [  1.   9.   9.  23. 134.  20.   1.   3.]
 [  0.   0.   1.   5.   0. 172.   1.   0.]
 [ 13.  11.   0.  28.   1.  47. 109.   2.]
 [  7.  10.   0.   7.   2.   4.   0. 162.]]
   Model saved to checkModel.pth
ep 91, loss: 55.71, 6400 train 80.05%, 1600 test 68.19%
ep 92, loss: 55.16, 6400 train 80.88%, 1600 test 70.38%
ep 93, loss: 53.83, 6400 train 80.80%, 1600 test 71.44%
ep 94, loss: 54.05, 6400 train 80.38%, 1600 test 70.88%
ep 95, loss: 54.99, 6400 train 80.75%, 1600 test 74.88%
ep 96, loss: 51.64, 6400 train 81.67%, 1600 test 73.00%
ep 97, loss: 53.57, 6400 train 80.86%, 1600 test 72.12%
ep 98, loss: 50.10, 6400 train 82.31%, 1600 test 72.31%
ep 99, loss: 53.67, 6400 train 81.44%, 1600 test 71.88%
ep 100, loss: 51.74, 6400 train 81.59%, 1600 test 62.00%
[[146.   0.   0.  40.   0.   3.   0.  10.]
 [  0.  69.   1.  27.   4.  79.   1.  28.]
 [  8.   0.  98.  42.   3.  44.   3.  11.]
 [  2.   1.   0. 186.   0.  11.   0.   1.]
 [  2.   3.   6.  51. 109.  25.   0.   4.]
 [  0.   0.   0.  15.   0. 164.   0.   0.]
 [ 25.   2.   0.  65.   0.  52.  58.   9.]
 [  9.   6.   0.  12.   0.   3.   0. 162.]]
   Model saved to checkModel.pth
ep 101, loss: 52.58, 6400 train 81.09%, 1600 test 71.44%
ep 102, loss: 50.69, 6400 train 81.92%, 1600 test 67.81%
ep 103, loss: 49.80, 6400 train 82.22%, 1600 test 74.31%
ep 104, loss: 49.44, 6400 train 82.27%, 1600 test 73.75%
ep 105, loss: 49.33, 6400 train 82.42%, 1600 test 74.62%
ep 106, loss: 49.29, 6400 train 82.45%, 1600 test 74.31%
ep 107, loss: 48.86, 6400 train 82.83%, 1600 test 73.69%
ep 108, loss: 50.17, 6400 train 82.34%, 1600 test 72.31%
ep 109, loss: 47.19, 6400 train 83.22%, 1600 test 72.38%
ep 110, loss: 48.66, 6400 train 83.03%, 1600 test 74.56%
[[165.   1.   0.  15.   4.   4.   5.   5.]
 [  1. 131.   0.   5.  18.  39.  10.   5.]
 [  9.   5. 111.  18.  25.  29.   8.   4.]
 [  6.   1.   2. 170.   6.  16.   0.   0.]
 [  0.   4.   1.  10. 166.  17.   2.   0.]
 [  0.   2.   0.   3.   2. 170.   2.   0.]
 [ 19.  12.   2.  13.   1.  32. 131.   1.]
 [ 10.  15.   0.   8.   4.   5.   1. 149.]]
   Model saved to checkModel.pth
ep 111, loss: 48.56, 6400 train 82.59%, 1600 test 72.75%
ep 112, loss: 47.69, 6400 train 82.92%, 1600 test 72.81%
ep 113, loss: 45.95, 6400 train 83.91%, 1600 test 74.56%
ep 114, loss: 47.69, 6400 train 82.81%, 1600 test 75.25%
ep 115, loss: 45.83, 6400 train 83.94%, 1600 test 74.44%
ep 116, loss: 47.70, 6400 train 82.64%, 1600 test 74.81%
ep 117, loss: 45.08, 6400 train 84.36%, 1600 test 69.75%
ep 118, loss: 45.69, 6400 train 83.89%, 1600 test 74.69%
ep 119, loss: 45.21, 6400 train 83.70%, 1600 test 75.19%
ep 120, loss: 44.46, 6400 train 84.41%, 1600 test 72.81%
[[174.   0.   0.  12.   5.   1.   2.   5.]
 [  0. 131.   0.  11.  16.  20.  13.  18.]
 [ 11.   8. 101.  26.  34.  14.   9.   6.]
 [ 15.   0.   0. 175.   7.   3.   1.   0.]
 [  2.  11.   2.  16. 162.   3.   2.   2.]
 [  1.   2.   1.  22.   3. 143.   5.   2.]
 [ 31.  12.   1.  24.   2.  10. 130.   1.]
 [ 21.   8.   0.  11.   2.   1.   0. 149.]]
   Model saved to checkModel.pth
ep 121, loss: 44.12, 6400 train 84.16%, 1600 test 76.25%
ep 122, loss: 46.05, 6400 train 83.78%, 1600 test 74.06%
ep 123, loss: 45.14, 6400 train 84.22%, 1600 test 72.44%
ep 124, loss: 44.97, 6400 train 84.12%, 1600 test 75.31%
ep 125, loss: 42.70, 6400 train 84.45%, 1600 test 73.62%
ep 126, loss: 42.06, 6400 train 84.41%, 1600 test 72.31%
ep 127, loss: 43.40, 6400 train 84.78%, 1600 test 71.06%
ep 128, loss: 41.67, 6400 train 85.20%, 1600 test 73.50%
ep 129, loss: 40.41, 6400 train 84.83%, 1600 test 76.12%
ep 130, loss: 43.06, 6400 train 84.66%, 1600 test 73.06%
[[163.   0.   0.  16.   5.   6.   1.   8.]
 [  0. 119.   3.  10.   4.  37.   0.  36.]
 [  6.   6. 129.  22.  10.  25.   5.   6.]
 [  8.   1.   0. 171.   2.  17.   0.   2.]
 [  0.   8.  11.  18. 143.  15.   1.   4.]
 [  0.   2.   1.   6.   0. 167.   1.   2.]
 [ 19.   6.   1.  22.   0.  49. 109.   5.]
 [  8.   4.   1.   7.   2.   2.   0. 168.]]
   Model saved to checkModel.pth
ep 131, loss: 43.06, 6400 train 84.92%, 1600 test 76.69%
ep 132, loss: 42.72, 6400 train 85.00%, 1600 test 74.00%
ep 133, loss: 41.20, 6400 train 85.52%, 1600 test 72.50%
ep 134, loss: 41.32, 6400 train 85.30%, 1600 test 78.94%
ep 135, loss: 38.82, 6400 train 86.44%, 1600 test 73.88%
ep 136, loss: 40.64, 6400 train 85.81%, 1600 test 74.31%
ep 137, loss: 41.26, 6400 train 85.41%, 1600 test 75.00%
ep 138, loss: 41.70, 6400 train 84.77%, 1600 test 74.38%
ep 139, loss: 39.60, 6400 train 85.56%, 1600 test 73.00%
ep 140, loss: 40.65, 6400 train 85.86%, 1600 test 72.88%
[[151.   0.   1.  36.   3.   2.   1.   5.]
 [  1. 124.   2.  17.  21.  19.   2.  23.]
 [  7.   4. 132.  24.  23.  11.   2.   6.]
 [  2.   2.   0. 186.   5.   5.   0.   1.]
 [  0.   6.   8.  21. 158.   4.   0.   3.]
 [  0.   4.   2.  14.   3. 154.   1.   1.]
 [ 30.  13.   4.  32.   3.  22. 101.   6.]
 [ 11.   6.   1.  11.   2.   1.   0. 160.]]
   Model saved to checkModel.pth

```
