# report

## notes

model 3 training seems good, loss decrease steadily, but testing accuracy improves hard around 75%

**TODO**：   

## brief

model 1: kernel size 5 3 3 3 -> 71%

model 2: change kernel size to 3 2 2 2 -> 69%

model 3: kernel size 5 2 2 2 and add batchnorm -> 78%


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
