# Approaches

## 1st attempt

No transformations applied to input image, size 80x80 images with 3 channels (RGB)

**2 Convolution Layer + Max Pooling:**

Layer 1 => input channel = 3, output channel = 30, filter size = 5; width after maxpool => (80+1-5)/2 = 38;

layer 2 => input channel = 30, output channel = 120, filter size = 5; width after maxpool => (38+1-5)/2 = 17;

**3 Fully connected layers:**

Layer 1 => input = ```120*17*17 = 34680``` to 200 nodes in 1st Hidden Layer

Layer 2 => input = 200 to 200 nodes in 2nd Hidden Later

Layer 3 => input = 200 to 8 nodes in OUTPUT layer.

10 epochs, with 80-20 traing to test dataset split. SGD Optimizer, lr = 0.01, momentum = 0.5

```shell
ep 1, loss: 66.38, 6400 train 12.45%, 1600 test 12.81%
ep 2, loss: 65.72, 6400 train 15.59%, 1600 test 21.81%
ep 3, loss: 63.61, 6400 train 23.70%, 1600 test 26.50%
ep 4, loss: 60.21, 6400 train 25.92%, 1600 test 24.81%
ep 5, loss: 59.14, 6400 train 27.02%, 1600 test 24.44%
ep 6, loss: 58.61, 6400 train 27.67%, 1600 test 25.06%
ep 7, loss: 58.10, 6400 train 28.58%, 1600 test 26.31%
ep 8, loss: 57.63, 6400 train 29.50%, 1600 test 27.31%
ep 9, loss: 57.22, 6400 train 30.38%, 1600 test 28.62%
ep 10, loss: 56.82, 6400 train 31.00%, 1600 test 28.69%
[[158.   0.   0.  22.   1.   1.  18.  12.]
 [  9.  37.  26.  45.  39.   6.   4.  23.]
 [ 24.  14.  55.  42.  48.   4.   5.  13.]
 [ 43.   4.   8.  77.  14.  11.  18.   7.]
 [  8.  21.  56.  58.  58.  10.   4.   8.]
 [ 20.  25.  14.  84.  12.   9.   9.  12.]
 [ 80.   1.   5.  77.   2.   2.  27.  17.]
 [ 55.  17.   3.  54.   6.   5.  15.  38.]]
```

## 2nd Attempt

No transformations applied to input image, size 80x80 images with 3 channels (RGB)

**3 Convolution Layer + Max Pooling:**

Layer 1 => input channel = 3, output channel = 30, filter size = 5; width after maxpool => (80+1-5)/2 = 38;

layer 2 => input channel = 30, output channel = 90, filter size = 5; width after maxpool => (38+1-5)/2 = 17;

layer 3 => input channel = 90, output channel = 270, filter size = 5; width after maxpool => (17+1-5)/2 = 6;

**3 Fully connected layers:**

Layer 1 => input = ```270*6*6 = 9720``` to 300 nodes in 1st Hidden Layer

Layer 2 => input = 300 to 300 nodes in 2nd Hidden Later

Layer 3 => input = 300 to 8 nodes in OUTPUT layer.

**10** epochs, with 80-20 traing to test dataset split. SGD Optimizer, lr = 0.01, momentum = 0.5

```shell
Start training...
ep 1, loss: 66.53, 6400 train 12.36%, 1600 test 13.94%
ep 2, loss: 66.46, 6400 train 12.73%, 1600 test 13.81%
ep 3, loss: 66.39, 6400 train 13.50%, 1600 test 13.75%
ep 4, loss: 66.28, 6400 train 13.95%, 1600 test 14.19%
ep 5, loss: 66.08, 6400 train 14.56%, 1600 test 14.75%
ep 6, loss: 65.68, 6400 train 16.31%, 1600 test 17.25%
ep 7, loss: 64.76, 6400 train 20.53%, 1600 test 21.00%
ep 8, loss: 62.69, 6400 train 23.94%, 1600 test 22.00%
ep 9, loss: 60.49, 6400 train 25.64%, 1600 test 21.94%
ep 10, loss: 59.70, 6400 train 26.67%, 1600 test 21.56%
[[128.   3.   0.  20.   0.   1.   0.  35.]
 [ 12.  67.   0.  19.   1.  16.   0.  75.]
 [ 20.  48.   0.  26.   7.  37.   0.  55.]
 [ 64.  17.   0.  36.   1.  15.   0.  55.]
 [  5.  75.   0.  18.   9.  36.   0.  66.]
 [ 20.  43.   0.  39.   0.  17.   0.  87.]
 [ 77.  11.   0.  43.   0.  15.   0.  60.]
 [ 69.  38.   0.  18.   1.   7.   0.  88.]]
 ```

Changing Optimzer from SGD to Adam

```shell
Start training...
ep 1, loss: 252.57, 6400 train 13.86%, 1600 test 13.88%
ep 2, loss: 66.15, 6400 train 13.25%, 1600 test 13.25%
ep 3, loss: 66.58, 6400 train 13.44%, 1600 test 13.12%
ep 4, loss: 64.74, 6400 train 16.00%, 1600 test 19.31%
ep 5, loss: 61.81, 6400 train 21.44%, 1600 test 24.31%
ep 6, loss: 59.46, 6400 train 24.89%, 1600 test 27.50%
ep 7, loss: 57.96, 6400 train 27.28%, 1600 test 28.94%
ep 8, loss: 56.98, 6400 train 28.83%, 1600 test 30.56%
ep 9, loss: 56.26, 6400 train 30.52%, 1600 test 30.69%
ep 10, loss: 55.87, 6400 train 30.77%, 1600 test 31.44%
[[138.   0.   0.   8.   0.   1.  23.  38.]
 [  6.  44.  18.  18.  26.   8.  24.  35.]
 [ 15.  31.  72.  15.  40.   3.  14.  18.]
 [ 15.  14.  15.  33.  26.  10.  52.  32.]
 [  3.  38.  48.  14.  47.   9.  15.  19.]
 [  2.  34.  38.  35.  30.   7.  27.  37.]
 [ 41.  11.   4.  20.   7.   1.  58.  53.]
 [ 50.   9.   0.   9.   6.   0.  32. 104.]]
```

## 4th attempt
We get better accuracy on training and test sets. Changing lr to 0.001, Loss Func to Cross Entropy Loss, 120 layer output 2nd conv layer, 360 layer output in 3rd conv layer and 500 nodes in 2 FC hidden layers (followed by output layer):

```shell
Start training...
ep 1, loss: 60.39, 6400 train 23.06%, 1600 test 31.87%
ep 2, loss: 53.13, 6400 train 35.50%, 1600 test 34.81%
ep 3, loss: 49.55, 6400 train 41.05%, 1600 test 40.31%
ep 4, loss: 46.79, 6400 train 45.11%, 1600 test 43.44%
ep 5, loss: 43.90, 6400 train 49.14%, 1600 test 48.38%
ep 6, loss: 42.09, 6400 train 50.56%, 1600 test 48.75%
ep 7, loss: 37.29, 6400 train 56.91%, 1600 test 51.94%
ep 8, loss: 33.38, 6400 train 62.50%, 1600 test 49.94%
ep 9, loss: 29.11, 6400 train 67.20%, 1600 test 52.38%
ep 10, loss: 26.82, 6400 train 69.38%, 1600 test 52.69%
[[156.   3.   2.   5.   0.   0.  22.  18.]
 [  1.  85.  13.   2.  16.  17.  44.  32.]
 [ 15.  24.  86.  14.  15.  16.  13.   6.]
 [ 46.   7.  14.  98.   5.  12.  26.   7.]
 [  2.  49.  63.  10.  65.   3.  13.   5.]
 [  1.   7.  14.  12.   1. 119.  32.   0.]
 [ 21.  11.   2.   6.   1.   5. 128.   9.]
 [ 49.  25.   2.   0.   3.   1.  15. 106.]]
 ```

Then we add a padding of 2 to our all 3 of our convolution layers mentioned above..

## Attempt 5

Additional Conv layer and dropout (20%)

```python
Network(
  (conv1): Conv2d(3, 30, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (conv2): Conv2d(30, 120, kernel_size=(5, 5), stride=(1, 1))
  (conv3): Conv2d(120, 360, kernel_size=(3, 3), stride=(1, 1))
  (conv4): Conv2d(360, 540, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=4860, out_features=1000, bias=True)
  (fc2): Linear(in_features=1000, out_features=1000, bias=True)
  (fc3): Linear(in_features=1000, out_features=8, bias=True)
  (dropout): Dropout(p=0.25, inplace=False)
)
```

Results :

```shell
Start training...
ep 1, loss: 64.05, 6400 train 18.41%, 1600 test 23.44%
ep 2, loss: 57.54, 6400 train 26.44%, 1600 test 32.88%
ep 3, loss: 52.35, 6400 train 34.80%, 1600 test 37.25%
ep 4, loss: 50.65, 6400 train 37.61%, 1600 test 40.38%
ep 5, loss: 48.11, 6400 train 41.19%, 1600 test 42.06%
ep 6, loss: 44.31, 6400 train 47.61%, 1600 test 50.12%
ep 7, loss: 42.10, 6400 train 50.44%, 1600 test 51.12%
ep 8, loss: 39.20, 6400 train 54.47%, 1600 test 51.38%
ep 9, loss: 36.21, 6400 train 58.20%, 1600 test 45.38%
ep 10, loss: 34.87, 6400 train 60.09%, 1600 test 54.19%
[[105.   2.   0.  11.   0.   0.  26.  56.]
 [  0.  94.   3.   4.  22.   7.  33.  37.]
 [  8.  16.  54.  36.  60.   6.  14.   9.]
 [  4.   7.   2. 139.   8.   5.  31.   8.]
 [  0.  47.  21.  17.  72.   2.  17.   6.]
 [  0.  14.  11.  26.   7. 105.  49.   0.]
 [  9.  15.   0.   9.   3.   6. 146.  16.]
 [ 13.   9.   1.   3.   1.   0.  16. 152.]]
   Model saved to checkModel.pth
ep 11, loss: 32.56, 6400 train 63.27%, 1600 test 53.69%
ep 12, loss: 28.60, 6400 train 67.73%, 1600 test 56.38%
ep 13, loss: 28.30, 6400 train 68.06%, 1600 test 55.12%
ep 14, loss: 26.72, 6400 train 70.22%, 1600 test 55.62%
ep 15, loss: 23.72, 6400 train 72.97%, 1600 test 55.56%
```

Using this code we can find the mean and std deviation across 3 image channels and apply that to normalize (transform)

```python
from torch.utils.data import TensorDataset, DataLoader

nimages = 0
mean = 0.0
var = 0.0
for i_batch, batch_target in enumerate(trainloader):
    batch = batch_target[0]
    # Rearrange batch to be the shape of [B, C, W * H]
    batch = batch.view(batch.size(0), batch.size(1), -1)
    # Update total number of images
    nimages += batch.size(0)
    # Compute mean and std here
    mean += batch.mean(2).sum(0) 
    var += batch.var(2).sum(0)

mean /= nimages
var /= nimages
std = torch.sqrt(var)

print(mean)
print(std)
```

Source: https://stackoverflow.com/a/60803379

We find the output:

```shell
tensor([0.4817, 0.4347, 0.3928]) #mean
tensor([0.2483, 0.2402, 0.2335]) #std
```

```python3
Network(
  (conv1): Conv2d(3, 30, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (conv2): Conv2d(30, 120, kernel_size=(5, 5), stride=(1, 1))
  (conv3): Conv2d(120, 360, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=5760, out_features=1000, bias=True)
  (fc2): Linear(in_features=1000, out_features=1000, bias=True)
  (fc3): Linear(in_features=1000, out_features=8, bias=True)
  (dropout): Dropout(p=0.4, inplace=False)
)
```

```shell
Start training...
ep 1, loss: 59.04, 6400 train 26.70%, 1600 test 31.81%
ep 2, loss: 53.37, 6400 train 35.67%, 1600 test 34.75%
ep 3, loss: 49.79, 6400 train 40.69%, 1600 test 46.06%
ep 4, loss: 45.78, 6400 train 46.70%, 1600 test 48.81%
ep 5, loss: 43.42, 6400 train 49.28%, 1600 test 49.56%
ep 6, loss: 41.39, 6400 train 52.28%, 1600 test 52.25%
ep 7, loss: 38.63, 6400 train 55.95%, 1600 test 53.44%
ep 8, loss: 36.48, 6400 train 58.63%, 1600 test 54.06%
ep 9, loss: 35.95, 6400 train 59.38%, 1600 test 53.00%
ep 10, loss: 33.66, 6400 train 62.28%, 1600 test 56.00%
[[111.   4.   2.  23.   6.   1.   8.  42.]
 [  0. 129.  10.   1.  20.   8.  11.  13.]
 [ 12.  24.  83.  23.  21.  11.   7.  12.]
 [  9.  16.  10. 114.  19.   9.   4.  12.]
 [  2.  50.  36.   8.  99.   7.   0.  13.]
 [  0.  28.  11.  10.   5. 127.  15.   1.]
 [ 12.  44.   2.  14.   2.  14.  86.  22.]
 [ 13.  39.   1.   4.   6.   0.   7. 147.]]
   Model saved to checkModel.pth
ep 11, loss: 31.53, 6400 train 64.50%, 1600 test 55.12%
ep 12, loss: 28.74, 6400 train 67.73%, 1600 test 54.69%
ep 13, loss: 28.50, 6400 train 67.83%, 1600 test 53.87%
ep 14, loss: 26.49, 6400 train 70.03%, 1600 test 55.38%
ep 15, loss: 24.62, 6400 train 72.19%, 1600 test 53.69%
ep 16, loss: 21.73, 6400 train 75.92%, 1600 test 54.44%
ep 17, loss: 18.82, 6400 train 79.14%, 1600 test 51.38%
ep 18, loss: 18.16, 6400 train 80.19%, 1600 test 54.75%
ep 19, loss: 16.69, 6400 train 81.53%, 1600 test 53.25%
ep 20, loss: 18.23, 6400 train 80.08%, 1600 test 52.06%
[[141.   0.   5.   4.   2.   5.  31.   9.]
 [  0.  78.  24.   2.  15.  24.  43.   6.]
 [ 21.  14.  92.  16.  11.  23.  14.   2.]
 [ 27.   6.  13.  85.   3.  29.  25.   5.]
 [  5.  45.  62.  11.  65.  15.   9.   3.]
 [  1.   5.  12.   3.   1. 163.  12.   0.]
 [ 16.  10.   7.   3.   2.  25. 129.   4.]
 [ 44.  28.  11.   5.   4.   6.  39.  80.]]
   Model saved to checkModel.pth
ep 21, loss: 17.46, 6400 train 80.48%, 1600 test 51.88%
ep 22, loss: 14.85, 6400 train 83.86%, 1600 test 55.75%
ep 23, loss: 12.51, 6400 train 86.80%, 1600 test 54.62%
ep 24, loss: 10.99, 6400 train 87.92%, 1600 test 54.00%
ep 25, loss: 9.46, 6400 train 89.97%, 1600 test 53.56%
ep 26, loss: 8.06, 6400 train 91.22%, 1600 test 54.75%
ep 27, loss: 7.92, 6400 train 91.50%, 1600 test 55.31%
ep 28, loss: 7.51, 6400 train 92.00%, 1600 test 52.94%
ep 29, loss: 7.10, 6400 train 92.69%, 1600 test 54.44%
ep 30, loss: 5.67, 6400 train 94.20%, 1600 test 53.87%
[[162.   0.   1.   4.   3.   0.  12.  15.]
 [  5.  64.  16.   3.  28.   7.  43.  26.]
 [ 22.  14.  74.  18.  37.  12.  12.   4.]
 [ 38.   6.  16.  98.   9.   3.  20.   3.]
 [  6.  26.  33.   8. 115.   3.   9.  15.]
 [  2.  14.  18.  11.   7. 117.  27.   1.]
 [ 31.  13.   6.   5.   4.  10. 115.  12.]
 [ 57.  12.   4.   6.   4.   1.  16. 117.]]
   Model saved to checkModel.pth
ep 31, loss: 5.20, 6400 train 94.75%, 1600 test 54.87%
ep 32, loss: 4.34, 6400 train 95.52%, 1600 test 54.56%
ep 33, loss: 4.15, 6400 train 95.56%, 1600 test 55.38%
ep 34, loss: 3.92, 6400 train 95.83%, 1600 test 55.38%
ep 35, loss: 3.85, 6400 train 95.98%, 1600 test 54.00%
ep 36, loss: 3.97, 6400 train 95.72%, 1600 test 55.25%
ep 37, loss: 3.74, 6400 train 96.34%, 1600 test 55.38%
ep 38, loss: 2.86, 6400 train 97.22%, 1600 test 54.50%
ep 39, loss: 2.66, 6400 train 97.42%, 1600 test 54.19%
ep 40, loss: 3.04, 6400 train 96.69%, 1600 test 54.94%
[[133.   2.   5.   9.   5.   5.  25.  13.]
 [  0.  96.  10.   1.  25.  12.  32.  16.]
 [ 14.  15.  76.  18.  36.  20.  10.   4.]
 [ 13.  15.  18.  81.   6.  33.  15.  12.]
 [  3.  39.  27.   8. 117.   6.   4.  11.]
 [  0.  16.   9.   5.   7. 149.  10.   1.]
 [ 10.  19.   8.   4.   5.  29. 114.   7.]
 [ 26.  35.   3.   3.   3.   1.  33. 113.]]
   Model saved to checkModel.pth
ep 41, loss: 3.04, 6400 train 96.88%, 1600 test 54.62%
ep 42, loss: 3.51, 6400 train 96.31%, 1600 test 55.25%
ep 43, loss: 2.93, 6400 train 96.91%, 1600 test 53.19%
ep 44, loss: 2.77, 6400 train 97.22%, 1600 test 55.38%
ep 45, loss: 3.08, 6400 train 96.94%, 1600 test 54.50%
ep 46, loss: 2.67, 6400 train 97.38%, 1600 test 54.87%
ep 47, loss: 2.92, 6400 train 97.03%, 1600 test 55.88%
ep 48, loss: 2.58, 6400 train 97.36%, 1600 test 55.38%
ep 49, loss: 2.27, 6400 train 97.62%, 1600 test 54.56%
ep 50, loss: 2.29, 6400 train 97.72%, 1600 test 55.12%
[[125.   0.   4.  17.   4.   3.  18.  26.]
 [  0.  81.  16.   9.  26.  13.  30.  17.]
 [ 17.  11.  69.  27.  39.  13.  12.   5.]
 [ 19.   7.   8. 122.   8.   6.  15.   8.]
 [  4.  25.  27.  13. 121.   9.   8.   8.]
 [  0.  15.   7.  19.   6. 128.  20.   2.]
 [ 15.  16.   1.  15.   8.  16. 109.  16.]
 [ 22.  22.   4.  14.   5.   2.  21. 127.]]
```

Network(
  (conv1): Conv2d(3, 30, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (conv2): Conv2d(30, 120, kernel_size=(5, 5), stride=(1, 1))
  (conv3): Conv2d(120, 360, kernel_size=(3, 3), stride=(1, 1))
  (conv4): Conv2d(360, 540, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=540, out_features=1000, bias=True)
  (fc2): Linear(in_features=1000, out_features=1000, bias=True)
  (fc3): Linear(in_features=1000, out_features=8, bias=True)
  (dropout): Dropout(p=0.4, inplace=False)
)

lr = 


### Read later

https://towardsdatascience.com/improves-cnn-performance-by-applying-data-transformation-bf86b3f4cef4
