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
