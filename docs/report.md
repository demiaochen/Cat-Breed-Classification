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
ep 1, loss: 59.43, 6400 train 27.14%, 1600 test 25.87%
ep 2, loss: 58.79, 6400 train 28.08%, 1600 test 25.69%
ep 3, loss: 58.14, 6400 train 29.22%, 1600 test 25.44%
ep 4, loss: 57.50, 6400 train 29.33%, 1600 test 28.50%
ep 5, loss: 56.90, 6400 train 30.39%, 1600 test 29.62%
ep 6, loss: 56.34, 6400 train 30.86%, 1600 test 30.25%
ep 7, loss: 55.79, 6400 train 31.69%, 1600 test 30.50%
ep 8, loss: 55.30, 6400 train 32.44%, 1600 test 31.06%
ep 9, loss: 54.88, 6400 train 33.16%, 1600 test 31.25%
ep 10, loss: 54.50, 6400 train 33.80%, 1600 test 31.94%
[[134.   3.   0.  15.   0.   2.  12.  35.]
 [  5.  42.   0.   8.  37.  39.   4.  51.]
 [  8.  22.   0.  15.  63.  56.   7.  17.]
 [ 31.  12.   0.  78.  12.  46.  13.  17.]
 [  3.  37.   0.  29.  89.  54.   2.  13.]
 [  4.  29.   0.  54.  32.  57.  11.  20.]
 [ 41.   9.   0.  54.   8.  22.  23.  40.]
 [ 37.  23.   0.   7.   5.  11.  14.  88.]]
 ```

ToDo: Changing Optimzer from SGD with Momentum to Adam?
