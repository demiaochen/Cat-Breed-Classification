## report

model size: 45.6 MB

networks: 4 cnn + 3 fc

highest accuracy: 60%

**conclusion**: 
* very easy to overfit 
* no benefits of using log_softmax *
* naive cnn + fc + data augumentation hits limitations around 70% (a rough guess based on some 100MB models I trained)

**TODO**: searching papers to look for a better image classfication model, to get insipirations to make a model with less than 13M num of parameters (we have 50MB size restriction on .pth file).

## parameters
```shell
+---------------------+------------+
|       Modules       | Parameters |
+---------------------+------------+
| cnn_layers.0.weight |    2400    |
|  cnn_layers.0.bias  |     32     |
| cnn_layers.2.weight |   27648    |
|  cnn_layers.2.bias  |     96     |
| cnn_layers.5.weight |   51840    |
|  cnn_layers.5.bias  |     60     |
| cnn_layers.8.weight |   17280    |
|  cnn_layers.8.bias  |     32     |
|  fc_layers.1.weight |  11089920  |
|   fc_layers.1.bias  |    960     |
|  fc_layers.4.weight |   768000   |
|   fc_layers.4.bias  |    800     |
|  fc_layers.6.weight |    6400    |
|   fc_layers.6.bias  |     8      |
+---------------------+------------+
Total Trainable Params: 11,965,476
```
Function to generate parameter table
```python
    from prettytable import PrettyTable
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
    count_parameters(net)
```
## result without softmax (lr=0.0005)

```shell
Start training...
ep 1, loss: 192.91, 6400 train 21.28%, 1600 test 29.12%
ep 2, loss: 181.45, 6400 train 26.88%, 1600 test 32.56%
ep 3, loss: 171.77, 6400 train 33.05%, 1600 test 38.81%
ep 4, loss: 167.25, 6400 train 35.27%, 1600 test 42.56%
ep 5, loss: 161.09, 6400 train 38.31%, 1600 test 41.44%
ep 6, loss: 156.95, 6400 train 41.33%, 1600 test 46.38%
ep 7, loss: 152.90, 6400 train 41.77%, 1600 test 47.25%
ep 8, loss: 147.00, 6400 train 45.05%, 1600 test 47.88%
ep 9, loss: 146.09, 6400 train 45.17%, 1600 test 50.69%
ep 10, loss: 143.45, 6400 train 46.62%, 1600 test 51.38%
[[163.   0.   0.   4.   2.   2.  15.  21.]
 [  1.  63.   9.   7.  36.  21.  26.  30.]
 [ 16.   9.  58.  15.  39.  27.   9.   9.]
 [ 40.   7.   9.  71.  10.  33.  26.   4.]
 [  0.  30.  35.  14. 112.   8.  11.  13.]
 [  3.   4.  10.   6.   4. 120.  26.   1.]
 [ 31.  13.   0.   6.   2.  32. 101.  15.]
 [ 42.  16.   2.   2.   8.   0.  17. 134.]]
   Model saved to checkModel.pth
ep 11, loss: 138.69, 6400 train 48.33%, 1600 test 51.38%
ep 12, loss: 135.43, 6400 train 50.11%, 1600 test 51.31%
ep 13, loss: 133.71, 6400 train 50.05%, 1600 test 51.19%
ep 14, loss: 129.47, 6400 train 51.28%, 1600 test 51.69%
ep 15, loss: 125.19, 6400 train 52.78%, 1600 test 55.12%
ep 16, loss: 123.96, 6400 train 53.97%, 1600 test 55.88%
ep 17, loss: 118.58, 6400 train 56.20%, 1600 test 56.81%
ep 18, loss: 117.56, 6400 train 56.58%, 1600 test 56.31%
ep 19, loss: 115.24, 6400 train 57.80%, 1600 test 56.38%
ep 20, loss: 111.60, 6400 train 59.17%, 1600 test 56.50%
[[144.   0.   1.   8.   1.   2.  18.  33.]
 [  0.  44.  10.   3.  53.  13.  30.  40.]
 [ 13.  10.  77.  16.  29.  20.   9.   8.]
 [ 22.   1.  13.  95.  14.  12.  32.  11.]
 [  3.  11.  39.  13. 132.   4.  10.  11.]
 [  3.   4.  10.   6.   5. 122.  24.   0.]
 [ 16.  12.   1.   9.   1.  12. 134.  15.]
 [ 24.   7.   2.   1.  11.   0.  20. 156.]]
   Model saved to checkModel.pth
ep 21, loss: 110.05, 6400 train 60.11%, 1600 test 56.44%
ep 22, loss: 107.10, 6400 train 60.47%, 1600 test 56.94%
ep 23, loss: 103.48, 6400 train 62.77%, 1600 test 57.31%
ep 24, loss: 101.84, 6400 train 63.31%, 1600 test 57.12%
ep 25, loss: 98.88, 6400 train 63.91%, 1600 test 58.81%
ep 26, loss: 97.59, 6400 train 64.88%, 1600 test 57.00%
ep 27, loss: 95.92, 6400 train 65.27%, 1600 test 57.38%
ep 28, loss: 93.74, 6400 train 66.36%, 1600 test 58.56%
ep 29, loss: 89.75, 6400 train 67.95%, 1600 test 58.50%
ep 30, loss: 91.12, 6400 train 67.45%, 1600 test 60.06%
[[143.   1.   2.  13.   1.   2.  24.  21.]
 [  0.  93.  10.   4.  20.  12.  22.  32.]
 [ 12.  14.  94.   8.  20.  20.   8.   6.]
 [ 20.   8.  12. 105.  13.  13.  18.  11.]
 [  3.  27.  47.  14. 115.   5.   4.   8.]
 [  2.   7.  13.   5.   1. 130.  14.   2.]
 [ 17.  19.   2.  10.   0.  10. 136.   6.]
 [ 21.  25.   4.   3.   5.   0.  18. 145.]]
   Model saved to checkModel.pth
ep 31, loss: 85.32, 6400 train 69.84%, 1600 test 57.44%
ep 32, loss: 84.52, 6400 train 70.08%, 1600 test 58.50%
ep 33, loss: 84.08, 6400 train 70.11%, 1600 test 58.88%
ep 34, loss: 78.66, 6400 train 71.83%, 1600 test 58.75%
ep 35, loss: 78.65, 6400 train 71.78%, 1600 test 58.75%
ep 36, loss: 76.39, 6400 train 72.48%, 1600 test 59.25%
ep 37, loss: 75.03, 6400 train 73.28%, 1600 test 60.12%
ep 38, loss: 71.93, 6400 train 74.62%, 1600 test 59.31%
ep 39, loss: 71.82, 6400 train 73.97%, 1600 test 58.94%
ep 40, loss: 69.03, 6400 train 75.25%, 1600 test 58.81%
[[142.   0.   3.  11.   1.   1.  21.  28.]
 [  0. 105.   8.   6.  16.   9.  17.  32.]
 [ 12.  21.  83.  13.  21.  16.   9.   7.]
 [ 16.  17.  14.  97.  14.  12.  21.   9.]
 [  3.  42.  42.  11. 112.   3.   2.   8.]
 [  1.  17.  12.   5.   2. 125.  12.   0.]
 [ 13.  23.   1.  12.   1.  11. 129.  10.]
 [ 16.  27.   4.   4.   3.   1.  18. 148.]]
   Model saved to checkModel.pth
ep 41, loss: 68.28, 6400 train 75.75%, 1600 test 59.38%
ep 42, loss: 67.62, 6400 train 76.11%, 1600 test 60.38%
ep 43, loss: 65.51, 6400 train 77.08%, 1600 test 57.56%
ep 44, loss: 63.84, 6400 train 77.45%, 1600 test 57.38%
ep 45, loss: 63.96, 6400 train 77.84%, 1600 test 60.38%
ep 46, loss: 60.66, 6400 train 79.06%, 1600 test 60.56%
ep 47, loss: 61.50, 6400 train 78.56%, 1600 test 58.69%
ep 48, loss: 58.94, 6400 train 79.44%, 1600 test 59.06%
ep 49, loss: 57.96, 6400 train 79.78%, 1600 test 59.00%
ep 50, loss: 56.61, 6400 train 80.58%, 1600 test 59.19%
[[157.   0.   2.   3.   1.   1.  24.  19.]
 [  2.  86.  10.   6.  23.   6.  24.  36.]
 [ 10.  13.  82.  12.  32.   9.  15.   9.]
 [ 32.   6.  13.  84.  21.   6.  27.  11.]
 [  4.  24.  32.   9. 137.   3.   5.   9.]
 [  4.  11.  10.   5.   5.  96.  43.   0.]
 [ 15.  10.   0.   7.   3.   2. 152.  11.]
 [ 20.  17.   4.   1.   6.   0.  20. 153.]]
   Model saved to checkModel.pth
```

## result with softmax (lr=0.0005)

```shell
Start training...
ep 1, loss: 202.74, 6400 train 20.53%, 1600 test 26.31%
ep 2, loss: 199.39, 6400 train 25.17%, 1600 test 26.62%
ep 3, loss: 198.99, 6400 train 26.02%, 1600 test 29.19%
ep 4, loss: 197.03, 6400 train 28.56%, 1600 test 28.94%
ep 5, loss: 195.67, 6400 train 29.33%, 1600 test 31.62%
ep 6, loss: 195.20, 6400 train 29.88%, 1600 test 34.12%
ep 7, loss: 194.59, 6400 train 30.63%, 1600 test 31.81%
ep 8, loss: 194.20, 6400 train 31.48%, 1600 test 35.56%
ep 9, loss: 193.48, 6400 train 32.23%, 1600 test 32.44%
ep 10, loss: 192.34, 6400 train 33.34%, 1600 test 37.19%
[[ 98.   5.   1.  15.   0.   2.  12.  57.]
 [  1.  86.  19.   3.  20.  15.   4.  42.]
 [ 14.  30.  97.  24.  15.  16.   3.  15.]
 [ 20.  23.  15.  80.   7.  24.  16.  15.]
 [  0.  62.  74.   9.  36.  11.   4.   5.]
 [  4.  39.  39.  35.  28.  41.  13.  14.]
 [ 31.  29.   5.  22.   8.  18.  37.  42.]
 [ 27.  35.   3.   3.   4.   0.   8. 120.]]
   Model saved to checkModel.pth
ep 11, loss: 191.67, 6400 train 34.28%, 1600 test 31.62%
ep 12, loss: 191.35, 6400 train 34.42%, 1600 test 36.69%
ep 13, loss: 190.91, 6400 train 35.12%, 1600 test 37.12%
ep 14, loss: 190.97, 6400 train 34.91%, 1600 test 40.12%
ep 15, loss: 190.27, 6400 train 36.03%, 1600 test 36.75%
ep 16, loss: 189.57, 6400 train 36.89%, 1600 test 39.44%
ep 17, loss: 189.24, 6400 train 36.98%, 1600 test 41.94%
ep 18, loss: 187.71, 6400 train 38.31%, 1600 test 38.62%
ep 19, loss: 187.31, 6400 train 39.36%, 1600 test 39.94%
ep 20, loss: 187.41, 6400 train 39.20%, 1600 test 43.75%
[[ 97.   3.   0.  13.   1.   4.  23.  49.]
 [  1.  77.   3.   6.  36.  18.  24.  25.]
 [ 11.  24.  42.  21.  51.  39.  14.  12.]
 [ 19.  16.   3.  87.  10.  22.  30.  13.]
 [  1.  38.  19.  13.  92.  24.  10.   4.]
 [  2.  18.   3.  29.  14. 112.  31.   4.]
 [ 11.  22.   1.  16.   5.  23.  79.  35.]
 [ 29.  25.   0.   2.  13.   1.  16. 114.]]
   Model saved to checkModel.pth
ep 21, loss: 186.48, 6400 train 40.17%, 1600 test 42.50%
ep 22, loss: 186.66, 6400 train 39.59%, 1600 test 44.06%
ep 23, loss: 184.66, 6400 train 41.95%, 1600 test 42.44%
ep 24, loss: 184.27, 6400 train 42.55%, 1600 test 44.44%
ep 25, loss: 185.05, 6400 train 41.53%, 1600 test 45.69%
ep 26, loss: 183.11, 6400 train 43.86%, 1600 test 40.38%
ep 27, loss: 182.83, 6400 train 43.92%, 1600 test 44.19%
ep 28, loss: 182.61, 6400 train 44.00%, 1600 test 46.25%
ep 29, loss: 182.13, 6400 train 44.77%, 1600 test 42.88%
ep 30, loss: 181.24, 6400 train 45.36%, 1600 test 43.88%
[[123.   3.   0.  13.   2.   0.   6.  43.]
 [  2.  49.   2.   7.  55.   2.   4.  69.]
 [ 17.  19.  47.  23.  79.   9.   4.  16.]
 [ 29.  12.   7.  95.  29.   2.   5.  21.]
 [  2.  32.  16.  12. 120.   1.   1.  17.]
 [  2.  24.   7.  48.  20.  77.  25.  10.]
 [ 38.  14.   1.  24.  11.   8.  53.  43.]
 [ 37.   7.   1.   2.  12.   0.   3. 138.]]
   Model saved to checkModel.pth
ep 31, loss: 181.68, 6400 train 45.56%, 1600 test 42.88%
ep 32, loss: 181.11, 6400 train 45.67%, 1600 test 48.75%
ep 33, loss: 180.16, 6400 train 46.70%, 1600 test 47.69%
ep 34, loss: 179.68, 6400 train 46.86%, 1600 test 44.81%
ep 35, loss: 179.76, 6400 train 47.02%, 1600 test 48.38%
ep 36, loss: 179.63, 6400 train 47.33%, 1600 test 48.25%
ep 37, loss: 178.45, 6400 train 48.33%, 1600 test 47.56%
ep 38, loss: 178.85, 6400 train 48.33%, 1600 test 48.25%
ep 39, loss: 178.33, 6400 train 48.17%, 1600 test 48.06%
ep 40, loss: 177.29, 6400 train 49.38%, 1600 test 48.38%
[[ 96.   2.   0.   5.   1.   0.  49.  37.]
 [  1.  83.   4.   4.  18.   5.  49.  26.]
 [ 10.  26.  67.  21.  32.  16.  30.  12.]
 [ 22.   6.   5.  85.   9.  16.  49.   8.]
 [  1.  57.  29.  14.  72.   5.  15.   8.]
 [  1.   8.   6.  17.   8. 131.  40.   2.]
 [ 13.  11.   0.   7.   1.  13. 129.  18.]
 [ 24.  21.   1.   3.   7.   1.  32. 111.]]
   Model saved to checkModel.pth
ep 41, loss: 177.71, 6400 train 49.14%, 1600 test 50.00%
ep 42, loss: 176.98, 6400 train 49.78%, 1600 test 46.44%
ep 43, loss: 177.09, 6400 train 49.78%, 1600 test 47.94%
ep 44, loss: 176.30, 6400 train 50.47%, 1600 test 48.81%
ep 45, loss: 176.41, 6400 train 50.38%, 1600 test 50.19%
ep 46, loss: 175.34, 6400 train 51.48%, 1600 test 49.12%
ep 47, loss: 175.64, 6400 train 51.47%, 1600 test 48.19%
ep 48, loss: 177.09, 6400 train 49.92%, 1600 test 49.00%
ep 49, loss: 175.82, 6400 train 51.30%, 1600 test 48.12%
ep 50, loss: 176.86, 6400 train 50.02%, 1600 test 51.69%
[[135.   1.   0.  11.   3.   0.   9.  31.]
 [  4.  61.   6.   8.  47.   7.  16.  41.]
 [ 21.  12.  64.  27.  49.  19.  13.   9.]
 [ 34.   3.   5. 108.  15.  15.  11.   9.]
 [  2.  33.  27.  14. 107.   8.   2.   8.]
 [  3.   8.   7.  19.   6. 144.  22.   4.]
 [ 38.  12.   2.  16.   4.  15.  85.  20.]
 [ 37.  15.   1.   2.  12.   1.   9. 123.]]
   Model saved to checkModel.pth
ep 51, loss: 173.95, 6400 train 52.98%, 1600 test 49.19%
ep 52, loss: 173.62, 6400 train 53.33%, 1600 test 48.81%
ep 53, loss: 174.08, 6400 train 52.89%, 1600 test 51.12%
ep 54, loss: 174.47, 6400 train 52.45%, 1600 test 50.56%
ep 55, loss: 173.94, 6400 train 53.12%, 1600 test 50.44%
ep 56, loss: 173.74, 6400 train 53.03%, 1600 test 51.62%
ep 57, loss: 172.74, 6400 train 54.19%, 1600 test 50.88%
ep 58, loss: 173.28, 6400 train 53.70%, 1600 test 51.31%
ep 59, loss: 172.11, 6400 train 55.05%, 1600 test 49.31%
ep 60, loss: 173.09, 6400 train 53.89%, 1600 test 52.94%
[[102.   3.   0.  11.   3.   0.  23.  48.]
 [  2.  69.   3.   7.  27.  12.  34.  36.]
 [ 15.  21.  56.  22.  44.  20.  23.  13.]
 [ 23.   3.   4. 117.   9.  12.  22.  10.]
 [  1.  38.  19.  12. 105.   4.  10.  12.]
 [  3.   6.   5.  15.   6. 146.  30.   2.]
 [ 21.   9.   2.   8.   2.  15. 115.  20.]
 [ 17.  11.   1.   2.   9.   1.  22. 137.]]
   Model saved to checkModel.pth
ep 61, loss: 171.79, 6400 train 55.25%, 1600 test 53.75%
ep 62, loss: 172.28, 6400 train 54.94%, 1600 test 52.88%
ep 63, loss: 173.78, 6400 train 53.19%, 1600 test 48.75%
ep 64, loss: 172.47, 6400 train 54.61%, 1600 test 51.06%
ep 65, loss: 172.37, 6400 train 54.34%, 1600 test 52.19%
ep 66, loss: 172.29, 6400 train 54.81%, 1600 test 51.25%
ep 67, loss: 170.10, 6400 train 57.11%, 1600 test 52.69%
ep 68, loss: 171.13, 6400 train 56.03%, 1600 test 51.25%
ep 69, loss: 170.40, 6400 train 56.42%, 1600 test 53.56%
ep 70, loss: 171.45, 6400 train 55.27%, 1600 test 53.69%
[[139.   2.   0.   8.   1.   0.  13.  27.]
 [  4.  73.   3.   7.  33.   9.  34.  27.]
 [ 22.  17.  41.  21.  51.  29.  23.  10.]
 [ 36.   2.   4. 106.   6.  17.  23.   6.]
 [  1.  36.  16.  16. 111.   6.   9.   6.]
 [  2.   6.   4.  11.   7. 153.  28.   2.]
 [ 30.   8.   1.   2.   3.  11. 121.  16.]
 [ 33.  16.   1.   0.  12.   2.  21. 115.]]
   Model saved to checkModel.pth
ep 71, loss: 169.26, 6400 train 57.86%, 1600 test 50.75%
ep 72, loss: 169.54, 6400 train 57.75%, 1600 test 51.50%
ep 73, loss: 170.48, 6400 train 56.48%, 1600 test 55.81%
ep 74, loss: 169.32, 6400 train 57.59%, 1600 test 54.12%
ep 75, loss: 169.38, 6400 train 57.75%, 1600 test 55.56%
ep 76, loss: 168.73, 6400 train 58.17%, 1600 test 55.75%
ep 77, loss: 168.92, 6400 train 58.19%, 1600 test 54.81%
ep 78, loss: 167.80, 6400 train 59.52%, 1600 test 54.19%
ep 79, loss: 167.07, 6400 train 60.16%, 1600 test 54.00%
ep 80, loss: 167.99, 6400 train 59.06%, 1600 test 53.25%
[[114.   2.   0.   3.   1.   0.  26.  44.]
 [  3.  85.   2.   6.  17.   9.  36.  32.]
 [ 19.  18.  55.  17.  40.  18.  31.  16.]
 [ 37.   5.   2.  82.   9.   8.  39.  18.]
 [  1.  49.  14.  11.  93.   4.  15.  14.]
 [  3.   7.   5.   7.   3. 140.  45.   3.]
 [ 13.  10.   1.   1.   0.   5. 143.  19.]
 [ 16.  14.   3.   1.   4.   2.  20. 140.]]
   Model saved to checkModel.pth
ep 81, loss: 169.35, 6400 train 57.73%, 1600 test 54.37%
ep 82, loss: 168.47, 6400 train 58.63%, 1600 test 51.69%
ep 83, loss: 168.10, 6400 train 59.03%, 1600 test 53.31%
ep 84, loss: 168.22, 6400 train 58.64%, 1600 test 55.62%
ep 85, loss: 167.56, 6400 train 59.52%, 1600 test 55.94%
ep 86, loss: 168.43, 6400 train 58.59%, 1600 test 56.19%
ep 87, loss: 166.63, 6400 train 60.39%, 1600 test 51.06%
ep 88, loss: 168.01, 6400 train 59.05%, 1600 test 55.00%
ep 89, loss: 166.29, 6400 train 60.80%, 1600 test 56.38%
ep 90, loss: 165.31, 6400 train 61.67%, 1600 test 56.62%
[[141.   2.   0.   5.   4.   1.  19.  18.]
 [  3.  77.  11.   6.  24.  14.  25.  30.]
 [ 18.  17.  92.  23.  21.  24.  12.   7.]
 [ 25.   3.  16. 102.  11.  21.  15.   7.]
 [  1.  32.  34.  15. 100.   9.   3.   7.]
 [  2.   6.  10.  10.   5. 164.  14.   2.]
 [ 25.   8.   6.  19.   2.  17. 107.   8.]
 [ 27.  13.   6.   3.   5.   1.  22. 123.]]
   Model saved to checkModel.pth
```
## result with softmax (lr=0.0015)

```shell
Using device: cuda:0

Start training...
ep 1, loss: 203.27, 6400 train 20.12%, 1600 test 28.25%
ep 2, loss: 199.57, 6400 train 25.16%, 1600 test 30.63%
ep 3, loss: 197.51, 6400 train 27.38%, 1600 test 29.50%
ep 4, loss: 195.94, 6400 train 28.97%, 1600 test 35.88%
ep 5, loss: 195.08, 6400 train 30.72%, 1600 test 33.00%
ep 6, loss: 192.96, 6400 train 32.58%, 1600 test 34.00%
ep 7, loss: 194.02, 6400 train 31.53%, 1600 test 33.06%
ep 8, loss: 192.57, 6400 train 33.30%, 1600 test 37.19%
ep 9, loss: 191.81, 6400 train 34.16%, 1600 test 39.94%
ep 10, loss: 190.92, 6400 train 35.31%, 1600 test 35.75%
[[139.   0.   0.  19.   4.   1.   1.  42.]
 [ 12.   9.  42.   8.  87.   9.   1.  63.]
 [ 15.   2. 100.   9.  23.   4.   1.  14.]
 [ 28.   1.  35.  77.  27.  11.   2.  15.]
 [  6.   4. 108.  17.  62.   1.   0.  14.]
 [ 15.   3.  47.  37.  36.  38.   5.   8.]
 [ 69.   8.   8.  26.  21.  10.  14.  45.]
 [ 39.   1.   6.   4.  14.   0.   0. 133.]]
   Model saved to checkModel.pth
ep 11, loss: 191.44, 6400 train 34.52%, 1600 test 39.00%
ep 12, loss: 189.53, 6400 train 37.03%, 1600 test 41.50%
ep 13, loss: 189.04, 6400 train 37.34%, 1600 test 43.12%
ep 14, loss: 189.23, 6400 train 37.00%, 1600 test 41.38%
ep 15, loss: 187.71, 6400 train 38.61%, 1600 test 39.25%
ep 16, loss: 187.13, 6400 train 39.48%, 1600 test 44.31%
ep 17, loss: 186.28, 6400 train 40.00%, 1600 test 42.94%
ep 18, loss: 185.91, 6400 train 40.88%, 1600 test 43.94%
ep 19, loss: 186.31, 6400 train 40.27%, 1600 test 45.94%
ep 20, loss: 185.29, 6400 train 40.88%, 1600 test 47.62%
[[123.   2.   0.  11.   2.   1.  33.  34.]
 [  4.  91.   5.   5.  52.  18.  28.  28.]
 [ 12.  10.  40.  16.  57.  17.   8.   8.]
 [ 13.  10.   6.  95.  20.  15.  21.  16.]
 [  0.  45.  28.  12. 104.   6.  10.   7.]
 [  6.  17.   6.  20.  13. 114.  11.   2.]
 [ 21.  21.   0.  28.   6.  30.  81.  14.]
 [ 28.  21.   0.   3.  14.   0.  17. 114.]]
   Model saved to checkModel.pth
ep 21, loss: 183.37, 6400 train 43.38%, 1600 test 45.50%
ep 22, loss: 184.25, 6400 train 42.33%, 1600 test 45.06%
ep 23, loss: 184.01, 6400 train 42.73%, 1600 test 46.06%
ep 24, loss: 184.77, 6400 train 41.64%, 1600 test 48.38%
ep 25, loss: 183.06, 6400 train 43.50%, 1600 test 47.50%
ep 26, loss: 182.84, 6400 train 43.53%, 1600 test 45.31%
ep 27, loss: 182.71, 6400 train 44.06%, 1600 test 48.00%
ep 28, loss: 181.36, 6400 train 45.22%, 1600 test 51.25%
ep 29, loss: 180.72, 6400 train 46.22%, 1600 test 46.00%
ep 30, loss: 180.90, 6400 train 45.69%, 1600 test 43.69%
[[125.   0.   0.   9.   3.   0.  18.  51.]
 [  1.  74.  10.   5.  53.   3.  14.  71.]
 [ 12.   9.  39.  19.  53.   5.  12.  19.]
 [ 24.  12.   5.  64.  42.   3.  27.  19.]
 [  3.  27.  36.  10. 115.   0.   5.  16.]
 [  6.  24.   8.  20.  13.  53.  54.  11.]
 [ 26.   8.   0.  11.  16.   2.  83.  55.]
 [ 20.   9.   0.   2.  14.   0.   6. 146.]]
   Model saved to checkModel.pth
ep 31, loss: 181.59, 6400 train 45.14%, 1600 test 47.88%
ep 32, loss: 182.73, 6400 train 44.20%, 1600 test 48.00%
ep 33, loss: 182.48, 6400 train 44.17%, 1600 test 48.31%
ep 34, loss: 180.01, 6400 train 46.89%, 1600 test 46.12%
ep 35, loss: 181.48, 6400 train 45.45%, 1600 test 46.69%
ep 36, loss: 179.76, 6400 train 46.88%, 1600 test 49.00%
ep 37, loss: 180.82, 6400 train 45.98%, 1600 test 48.31%
ep 38, loss: 181.07, 6400 train 45.72%, 1600 test 47.19%
ep 39, loss: 179.38, 6400 train 47.50%, 1600 test 46.00%
ep 40, loss: 179.50, 6400 train 47.25%, 1600 test 50.50%
[[149.   1.   0.   9.   1.   3.  10.  33.]
 [  2.  72.  13.   7.  34.  39.   8.  56.]
 [ 16.   6.  48.  10.  40.  33.   4.  11.]
 [ 22.   7.   7.  86.  10.  46.   8.  10.]
 [  1.  27.  39.  11. 101.  13.   5.  15.]
 [  3.   7.   8.   7.   9. 147.   6.   2.]
 [ 25.   6.   0.  18.   4.  58.  62.  28.]
 [ 28.   9.   1.   5.   7.   2.   2. 143.]]
   Model saved to checkModel.pth
ep 41, loss: 178.97, 6400 train 47.94%, 1600 test 51.25%
ep 42, loss: 178.86, 6400 train 48.02%, 1600 test 44.81%
ep 43, loss: 178.36, 6400 train 48.41%, 1600 test 52.38%
ep 44, loss: 178.77, 6400 train 48.25%, 1600 test 50.38%
ep 45, loss: 178.49, 6400 train 48.36%, 1600 test 49.81%
ep 46, loss: 179.07, 6400 train 47.73%, 1600 test 50.62%
ep 47, loss: 178.62, 6400 train 48.27%, 1600 test 50.94%
ep 48, loss: 178.74, 6400 train 48.14%, 1600 test 43.38%
ep 49, loss: 179.43, 6400 train 47.55%, 1600 test 49.88%
ep 50, loss: 176.73, 6400 train 50.25%, 1600 test 48.00%
[[126.   3.   1.  15.   1.   5.  23.  32.]
 [  0. 108.  17.   8.  23.  38.  13.  24.]
 [  9.  15.  59.   7.  29.  35.   6.   8.]
 [ 10.  22.   8.  71.  11.  64.   6.   4.]
 [  0.  60.  60.   7.  64.  12.   1.   8.]
 [  1.  11.   8.   8.   3. 154.   3.   1.]
 [ 10.  17.   1.  20.   3.  75.  64.  11.]
 [ 19.  32.   3.  10.   1.   1.   9. 122.]]
   Model saved to checkModel.pth
ep 51, loss: 183.36, 6400 train 43.34%, 1600 test 46.62%
ep 52, loss: 178.67, 6400 train 48.31%, 1600 test 48.75%
ep 53, loss: 178.68, 6400 train 48.39%, 1600 test 50.25%
ep 54, loss: 179.11, 6400 train 47.92%, 1600 test 50.88%
ep 55, loss: 177.67, 6400 train 49.33%, 1600 test 48.75%
ep 56, loss: 177.42, 6400 train 49.45%, 1600 test 40.62%
ep 57, loss: 178.99, 6400 train 47.92%, 1600 test 50.69%
ep 58, loss: 178.54, 6400 train 48.52%, 1600 test 49.25%
ep 59, loss: 178.25, 6400 train 48.64%, 1600 test 50.81%
ep 60, loss: 177.64, 6400 train 49.33%, 1600 test 44.19%
[[151.   2.   0.   9.   1.   8.  18.  17.]
 [  2.  69.  12.   9.  22.  82.  24.  11.]
 [ 16.   8.  49.   6.  20.  57.   8.   4.]
 [ 18.   4.   5.  55.   4.  89.  18.   3.]
 [  1.  34.  46.  18.  58.  40.  11.   4.]
 [  1.   3.   4.   4.   2. 168.   6.   1.]
 [ 23.   4.   0.   6.   2.  90.  74.   2.]
 [ 44.  27.   0.   6.   5.  10.  22.  83.]]
   Model saved to checkModel.pth
ep 61, loss: 177.86, 6400 train 49.20%, 1600 test 48.06%
ep 62, loss: 177.64, 6400 train 49.56%, 1600 test 50.69%
ep 63, loss: 179.20, 6400 train 47.86%, 1600 test 52.75%
ep 64, loss: 178.35, 6400 train 48.73%, 1600 test 47.19%
ep 65, loss: 176.97, 6400 train 50.14%, 1600 test 52.81%
ep 66, loss: 176.13, 6400 train 50.92%, 1600 test 51.50%
ep 67, loss: 177.28, 6400 train 49.70%, 1600 test 53.87%
ep 68, loss: 178.16, 6400 train 48.95%, 1600 test 49.31%
ep 69, loss: 178.03, 6400 train 49.33%, 1600 test 48.88%
ep 70, loss: 178.53, 6400 train 48.62%, 1600 test 48.50%
[[136.   1.   0.  22.   3.   5.  25.  14.]
 [  2.  84.  16.   3.  50.  36.  25.  15.]
 [ 13.  10.  49.  10.  57.  20.   6.   3.]
 [ 10.  11.  22.  85.  22.  35.   8.   3.]
 [  1.  19.  47.   9. 116.  11.   7.   2.]
 [  1.   9.  14.   9.  11. 136.   7.   2.]
 [ 15.  10.   0.  18.  11.  52.  92.   3.]
 [ 26.  23.   2.  11.  20.   2.  35.  78.]]
   Model saved to checkModel.pth
ep 71, loss: 181.65, 6400 train 45.38%, 1600 test 46.88%
ep 72, loss: 176.31, 6400 train 50.92%, 1600 test 52.50%
ep 73, loss: 180.46, 6400 train 46.80%, 1600 test 48.31%
ep 74, loss: 179.58, 6400 train 47.64%, 1600 test 47.50%
ep 75, loss: 177.18, 6400 train 49.95%, 1600 test 50.50%
```

## result with log_softmax (lr=0.0008)

best accuracy: 59%, then overfits
