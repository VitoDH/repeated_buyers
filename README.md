# Predicting the potential repeated buyers
#### **Author**: Dehai Liu

**Department of Mathematics, Sun Yat-Sen University**



## 1. Abstract

Our project mainly focus on digging out the potential online repeated-buyers with data-driven methods and state-of-the-art algorithms. It can not only provide guidance for the sellers about their operation,but also help the online shopping platform to pinpoint the target customers for advertisements and coupons.



## 2. Data Description

`T-mall` is one of the largest online shopping platform in China.  On November 11th, "the Single Day", the trading volume on T-mall reaches 120 billion CNY, indicating tremendous profit behind this shopping festival. In this project, our data includes the shopping records of the buyers for six months before the and on the Single Day. The data is divided into **3** parts as follows:



**User Log**

| Attributes  |                      Definitions                      |
| :---------: | :---------------------------------------------------: |
|   user_id   |                  Unique ID for user                   |
|   item_id   |                Unique ID for commodity                |
|   cat_id    |        Unique ID for the category of the good         |
| merchant_id |              Unique ID for the merchant               |
|  brand_id   |                Unique ID for the brand                |
| time_stamp  |         The date of action given by customer          |
| action_type | 0:click, 1: add to cart, 2: order, 3: add to favorite |



**User Info**

| Attributes |                         Definitions                          |
| :--------: | :----------------------------------------------------------: |
|  user_id   |                      Unique ID for user                      |
| age_range  | 1: < 18, 2: [18,24], 3: [25:29], 4: [30,34], 5: [35,39], 6: [40,49], 7: >=50, 0: unknown |
|   gender   |                 0: female, 1: male, 2: NULL                  |



**Training Set and Test Set**

| Attributes  |                Definitions                 |
| :---------: | :----------------------------------------: |
|   user_id   |             Unique ID for user             |
| merchant_id |           Unique ID for merchant           |
|    label    | 1: repeated buyers, 0: non-repeated buyers |





## 3. Data Preprocessing

### (a) Load the data

* Load use_log.csv and user_info.csv
* Remove the data points including NA in user_info
* Load train_format1.csv, pick a subset of 100,000 samples and denote it as trainSet



### (b) Outlier Detection









## Preprocessing

### a. Preprocessing for Decision Tree

#### (1)  Normalization

Supposed the data matrix is <img src="https://latex.codecogs.com/svg.latex?\Large&space;X" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />, we can normalize it to speed up training by dividing by 255. 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;X_{norm}=\frac{X}{255}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />

#### (2) Max Absolute Scale

To scale each feature to the<img src="https://latex.codecogs.com/svg.latex?\Large&space;[-1,1]" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />  range without breaking the sparsity of the images, we can use max absolute scaling,

<img src="https://latex.codecogs.com/svg.latex?\Large&space;X_{MAbs}=\frac{X_{norm}}{max(abs(X_{norm}),axis=0)}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />



#### (3) PCA

Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. Since there are 10​ classes in the dataset, we can first set the decomposition components to be ​10. After careful trial, 10​ tend to be a good choice.



### b. Preprocessing for LeNet

#### (1) Reshape 

Since convolution network requires the input to be 3D​ images, we need to reshape the image vector into image matrix with size of 28 x 28​ and one channel. Now the range of each component is [0,255]​ .

Dimension of data  before reshape: (40000,784)​

Dimension of data  after reshape: (40000,28,28,1)​

#### (2) Scaling

Using the function "transform.ToTensor" in Gluon, we can again reshape the data for the standard input in the framework and scale the data in the range [0,1)​ .

Dimension of data  before scaling: (40000,28,28,1)​

Dimension of data  after scaling: (40000,1,28,28)​



## Predictors Description

### a. Decision Tree

#### Parametrization

|    Parameters    |                       Description                       | Value  |
| :--------------: | :-----------------------------------------------------: | :----: |
|    max depth     |                maximum depth of the tree                | 10:100 |
| min samples leaf | minimum number of samples required to be at a leaf node |  1:5   |



### b. LeNet

#### (1) Model Description

<img src="https://github.com/VitoDH/stat535_project/raw/master/img/network_2.png" style="zoom:90%" />

<center><B>Fig. 2 </B>Model Architechture (by TensorSpace)</center>

The model is constructed as follows:

|    layer     | filters/channels | kernel size | pool size | strides | activation |
| :----------: | :--------------: | :---------: | :-------: | :-----: | :--------: |
|    input     |        1         |    None     |   None    |  None   |    None    |
|     Conv     |        20        |      5      |   None    |  None   |    Relu    |
|   MaxPool    |        20        |    None     |     2     |    2    |    None    |
|     Conv     |        50        |      5      |   None    |  None   |    Relu    |
|   MaxPool    |        50        |    None     |     2     |    2    |    None    |
| FullyConnect |       120        |    None     |   None    |  None   |    None    |
| FullyConnect |        84        |    None     |   None    |  None   |    None    |
|    output    |        10        |    None     |   None    |  None   |  Softmax   |

#### (2) Parametrization

|  Parameters   |                 Description                  | Value |
| :-----------: | :------------------------------------------: | :---: |
| learning rate |        control the speed for learning        | 0.001 |
|  num epochs   |               number of epochs               |  80   |
|  batch size   | number of  samples in one batch for training |  128  |
|    dropout    |      regularization term in the network      |  0.4  |





## Training Strategy

### a. Decision Tree

#### (1) Grid Search Cross Validation

- Parameters: minimum samples of leaf and maximum depth
- Training set size: 32000
- Validation set size: 8000
- **5 fold Cross Validation**: record the accuracy of training set and validation set
- Select the model with highest mean accuracy in the validation set

#### (2) Dataset Splitting and model demo

To demonstrate the model, we can again split the data into training set and validation set. At this time, the size of validation set can be much more smaller since it's used for demo instead of tuning the parameter.

- Training set size: 39600
- Validation set size: 400



### b. LeNet

#### (1) Xavier Initialization

According to the paper of Glorot & Bengio , we assume that for a specific layer $L$, the number of input  and output units are respectively, $n_{in}$ and $n_{out}$ .  

The requirement for the weight  layer L should be:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Var(W^L)=\frac{2}{n_{L}+n_{L+1}}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />

And the initialization of weights  follow the uniform distribution:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;W\sim U[-\sqrt{\frac{6}{n_L+n_{L+1}}},\sqrt{\frac{6}{n_L+n_{L+1}}}]" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />

#### (2)  Dataset Splitting

Split the data into training set and validation set and shuffle the training set for training.

- Training set size: 38000
- Validation set size: 2000

#### (3)  Network Architecture and dropout

Following the model description above,  we set up 

- 2 convolution layers with **Relu** activation followed by 2 max pooling layers respectively.
- 2 fully connected layers in the end,  add a **dropout** term serving as **regularization** 

#### (4) Training

Setting the training epochs to be $80$, we record the training loss, validation loss, training accuracy and validation accuracy. Pick the model in the last epoch and save its parameters.



## Experimental Results

### a. Decision Tree

#### (1) Tuning the parameters

The range of the hyper parameter is:

- max_depth: 10,20,30,40,50,60,70,80,90,100​
- min_samples_leaf: 1,2,3,4,5​



For a fixed min samples leaf, we can plot the accuracy vs max depth. Here we set min samples leaf=1.

<img src="https://github.com/VitoDH/stat535_project/raw/master/img/gridsearch_depth.png" style="zoom:40%" />

<center><B>Fig. 3 </B>GridSearchCV for max depth in Decision Tree</center>

- The accuracy of training set and validation set both goes up as maximum depth increases and become stable after max\_depth=40​ .



For a fixed max depth, we can plot the accuracy vs min sample leaf. Here we set max depth=None, which means that the tree will grow to the maximum depth.

<img src="https://github.com/VitoDH/stat535_project/raw/master/img/gridsearch_leaf.png" style="zoom:40%" />

<center><B>Fig. 4 </B>GridSearchCV for min sample leaf in Decision Tree</center>

- The accuracy of training set and validation set both goes down as min samples leaf increases.



From the two graphs above , we can conclude that the accuracy of validation(test) set is maximized when **max\_depth=40** and **min\_samples\_leaf=1​**. 

Actually, considering the combination of two parameters at the same time, we can still have the same conclusion that the best accuracy is **0.84**​. 



#### (2) Training and validation demo

From part (1), we can select the final model to be a decision tree with **max\_depth=40**​ and **min\_samples\_leaf=1**​ . Splitting the raw dataset into training set and validation set, we can fit the model again and obtain the following results.



**Estimation of classification error L**

| training error | validation error |
| :------------: | :--------------: |
|      0.0       |      0.0925      |



### b. LeNet

After training for $80$ epochs, we can plot the following learning curve including loss and accuracy of training set and validation set.

<img src="https://github.com/VitoDH/stat535_project/raw/master/img/learning_curve.png" style="zoom:60%" />

<center><B>Fig. 5 </B>Learning Curve</center>

- Training loss decreases as the number of epochs increases. However, the validation loss first decreases but goes up after $10$ epochs, which indicates minor overfitting.
- Both training and validation accuracy increase as the epochs increases and become steady after 40 epochs.



**Estimation of classification error L**

| training error | validation error |
| :------------: | :--------------: |
|     0.0049     |      0.056       |



------

## Reference

[1]Breiman L, Friedman J, Olshen R, etal. Classification and Regression Trees [M]. New York: Chapman
& Hall, 1984.

[2] Xavier Glorot, Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks [J]. PMLR, 2010.

[3]Y. LeCun, Y. Bengio. Convolutional networks for images, speech, and time-series [ J]. The Handbook of Brain Theory and Neural Networks. MIT Press, 1995.