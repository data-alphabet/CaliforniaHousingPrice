# California Housing Price Data Analysis Demo

## Project Description
This is a data analysis demo to learn the process of data analysis with **Python**. In this project, I experienced *data fetching*, *exploratory data analysis*, *feature engeneering*, *data preprocessing*, *data visulization*, *data modeling* and so on. It is really an iteresting learning experience! All the source codes own detailed notes, so you can read them easily!

- 地理信息可视化.xlsx

I try to visulize the geographic information by the Excel plug-in —— **PowerMap**! And I transform it into a short cute vedio file —— 地理信息可视化.mp4.

- DownloadData.py

It is a module to fetch dataset's raw file from https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz . This module mainly solves problems in data fetching and storing. It used Python's **os**, **tarfile** and **six.moves.urlib**!

- EDA.py

It is the module for exploratory data analysis. I can be familiar with the data as soon as quickly by knowing the distribution of data(statistical description & histogram plot), the missing values of attributes and the data sample of dataset! It used API in Python's **pandas**, **matplotlib.pyplot** and os!

- FeatureEngineering.py

This module is designed to exploring how to choose suitable features or create new appropriate features for modeling. Calculating the correlation coefficient and visulizing the CorrelationMatrix by heatmap help us to decide which features are used and which features need transforming. It used Python's pandas, matplotlib.pyplot, **seaborn**, os and **numpy**!

- SplitDataset.py

It defines several dataset splitting methods such as *Randomly Sampling*, *Indentifier Sampling*, *Stratified Random Sampling*. Splitting methods are realized by making designs of my own or invoking the API's of **sklearn.model_selection**. It used the **hashlib.md5** to translate index into hash value less than 256.

- DataProcessing.py

It use the **FeatureUnion** and **Pipeline** to combine the preprocessing together to form pipeline workflow. It used **SimpleImputer**, **LabelBinarizer**, **StandardScaler** from sklearn.

- CaliforniaHousingPrice.py

It is the main module! It creates three types of regression models —— *LinearRegression*, *DecisionTreeRegression*, *RandomForestRegression*. It fits the data and labels in trainset using the models and predicts the value by data in testset. It evaluated the models by *root of mean squared error(rmse)* in cross validation.
