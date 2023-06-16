# MACT
Our proposed MACT model is completely end-to-end, without any signal preprocessing, so no prior knowledge is required. We show the code for predicting bearing life with the PHM2012 data set (the same is true for the XJTU-SH data set).
Here, we introduce the implementation process of the code.

1. Download the Data set from the Internet and convert the data into PKZ category through data-PKZ.
2. Extract signals from training set and verification set through Train and Val_Dataset_Preparation
3. Extract the signal from the test set Test_Dataset_Preparation
4. Get prediction results through main and modelMACT
