# Real - Time Facemask Detection and Analytics
This Project focuses on detecting if a person is wearing a facemask or not (or wearing one incorrectly).

•	Inspired by the ongoing pandemic, a prototype system was built to identify if a person is wearing a face mask properly or not.

•	Identified people’s faces by applying a deep-learning technique in python called Single Shot Detector and scanned their faces for a face mask using another deep learning technique called MobileNet.

•	Stored the different types of data obtained to visualize them for insights such as the number of people who wore a face mask on a certain day.

•	Developed further into a research paper by adding person recognition using QR Codes and improved visualizations by using Tableau which was then presented at the 2021 ACMI 4.0 International Conference and published on IEEE Xplore. 

• Check out the paper here: https://ieeexplore.ieee.org/document/9528130

facemask_model_final.py is concerned with training the MobileNet model. 
mask_and_qr_detector.py focuses on the detection of the person and gathering the data.

Sources of Data (Images) used: 

• The dataset is collected from different sources such as Kaggle and Google images. Images of people wearing and not wearing masks, were collected from the Kaggle Dataset titled “Face Mask Detection Data” by Aneerban Chakraborty https://www.kaggle.com/aneerbanchakraborty/face-mask-detection-data. 

• For images of people wearing masks incorrectly, a majority of them were collected from the dataset created by Adnane Cabani et al. https://pubmed.ncbi.nlm.nih.gov/33521223/ .
Some were collected from the Kaggle Dataset named “Face Mask Detector Data” by Spandan Patnaik https://www.kaggle.com/spandanpatnaik09/face-mask-detectormask-not-mask-incorrect-mask. The remaining images were scraped from Google Images.
