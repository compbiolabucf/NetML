# Network-based transfer learning models for biomarker selection and cancer outcome prediction


## Softerware requirement
- Python 3.6 and up
- required packages:
- - Numpy
- - Pandas
- - Pickle
- - scikit-learn

## How to run the project
Download four .pk files(sample_breast_expression.pk, sample_breast_label.pk, sample_ov_expression.pk, sample_ov_label.pk) and two .py files(NetTL.py, NetSTL.py) into the same fold. Run python command and execute:
`python3 NetTL(NetSTL).py sample_breast_expression.pk sample_breast_label.pk sample_ov_expression.pk sample_ov_label.pk` 
The results will be exported as two txt files in the same fold. 

## File description 
* **sample_ov_expression.pk:** : **Ovarian Cancer** gene expression data set. The size of the data is 50 x 200, 50 samples and 200 genes. The value range is [0,20.4273].

* **sample_ov_label.pk:** : **Ovarian Cancer** label data. The size of it is 50.

* **sample_breast_expression.pk:** This is a sample file of **Breast Cancer** gene expression set. The size of the data is 50 x 200, 50 samples and 200 genes. The value range is [0,20.3229].

* **sample_breast_label.pk:** : **Breast Cancer** label data. The size of it is 50.

* **NetTL.py and NetSTL.py:** execution files.

## Data clean 
The genes with low mean and standard deviation are removed. The initial threshold is 50%, and it can be changed in the read_data function of the NetTL.py and NetSTL.py files.

## Cross-validation
All models will run though the same cross validation process together. For both data sets, 20% of samples are randomly selected for test, 20% for validation, the rest samples are for training. This process will repeat 50 times, and the average performance on test set are reported. 










