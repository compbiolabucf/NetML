# Network-based transfer learning models for biomarker selection and cancer outcome prediction


## Softerware requirement
- Python 3.6 and up
- required packages:
- - Numpy
- - Pandas
- - Pickle
- - scikit-learn
- - scipy
- - random
- - warnings

## How to run the project
Download eight .pk files(sample_breast_expression.pk, sample_breast_label.pk, sample_ov_expression.pk, sample_ov_label.pk, sample_PRAD.pkl, sample_PRAD_label.pkl, sample_ppi.pkl, sample_gene_names.pkl) and two .py files(NetTL_three_domains.py, NetSTL_three_domains.py) into the same fold. Run python command and execute:
`python3 NetTL(NetSTL)_three_domains.py` 
The results will be exported as txt files in the same fold. 

## File description 
* **sample_OV.pkl:** : **Ovarian Cancer** gene expression data set. The size of the data is 100 x 500, 100 samples and 500 genes. 

* **sample_OV_label.pkl:** : **Ovarian Cancer** label data. The size of it is 100.

* **sample_BRCA.pkl:** This is a sample file of **Breast Cancer** gene expression set. The size of the data is 100 x 500, 100 samples and 200 genes. 

* **sample_BRCA_label.pkl:** : **Breast Cancer** label data. The size of it is 100.

* **sample_PRAD.pkl:** This is a sample file of **Prostate adenocarcinoma Cancer** gene expression set. The size of the data is 100 x 500, 100 samples and 200 genes. 

* **sample_PRAD_label.pkl:** : **Prostate adenocarcinoma Cancer** label data. The size of it is 100.

* **sample_gene_names.pkl**: 500 gene names.

* **sample_ppi.pkl**: **Protein-protein interaction networks**, dict file with 7932 keys.

* **NetTL(NetSTL)_three_domains.py:** execution files.












