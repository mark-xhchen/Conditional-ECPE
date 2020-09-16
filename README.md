# Conditional-ECPE
This is a repository for our 2020 EMNLP paper "Conditional Causal Relationships between Emotions and Causes in Texts", which contains our final paper, manually-labeled dataset, and the code of our proposed module.

Note that our dataset is constructed based on an existing ECPE corpus. If you are interested in the original ECPE datset, please refer to: https://github.com/NUSTM/ECPE

Hardware Environment
- Windows 10
- 1 GPU, Nvidia Geforce RTX 2080 Ti

Dependency Requirement
- Python 3.6
- Tensorflow 1.14.0
- sklearn, numpy, scipy

Dataset Construction Steps
- Run the “preprocess.cy” to get the manually labeled dataset, which will be stored in a file called “data.txt”
- Run the “gen_nega_samples.py” to generate the constructed conditional-ECPE dataset, which is stored in a file called “data_wneg.txt”
-	If you prefer our training/testing split, please run “divide_fold.py” to get 20 files, which will be named as “foldx_train.txt” and “foldx_test.txt”, where “x” should be from 1 to 10.

Run a program:
- Make sure you complete the dataset construction first
-	Directly run “python programname.py”, where the “programename” is the python file you want to run.

Should you have any problem, contact xinhchen2-c@my.cityu.edu.hk
