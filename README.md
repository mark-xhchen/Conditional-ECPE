# Conditional-ECPE
This is a repository for our 2020 EMNLP paper "Conditional Causal Relationships between Emotions and Causes in Texts" \[[pdf](https://www.aclweb.org/anthology/2020.emnlp-main.252.pdf)\], which contains the manually-labeled dataset and the code of our proposed module.

Note that our dataset is constructed based on an existing ECPE corpus. If you are interested in the original ECPE datset, please refer to: https://github.com/NUSTM/ECPE

If you use our dataset or code, please cite our paper:
>Xinhong Chen, Qing Li, Jianping Wang. Conditional Causal Relationships between Emotions and Causes in Texts. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3111-3121.

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

Note that if you directly download the repo zip file from the github site, **the downloaded "w2v_200.txt" in directory "nega_data" may not be the correct file.** Please:
- open the "w2v_200.txt" file in github;
- right click on the website;
- choose "save as" to download the correct file, which should be around **80Mb**. 

If you are cloning the whole repo, the above issue should not be a problem.

To run a program:
- Make sure you complete the dataset construction first
-	Directly run “python programname.py”, where the “programename” is the python file you want to run.

Should you have any problem, contact xinhong.chen@my.cityu.edu.hk
