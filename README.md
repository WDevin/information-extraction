# Information Extraction Baseline System—InfoExtractor
## Abstract
InfoExtractor is an information extraction baseline system based on the Schema constrained Knowledge Extraction dataset(SKED). InfoExtractor adopt a pipeline architecture with a p-classification model and a so-labeling model which are both implemented with PaddlePaddle. The p-classification model is a multi-label classification which employs a stacked Bi-LSTM with max-pooling network, to identify the predicate involved in the given sentence. Then a deep Bi-LSTM-CRF network is adopted with BIEO tagging scheme in the so-labeling model to label the element of subject and object mention, given the predicate which is distinguished in the p-classification model. The F1 value of InfoExtractor on the development set is 0.668.

InfoExtractor是一个基于Schema约束知识提取数据集（SKED）的信息提取基线系统。 InfoExtractor采用具有p分类模型和so-labeling模型的流水线架构，这些模型都使用PaddlePaddle实现。 p分类模型是多标签分类模型，其使用具有最大池网络的堆叠Bi-LSTM来识别给定句子中涉及的谓词predicate。 然后在so-labeling模型中采用BIEO标记方式的深Bi-LSTM-CRF网络，在p分类模型中区分谓词predicate的情况下，标记句子中predicate对应的subject和object实体。 InfoExtractor在开发集上的F1值为0.668。

## [2019语言与智能技术竞赛](http://lic2019.ccf.org.cn/kg)
本代码是2019语言与智能技术竞赛中信息抽取比赛相关的基线模型。更多信息参见2019语言与智能技术竞赛。

### 竞赛任务
给定schema约束集合及句子sent，其中schema定义了关系P以及其对应的主体S和客体O的类别，例如（S_TYPE:人物，P:妻子，O_TYPE:人物）、（S_TYPE:公司，P:创始人，O_TYPE:人物）等。 任务要求参评系统自动地对句子进行分析，输出句子中所有满足schema约束的SPO三元组知识Triples=[(S1, P1, O1), (S2, P2, O2)…]。
输入/输出:
(1) 输入:schema约束集合及句子sent
(2) 输出:句子sent中包含的符合给定schema约束的三元组知识Triples

**例子**
输入句子： ```"text": "《古世》是连载于云中书城的网络小说，作者是未弱"```

输出三元组： ```"spo_list": [{"predicate": "作者", "object_type": "人物", "subject_type": "图书作品", "object": "未弱", "subject": "古世"}, {"predicate": "连载网站", "object_type": "网站", "subject_type": "网络小说", "object": "云中书城", "subject": "古世"}]}```

### 数据简介
本次竞赛使用的SKE数据集是业界规模最大的基于schema的中文信息抽取数据集，其包含超过43万三元组数据、21万中文句子及50个已定义好的schema，表1中展示了SKE数据集中包含的50个schema及对应的例子。数据集中的句子来自百度百科和百度信息流文本。数据集划分为17万训练集，2万验证集和2万测试集。其中训练集和验证集用于训练，可供自由下载，测试集分为两个，测试集1供参赛者在平台上自主验证，测试集2在比赛结束前一周发布，不能在平台上自主验证，并将作为最终的评测排名。

## Getting Started
### Environment Requirements
+ python 3.6+
+ Paddlepaddle v1.2.0+

### Step 1: Install paddlepaddle
For now we’ve only tested on PaddlePaddle Fluid v1.2.0, please install PaddlePaddle firstly and see more details about PaddlePaddle in [PaddlePaddle Homepage](http://www.paddlepaddle.org/).

### Step 2: Download the training data, dev data and schema files
Please download the training data, development data and schema files from [the competition website](http://lic2019.ccf.org.cn/kg), then unzip files and put them in ```./data/``` folder.
```
cd data
unzip train_data.json.zip 
unzip dev_data.json.zip
cd -
```
### Step 3: Get the vocabulary file
Obtain high frequency words from the field ‘postag’ of training and dev data, then compose these high frequency words into a vocabulary list.
```
python lib/get_vocab.py ./data/train_data.json ./data/dev_data.json > ./dict/word_idx
```
### Step 4: Train p-classification model
First, the classification model is trained to identify predicates in sentences. Note that if you need to change the default hyper-parameters, e.g. hidden layer size or whether to use GPU for training (By default, CPU training is used), etc. Please modify the specific argument in ```./conf/IE_extraction.conf```, then run the following command:
```
python bin/p_classification/p_train.py --conf_path=./conf/IE_extraction.conf
```
The trained p-classification model will be saved in the folder ```./model/p_model```.
### Step 5: Train so-labeling model
After getting the predicates that exist in the sentence, a sequence labeling model is trained to identify the s-o pairs corresponding to the relation that appear in the sentence. </br>
Before training the so-labeling model, you need to prepare the training data that meets the training model format to train a so-labeling model.
```
python lib/get_spo_train.py  ./data/train_data.json > ./data/train_data.p
python lib/get_spo_train.py  ./data/dev_data.json > ./data/dev_data.p
```
To train a so labeling model, you can run:
```
python bin/so_labeling/spo_train.py --conf_path=./conf/IE_extraction.conf
```
The trained so-labeling model will be saved in the folder ```./model/spo_model```.

### Step 6: Infer with two trained models
After the training is completed, you can choose a trained model for prediction. The following command is used to predict with the last model. You can also use the development set to select the optimal model for prediction. To do inference by using two trained models with the demo test data (under ```./data/test_demo.json```), please execute the command in two steps:
```
python bin/p_classification/p_infer.py --conf_path=./conf/IE_extraction.conf --model_path=./model/p_model/final/ --predict_file=./data/test_demo.json > ./data/test_demo.p
python bin/so_labeling/spo_infer.py --conf_path=./conf/IE_extraction.conf --model_path=./model/spo_model/final/ --predict_file=./data/test_demo.p > ./data/test_demo.res
```
The predicted SPO triples will be saved in the folder ```./data/test_demo.res```.

## Evaluation
Precision, Recall and F1 score are used as the basic evaluation metrics to measure the performance of participating systems. After obtaining the predicted triples of the model, you can run the following command. 
Considering data security, we don't provide the alias dictionary.
```
zip -r ./data/test_demo.res.zip ./data/test_demo.res
python bin/evaluation/calc_pr.py --golden_file=./data/test_demo_spo.json --predict_file=./data/test_demo.res.zip
```

## Discussion
If you have any question, you can submit an issue in github and we will respond periodically. </br>


## Copyright and License
Copyright 2019 Baidu.com, Inc. All Rights Reserved </br>
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may otain a copy of the License at </br>
```http://www.apache.org/licenses/LICENSE-2.0``` </br>
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# APPENDIX
In the released dataset, the field ‘postag’ of sentences represents the segmentation and part-of-speech tagging information. The abbreviations of part-of-speech tagging (PosTag) and their corresponding part of speech meanings are shown in the following table. </br>
In addition, the given segmentation and part-of-speech tagging of the dataset are only references and can be replaced with other segmentation results.</br>

|POS| Meaning |
|:---|:---|
| n |common nouns|
| f | localizer |
| s | space |
| t | time|
| nr | noun of people|
| ns | noun of space|
| nt | noun of tuan|
| nw | noun of work|
| nz | other proper noun|
| v | verbs |
| vd | verb of adverbs|
| vn |verb of noun|
| a | adjective |
| ad | adjective of adverb|
| an | adnoun |
| d | adverbs |
| m | numeral |
| q | quantity|
| r | pronoun |
| p | prepositions |
| c | conjunction |
| u | auxiliary |
| xc | other function word |
| w | punctuations |
