# nlp
2020自然语言处理课程大作业-事件抽取


## 项目目录说明

```
初步框架，后面可能会不断完善和修改
nlp
├── dataset                                  # 数据集文件夹，尝试的数据集都放在这里
│   └── CEC-Corpus                           # CEC-Corpus，目前打算就用这个了,数据预处理后的格式也存在这里面吧
│       ├── CEC                              # CEC项目原始文件夹
│       ├── raw corpus                       # CEC项目原始文件夹
│       ├── CEC_clean                        # 自己加的文件夹，用于存放预处理后的数据
│       ├── CEC数据集标注结构说明图.jpg      # CEC项目标注内容的说明图
│       └── README.md                        # CEC项目的描述文件夹
│
├── pretrained_model                         # 预训练模型文件夹，尝试的预训练模型都放在这里
│   └── chinese_roberta_wwm_ext_pytorch      # pytorch版的中文RoBERTa-wwm-ext模型，目前打算用这个了
│
├── src# 代码文件夹，代码都放在这里
│   ├── config.py                            # 一些配置，比如一些读取路径
│   ├── data_converter.py                    # 数据预处理（转格式、加标注、转token_id） 
│   ├── model.py                             # 模型定义
│   ├── train.py                             # 训练
│   └── predict.py                           # 预测
│
├── reference                                # 一些参考资料，包括开源项目、论文
│ 
├── saved_model                              # 用于存放训练好的模型，一些中间权重也可先放在这里
│                         
└── README.md                                # ...
```

## 目前的一些问题
pytorch版的中文RoBERTa-wwm-ext模型有点大，上传不上来，可以自己下载一下

目前的想法是：pipeline+pytorch+句子级+封闭域+中文的事件抽取，不知道效果怎么样，如果有人有更好的方案可以提，实在不行演示的时候直接拿训练数据演示

如果用CEC数据集，有个问题是CEC数据集标的太细了，目前还没想明白怎么转换合适，可以先确定系统的输出格式再考虑怎么转换

模型结构可以看看reference，在此基础上针对CEC数据集改改结构（因为不同数据集标注的东西、事件的类型、Trigger的类型等等都不同），可以进一步讨论一下

代码很多地方还在学，还得看看其他人的代码，另外比如模型结构BERT这块的实现还得再看看，看懂后能抄就抄
