# UCAS-NLP2020-Event-Extraction
UCAS2020《自然语言处理》编程作业-事件抽取

项目成员：

https://github.com/luanjie24

https://github.com/HuangXuerong-ucas

https://github.com/Gwzlchn

https://github.com/Shirley123121

https://github.com/cpppx


## 项目目录说明

```
初步框架，后面可能会不断完善和修改

│
├── dataset                                             # 数据集文件夹
│   ├── CEC-Corpus
│   └── xf_2020_Corpus 
│
├── pretrained_model                                    # 预训练模型文件夹，尝试的预训练模型都放在这里
│   └── chinese_roberta_wwm_ext_pytorch
│
├── nlp_web                                             # web模块，用于作业展示，网址就不公开了
│   
├── saved_model                                         # 用于存放训练好的模型和日志
│   
├── src                                                 # 代码文件夹
│   ├── config.py                                       # 配置定义
│   ├── convert_raw_data.py                             # 转换数据的格式 
│   ├── data_converter.py                               # 没用到
│   ├── dataset_utils.py                                # 处理数据
│   ├── evaluator.py                                    # 评估模型
│   ├── load_data.py                                    # 数据预处理
│   ├── main.py                                         # 集成训练模块，便于进行模型训练
│   ├── model.py                                        # 模型定义
│   ├── predict.py                                      # 模型批量提取文本事件
│   ├── processor.py                                    # 数据格式封装，转换为特征向量
│   ├── test.py                                         # 模型提取单个事件
│   └── train.py                                        # 训练
│
└── README.md

```

## 项目说明

建立本项目的目的是为了学习，同时也是为了编程时方便分工

本项目为基于pytorch的pipeline+句子级+中文的事件抽取，其中预训练模型用的是pytorch版的中文RoBERTa-wwm-ext模型

特别感谢以下工作在我们完成作业的过程中给予的参考和帮助：

https://github.com/ymcui/Chinese-BERT-wwm

https://github.com/WuHuRestaurant/xf_event_extraction2020Top1

https://github.com/xiaoqian19940510/Event-Extraction

https://github.com/dair-ai/ml-visuals

感谢清华大学刘洋教授开设的《自然语言处理》课程，这是我们完成本项目的契机。

## 注意事项
有些路径需要自己创建，预训练模型需要自己下载

## 改进方向
目前是单GPU训练，可以改成多GPU

目前触发词提取器和主体客体提取器能达到良好的效果，时间地点提取器模型设计已完成，但有待进一步进行训练验证。

此外，触发词的识别目前只支持单触发词识别，当输入的句子触发词个数等于0或者大于1时，精度较差或者识别不出来，因此这也是之后改进的方向。


