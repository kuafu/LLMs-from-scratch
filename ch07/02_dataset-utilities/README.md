# 第 7 章：指令遵循微调

此目录包含用于准备指令数据集的实用代码。

通过以下命令安装额外的软件包依赖项：

```bash
pip install -r requirements-extra.txt
```





### 查找近似重复项

`find-near-duplicates.py` 函数可用于识别指令数据集中的重复项和近似重复项。例如：



```bash
python find-near-duplicates.py --json_file instruction-examples.json
```

```
scikit-learn version: 1.3.1


==================================================
Searching 'instruction' for duplicates ...
==================================================
Duplicate pair found with similarity 0.94:
1. Edit the following sentence to make it more formal.
2. Edit the sentence to make it more formal.

Duplicate pair found with similarity 1.00:
1. Name a dwarf planet in our solar system.
2. Name a dwarf planet in our solar system.

Duplicate pair found with similarity 0.91:
1. Change the sentences from active voice to passive voice.
2. Change the sentence from passive to active voice.



==================================================
Searching 'input' for duplicates ...
==================================================
No duplicates found


==================================================
Searching 'output' for duplicates ...
==================================================
Duplicate pair found with similarity 1.00:
1. One dwarf planet in our solar system is Pluto.
2. One dwarf planet in our solar system is Pluto.


```

&nbsp;
你可以将 `--threshold` 设置为 0 到 1 之间的值，以降低或提高检测灵敏度。
默认阈值为 0.9。



&nbsp;
 ## 创建被动语态样本

 - [create-passive-voice-entries.ipynb](create-passive-voice-entries.ipynb) 笔记本使用 OpenAI 的 GPT-4 为指令数据集创建“被动语态”样本，如下例所示

 ```python
 {  
    'instruction': 'Identify the verb in the following sentence',
    'input': 'The cat sleeps on the couch.',
    'output': 'The verb in the sentence is "sleeps."',
    'output_2': 'The sentence is "sleeps."'   #  <---- Newly created entry
 }  
 ```
