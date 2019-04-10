# Pseudo-label-behavior-recognition

This project uses semi-supervised learning, conducts research in human behavior analysis, and analyzes neural networks based on MFNet([Project address](https://github.com/cypw/PyTorch-MFNet)) construction behavior, so that the effect of the classifier is effectively improved.



## Video Recognition (HMDB51)

| Labeling ratio | Params | Unselected pseudo label | Selected pseudo label |
| :------------: | :----: | :---------------------: | :-------------------: |
|       5%       | 24.37% |         25.22%          |        26.73%         |
|      15%       | 56.92% |         60.19%          |        61.63%         |
|      30%       | 63.92% |         66.47%          |        67.01%         |
|      100%      | 71.96% |                         |                       |

\* accuracy averaged on hmdb51 split1.

![Train loss](https://github.com/y1500730136/Pseudo-label-behavior-recognition/blob/yxy/image/4_1.png)