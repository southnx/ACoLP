# Open Set Video HOI detection from Action-centric Chain-of-Look Prompting.


Code for the paper *Open Set Video HOI detection from Action-centric Chain-of-Look Prompting.* 

[Paper Link]: (https://openaccess.thecvf.com/content/ICCV2023/papers/Xi_Open_Set_Video_HOI_detection_from_Action-Centric_Chain-of-Look_Prompting_ICCV_2023_paper.pdf)

Nan Xi, Jingjing Meng, Junsong Yuan, ICCV 2023.

Human-Object Interaction (HOI) detection is essential for understanding and modeling real-world events. Existing works on HOI detection mainly focus on static images and a closed setting, where all HOI classes are provided in the training set. In comparison, detecting HOIs in videos in open set scenarios is more challenging. First, under open set circumstances, HOI detectors are expected to hold strong generalizability to recognize unseen HOIs not included in the training data. Second, accurately capturing temporal contextual information from videos is difficult, but it is crucial for detecting temporal-related actions such as open, close, pull, push. To this end, we propose ACoLP, a model of Action-centric Chain-of-Look Prompting for open set video HOI detection. ACoLP regards actions as the carrier of semantics in videos, which captures the essential semantic information across frames. To make the model generalizable on unseen classes, inspired by the chain-of-thought prompting in natural language processing, we introduce the chain-of-look prompting scheme that decomposes prompt generation from large-scale vision-language model into a series of intermediate visual reasoning steps. Consequently, our model captures complex visual reasoning processes underlying the HOI events in videos, providing essential guidance for detecting unseen classes. Extensive experiments on two video HOI datasets, VidHOI and CAD120, demonstrate that ACoLP achieves competitive performance compared with the state-of-the-art methods in the conventional closed setting, and outperforms existing methods by a large margin in the open set setting.

