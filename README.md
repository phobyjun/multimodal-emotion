# (2023) 제 2회 휴먼이해 인공지능 논문경진대회
> 본 대회는 한국전자통신연구원(ETRI)이 주최하고 과학기술정보통신부와 국가과학기술연구회(NST)가 후원합니다
---
## Multi-model Emotion Recognition Model based on Temporal Graph Learning
---
## Abstract
> 최근 딥러닝 기술이 발전하면서, 인간-컴퓨터 상호작용(HCI) 분야의 연구가 활발히 진행되고 있다. 감정 인식 분야는 HI 연구 분야의 주요 과제 중 하나이다. 감정 인식을 통해 사용자는 더욱 현실감 있는 사용자 경험을 얻을 수 있다. 뛰어난 사용자 경험을 제공하기 위해서는 높은 정확도가 요구된다. 최근 딥러닝에서는 '멀티 모달' 기술을 통하여, 여러 모달리티에 있는 데이터를 융합하여 높은 정확도를 얻고자 하는 시도를 하였다. 이에 본 연구에서는, 인코더를 통한 좋은 잠재 공간을 구하고, 이를 GNN 모델 기반 네트워크를 통하여 감정 인식 작업을 진행했다. 이 작업을 진행하기 위하여 발화 상황의 오디오 데이터, EDA 데이터 그리고 온도 데이터를 융합하여 사용했다. 멀티 모달 데이터를 사용하여, 다양한 단서를 모델의 학습에 이용할 수 있게되었다.

## 1. 소개
### 1.1 대회 소개
#### 멀티모달 감정 데이터셋 활용 감정 인식 기술 분야
- 인간과 교감할 수 있는 인공지능 구현을 위해서는 인간의 행동과 감정을 이해하는 기술이 필요합니다.
- 사람의 행동과 감정을 이해하는 기술 연구를 가능토록 하기 위해 구축한 데이터셋을 활용하여 휴먼이해 인공지능 기술 연구를 확산시키고 창의적인 여구를 발굴하고자 ETRI에서 대회를 개최했습니다.

- Task는 다음과 같습니다.
    - 우리는 본 연구에서 일반인 대상 자유발화:[KEMDy20](https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR) 데이터셋을 활용하여 감정의 레이블(기쁨, 놀람, 분노, 중립, 혐오, 공포, 슬픔)에 대한 분류 정확도(F1)를 제시합니다.
    - 멀티모달 데이터를 혼합합니다. 발화음성과 EDA 데이터, 온도 데이터를 사용하여 멀티모달 데이터 감정인식 모델을 구축했습니다.
    - 임베딩 벡터로 만들어진 graph를 GNN 모델에 통과시켜 감정분류 예측을 합니다.


### 1.2 Methodolgy
#### Model Architecture
![model_architecture](./images/)

#### Audio Spectrogram
![mel-spectrogram](./images/)
- _mel-spectrogram 설명_

#### Korean Sentence Embedding[1]
- 

#### Encoder[2]
- 

![]()

### 1.3 코드 설명

```data_loader.py``` : data loading for generating graph and training, validation

```generate_graph.py``` : 임베딩 벡터로 graph 생성

```models_gnn.py``` : 우리의 모델

```train_val.py``` : 학습과 검증을 진행

### 1.4 데이터 전처리
- 우리가 수행한 데이터 전처리 과정을 제시합니다.
#### 오디오
우리는 Multimodal dataset을 활용하기위해 함께 첨부된 txt file을 전처리했습니다.
대화에서 "\c" "\n"과같은 문자부호 특수문자들을 모두 제거하고 KoBert를 활용해 text를 embedding을 모두 출력하고 이 embedding들의 평균을 문장의 embedding으로 인정했습니다.
또한 충분히 텍스트 정보를 잘 포함하는 embedding dimension을 768로 설정하였습니다.
이러한 처지의 결과는 <2.2 데이터셋 다운로드>의 구글드라이브 링크에서 ```embedding_768.npy```에서 확인 가능합니다.
</br>

#### EDA
오디오를 참조하기 쉽게 한 폴더(```./audio```)에 모았습니다. 각각의 오디오는 고유한 이름을 가지고 있으므로 한 폴더에서 참조해도 문제가 없습니다.

#### TEMP
20개의 Session에 약 10개씩의 대화상황이 있습니다. 또한 각 Session에 맞는 csv annotation file이 KEMDy19에 기본적으로 포함되어 있습니다.
우리는 각 대화상황마다 흩어진 annotation들을 하나로 묶은 ```annotation.csv``` file을 만들었습니다. 이것은 아래 <2.2 데이터셋 다운로드>에서 확인하실 수 있습니다.
각 대화마다 청자 또는 화자의 감정이 label되어있으므로 이것을 speaker와 listener 2개의 csv file(```df_listener.csv```,```df_speaker.csv```)로 나누었습니다. 역시 같은 섹션에서 결과를 확인 가능합니다.

### 1.5 Graph 생성

## 2. How To Use?
- 이 코드를 사용하는 방법을 다룹니다
- 순서와 지시를 __그대로__ 따라 사용해주세요

### 2.1 환경설정
0. 여러분의 PC나 서버에 GPU가 있고 cuda setting이 되어있어야합니다.
1. 여러분의 환경에 이 repo를 clone합니다 : ```git clone <this_repo>```
2. requirements libraries를 확인합니다 : ```pip install -r requirements.txt```

### 2.2 데이터셋 다운로드
1. [KEMDy20](https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR) dataset을 다운로드하여 'ETRI_2022_AI_Competition/data' 폴더에 넣으세요. 다운로드 권한을 신청해야할 수도 있습니다.
2. [Google_Drive]()에서 미리 가공된 데이터들을 다운로드하여 'multimodal-emotion/' 폴더에 넣으세요.
3. [Google_Drive]()에서 ```.zip```을 다운로드하여 압축을 풀어서 로컬인 ```multimodal-emotion/``` 폴더에 넣으세요. 

- 최종적으로 structure가 이렇게 되어있다면 모든 준비가 끝났습니다!
```
<2022_ETRI_AI_Competition>
                    ├ <data>
                        └ <KEMDy19>
                            ├ <annotation>
                            ├ <ECG>
                            ├ <EDA>
                            ├ <TEMP>
                            ├ <wav>
                            ├ annotation.csv
                            ├ df_listener.csv
                            ├ df_speaker.csv
                            └ embedding_768.npy
                    ├ <audio>    
                    ├ constants.py
                    ├ dataset.py
                    ├ loss.py
                    ├ main.py
                    ├ metric.py
                    ├ model.py
                    ├ utils.py
                    ├ EDA.ipynb
                    ├ prerprocessing.ipynb
                    ├ LICENSE
                    ├ requirements.txt
                    └ README.md                           
```

### 2.3 학습+추론
‼️ 여러분의 GPU에 따라서 ```gpu_id```(이름이 다르거나 없어서 오류)나 ```batch_size```(memory overflow)을 예시와 다르게 설정해야 할 수도 있습니다. 오류가 뜬다면 아래 "argparser parameter 소개"를 보면서 여러분의 환경에 맞게 조정해주세요.

#### Speaker 감정 추론 baseline
```
python main.py --SorL speaker
               --epochs 100
```

#### Listener 감정 추론 baseline
```
python main.py --SorL listener
               --epochs 100
```
- argparser parameter 소개
    - gup_id : 사용할 GPU의 id
    - save_path : 실험결과가 저장될 경로 -> 만지지 마시오
    - backbone : Backbone network -> 만지지 마시오
    - text_dim : sentence embedding의 dimension -> 만지지 마시오
    - bidirectional : 양방향 RNN옵션. Default는 False. True로 만드려면 ```--bidirectional```
    - ws : sliceing window size -> 만지지 마시오
    - SorL : 추론할 감정. ```speaker``` 또는 ```listener```
    - sr : audio의 sampling rate -> 만지지 마시오
    - test_split : 20개의 session중에서 test split으로 나눌 session. 예시) ```[1,8,9,13]```
    - batch_size : Batch size ```64```
    - optim : optimizer. choices=sgd,adam,adagrad
    - loss : Default는 ```normal```로 MSE와 KLDiv Loss를 사용합니다. ```cbloss```로 설정하면 Class Balanced Loss가 사용됩니다.
    - lam : Default는 ```0.66```으로 감정을 맞추는 가중치(lam)와 각성도,긍부정도를 맞추는 가중치(1-lam) 사이의 비율을 결정합니다.
    - beta : Default는 ```0.99```. CBLoss의 가중치 beta를 결정합니다.
    - lr_decay : lr decay term
    - lr : learning rate
    - weight_decay : weight decay term(L2 regularization)
    - epochs : total training epochs



### 2.4 추론만 하기
```
python main.py --test_only
               --./exp에있으면서_test할_모델이_있는_폴더_이름
```
예를 들어서 ```exp/lstm_speaker_adam_8/model.pth```가 있었고 이 모델을 테스트만 하고싶다면 아래와같이 명령하세요.
```
python main.py --test_only
               --lstm_speaker_adam_8
```

## 3. 성능
### 3.1 기존 성능[3]

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| SPSL | 0.608 | - | 0.599 | - | - |
| MPSL | 0.591 | - | 0.584 | - | - |
| MPGL | 0.608 | - | 0.598 | - | - |

### 3.2 Baseline 성능

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Speaker[1,2,3,4] | 0.687 | 0.663 | 0.674 | 0.771 | 0.845 |
| Speaker[5,6,7,8] | 0.685 | 0.663 | 0.673 | 0.781 | 0.777 |
| Speaker[9,10,11,12] | 0.719 | 0.691 | 0.704 | 0.802 | 0.891 |
| Speaker[13,14,15,16] | 0.748 | 0.719 | 0.733 | 0.745 | 0.860 |
| Speaker[17,18,19,20] | 0.718 | 0.688 | 0.702 | 0.751 | 0.872 |

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Listener[1,2,3,4] | 0.696 | 0.669 | 0.681 | 0.691 | 0.852 |
| Listener[5,6,7,8] | 0.671 | 0.651 | 0.661 | 0.767 | 0.816 |
| Listener[9,10,11,12] | 0.660 | 0.632 | 0.645 | 0.724 | 0.860 |
| Listener[13,14,15,16] | 0.744 | 0.713 | 0.728 | 0.756 | 0.865 |
| Listener[17,18,19,20] | 0.710 | 0.683 | 0.695 | 0.566 | 0.866 |

### 3.3 양방향 RNN
- configuration은 모두 default setting

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Speaker_bi | 0.755 | 0.726 | 0.740 | 0.784 | 0.884 |
| Listener_bi | 0.741 | 0.710 | 0.725 | 0.711 | 0.880 |

### 3.4 Emotion 정보를 concat한 것이 도움이 되었을까?
- configuration은 모두 default setting
- emotion 정보 concat을 모두 제거 후 실험

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Speaker_noCat | 0.719 | 0.690 | 0.704 | 0.751 | 0.827 |
| Listener_noCat | 0.722 | 0.692 | 0.706 | 0.689 | 0.876 |

### 3.5 CB Loss when ![](http://latex.codecogs.com/gif.latex?\lambda=0.9)

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Speaker(![](http://latex.codecogs.com/gif.latex?\beta=0.8)) | 0.716 | 0.686 | 0.700 | 0.755 | 0.796 |
| Speaker (![](http://latex.codecogs.com/gif.latex?\beta=0.9)) | 0.621 | 0.594 | 0.607 | 0.623 | 0.785 |
| Speaker(![](http://latex.codecogs.com/gif.latex?\beta=0.99)) | 0.721 | 0.691 | 0.705 | 0.612 | 0.829 |
| Speaker(![](http://latex.codecogs.com/gif.latex?\beta=0.999)) | 0.670 | 0.643 | 0.656 | 0.730 | 0.869 |

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Linstener(![](http://latex.codecogs.com/gif.latex?\beta=0.8)) | 0.727 | 0.697 | 0.711 | 0.701 | 0.857 |
| Listener(![](http://latex.codecogs.com/gif.latex?\beta=0.9)) | 0.745 | 0.715 | 0.729 | 0.723 | 0.877 |
| Listener(![](http://latex.codecogs.com/gif.latex?\beta=0.99))| 0.711 | 0.681 | 0.695 | 0.725 | 0.854 |
| Listener(![](http://latex.codecogs.com/gif.latex?\beta=0.999)) | 0.698 | 0.669 | 0.682 | 0.741 | 0.868 |


### 3.6 ![](http://latex.codecogs.com/gif.latex?\lambda)에 따른 baseline ablation study

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Speaker(![](http://latex.codecogs.com/gif.latex?\lambda=0.5))(1:1:1) | 0.738 | 0.709 | 0.722 | 0.780 | 0.824 |
| Speaker(![](http://latex.codecogs.com/gif.latex?\lambda=0.66))(2:1:1) | 0.748 | 0.719 | 0.733 | 0.745 | 0.860 |
| Speaker(![](http://latex.codecogs.com/gif.latex?\lambda=0.75))(3:1:1) | 0.759 | 0.731 | 0.744 | 0.791 | 0.869 |
| Speaker(![](http://latex.codecogs.com/gif.latex?\lambda=0.8))(4:1:1) | 0.696 | 0.670 | 0.682 | 0.783 | 0.803 |

| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Listener(![](http://latex.codecogs.com/gif.latex?\lambda=0.5))(1:1:1) | 0.712 | 0.683 | 0.696 | 0.746 | 0.880 |
| Listener(![](http://latex.codecogs.com/gif.latex?\lambda=0.66))(2:1:1) | 0.744 | 0.713 | 0.728 | 0.756 | 0.865 |
| Listener(![](http://latex.codecogs.com/gif.latex?\lambda=0.75))(3:1:1) | 0.740 | 0.710 | 0.724 | 0.712 | 0.870 |
| Listener(![](http://latex.codecogs.com/gif.latex?\lambda=0.8))(4:1:1) | 0.709 | 0.680 | 0.694 | 0.688 | 0.862 |


### 3.7 최고성능을 5-Folds 검증으로 확인한 최종성능
| Model | Precision | Recall | F1_emotion | Arousal | Valence |
| --- | --- | --- | --- | --- | --- |
| Speaker(![](http://latex.codecogs.com/gif.latex?\lambda=0.75))(3:1:1) | 0.728 | 0.702 | 0.714 | 0.778 | 0.848 |
| Listener(CBLoss,![](http://latex.codecogs.com/gif.latex?\lambda=0.9),![](http://latex.codecogs.com/gif.latex?\beta=0.9)) | 0.694 | 0.668 | 0.680 | 0.711 | 0.833 |


## License & citiation
### License
MIT License 하에 공개되었습니다. 모델 및 코드를 사용시 첨부된 ```LICENSE```를 참고하세요.
### Citiation
```

```


## Contact
- Junseok Yoon : 
- Hong-Ju Jeong : sub06038@khu.ac.kr
- Inhun Choi : 

- Hyeon-Joon Choi : 
- Junsick Hong:

## Reference
[1] Patrick, M, et al. "Space-time crop & attend: Improving cross-modal video representation learning." arXiv preprint arXiv:2103.10211 (2021).
</br>
[2] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
</br>
[3] Min, Kyle, et al. "Learning long-term spatial-temporal graphs for active speaker detection." Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXV. Cham: Springer Nature Switzerland, 2022.
</br>
[4] Deng, J.J.; Leung, C.H.C. Towards Learning a Joint Representation from Transformer in Multimodal Emotion Recognition. In Brain Informatics; Mahmud, M., Kaiser, M.S., Vassanelli, S., Dai, Q., Zhong, N., Eds.; Springer: Cham, Switzerland, 2021;
pp. 179–188.
</br>
[5] Georage, Barnum. et al. "On The Benefits of Early Fusion in Multimodal Representation Learning." arXiv preprint arXiv:2011.07171 (2020).
</br>
[6] K. Gadzicki, R. Khamsehashari and C. Zetzsche, "Early vs Late Fusion in Multimodal Convolutional Neural Networks," 2020 IEEE 23rd International Conference on Information Fusion (FUSION), Rustenburg, South Africa, 2020, pp. 1-6.
</br>
[7] Liang, Chen, et al. "S+ page: A speaker and position-aware graph neural network model for emotion recognition in conversation." arXiv preprint arXiv:2112.12389 (2021).
</br>
[8] Poria, Soujanya, et al. "Meld: A multimodal multi-party dataset for emotion recognition in conversations." arXiv preprint arXiv:1810.02508 (2018).
</br>
[9] K. J. Noh and H. Jeong, “KEMDy20,” https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR 