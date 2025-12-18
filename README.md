Handwritten Digit Recognition
Classical Machine Learning vs Deep Learning Comparison

1. Introduction

본 프로젝트는 손글씨 숫자 인식(handwritten digit recognition) 문제를 대상으로
전통적인 머신러닝 기법과 딥러닝 기법을 비교·분석하는 것을 목표로 한다.

이 프로젝트는 선수과목인 "수학과 프로그래밍" 수업에서 수행했던
CNN 기반 손글씨 인식 프로젝트를 기반으로 하며,
이번 "기계학습과응용" 수업에서 학습한 머신러닝 알고리즘을 추가로 적용하여
모델 간 차이와 특성을 비교한다.

이를 통해 수업에서 배운
- 선형 모델
- 차원 축소(PCA)
- 분류 알고리즘
- 딥러닝 모델의 특징

을 실제 문제에 적용하고,
머신러닝과 딥러닝의 접근 방식 차이를 실험적으로 확인한다.

--------------------------------------------------

2. Dataset

본 프로젝트에서는 MNIST 손글씨 숫자 데이터셋을 사용한다.

입력 데이터
- 28x28 크기의 흑백 손글씨 숫자 이미지

출력 라벨
- 0부터 9까지의 숫자 클래스

이미지는 모델에 따라 서로 다른 방식으로 처리된다.

전통적 머신러닝 모델
- 이미지 데이터를 1차원 벡터로 변환하여 사용

딥러닝 모델(CNN)
- 이미지의 공간적 구조를 유지한 채 입력

--------------------------------------------------

3. Methods

3.1 Classical Machine Learning Approach
PCA + Logistic Regression

전통적인 머신러닝 방식으로 PCA(Principal Component Analysis)와
Logistic Regression을 사용하였다.

처리 과정은 다음과 같다.

1. 각 이미지를 1차원 벡터로 변환한다
2. PCA를 적용하여 고차원 데이터를 저차원으로 축소한다
3. 축소된 특징 공간에서 Logistic Regression으로 숫자를 분류한다

이 방식은 수업 초반에 배운 선형 분류 모델과
차원 축소 기법의 역할을 확인하기에 적합하다.

--------------------------------------------------

3.2 Deep Learning Approach
Convolutional Neural Network (CNN)

딥러닝 방식으로는 기존에 구현한 CNN 기반 손글씨 인식 모델을 사용하였다.

CNN 모델의 특징은 다음과 같다.

- 이미지의 공간적 구조를 직접 학습
- 별도의 feature engineering 없이 특징 자동 추출
- 복잡한 패턴 학습에 유리

해당 모델은 합성곱 계층과 풀링 계층을 사용하여
손글씨 숫자의 형태적 특징을 효과적으로 학습한다.

--------------------------------------------------

4. Results

각 모델의 분류 성능을 정확도 기준으로 비교하였다.

Model: PCA + Logistic Regression
Accuracy: 상대적으로 낮음

Model: CNN
Accuracy: 높은 정확도

전통적인 머신러닝 모델은 기본적인 숫자 분류는 가능했으나,
이미지의 복잡한 공간적 패턴을 충분히 반영하는 데 한계가 있었다.

반면 CNN 모델은 이미지의 국소적 특징을 효과적으로 학습하여
더 높은 분류 성능을 보였다.

--------------------------------------------------

5. Discussion

PCA + Logistic Regression 모델은
차원 축소 이후에도 기본적인 분류 성능을 보였으나,
이미지 데이터의 공간적 정보를 직접 활용하지 못한다는 한계가 있었다.

이는 전통적인 머신러닝 모델이
feature engineering이나 전처리에 크게 의존한다는 점을 보여준다.

반면 CNN은 이미지의 공간적 구조를 그대로 입력으로 사용하여
특징을 자동으로 학습한다는 점에서
딥러닝 모델의 강점을 명확히 확인할 수 있었다.

이 결과는 수업에서 배운
- 선형 모델의 한계
- 딥러닝의 표현력
- 문제 유형에 따른 모델 선택의 중요성

을 실험적으로 확인한 사례이다.

--------------------------------------------------

6. Conclusion

본 프로젝트에서는 손글씨 숫자 인식 문제를 대상으로
전통적인 머신러닝 기법과 딥러닝 기법을 비교하였다.

그 결과,
- 단순한 선형 모델은 기본적인 분류는 가능하지만 성능에 한계가 있으며
- 딥러닝 모델은 복잡한 패턴을 효과적으로 학습할 수 있음을 확인하였다

이를 통해 "기계학습과응용" 수업에서 배운
머신러닝 알고리즘의 발전 흐름과
각 모델의 적용 범위를 실제 문제를 통해 이해할 수 있었다.

--------------------------------------------------

7. Repository Structure

.
├── classical_ml
│   └── pca_logistic.ipynb
├── deep_learning
│   └── cnn_model.py
├── data
├── results
│   └── accuracy_comparison.png
└── README.md

--------------------------------------------------

8. References

- MNIST Dataset
- Lecture notes of Machine Learning and Applications
- Previous project: HandToText (CNN-based handwritten digit recognition)
