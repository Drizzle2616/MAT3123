Model Complexity Comparison
Linear Regression vs Decision Tree vs Random Forest

1. Overview

본 프로젝트는 동일한 회귀(regression) 문제에서
모델 복잡도(model complexity)가 일반화 성능(generalization)과 과적합(overfitting)에
어떤 영향을 주는지 비교·분석하는 것을 목표로 한다.

비교 모델은 다음과 같다.

- Linear Regression (단순 선형 모델)
- Decision Tree Regressor (비선형 모델, 높은 분산 가능)
- Random Forest Regressor (앙상블로 분산 감소 기대)

본 프로젝트는 "기계학습과응용" 수업에서 학습한
선형 회귀, 결정트리, 랜덤포레스트의 핵심 아이디어를
하나의 문제에 일관된 실험 설계로 적용해 확인하는 데 초점을 둔다.

--------------------------------------------------

2. Dataset

본 프로젝트는 scikit-learn 내장 데이터셋인 California Housing을 사용한다.

- 입력: 여러 개의 수치형 특성(feature)
- 출력: 주택 가격(연속값)

내장 데이터셋을 사용하여
추가 다운로드 없이 재현 가능한 실험을 구성한다.

--------------------------------------------------

3. Methods

3.1 Train/Test Split

- 데이터를 train/test로 분리한다.
- 동일한 분리 기준(random_state)을 사용하여 공정한 비교를 수행한다.

3.2 Models

(1) Linear Regression
- 선형 관계를 가정하는 가장 단순한 기준 모델이다.

(2) Decision Tree Regressor
- 비선형 패턴을 강하게 학습할 수 있으나
  과적합 위험이 커질 수 있다.
- max_depth를 변화시키며 모델 복잡도를 조절한다.

(3) Random Forest Regressor
- 여러 트리를 결합하는 앙상블 모델이다.
- 단일 트리 대비 분산을 줄여 일반화 성능을 개선할 수 있다.

3.3 Evaluation Metrics

회귀 성능 비교를 위해 다음 지표를 사용한다.

- RMSE (낮을수록 좋다)
- R^2 (높을수록 좋다)

또한 train 성능과 test 성능의 차이를 통해
과적합 여부를 함께 해석한다.

--------------------------------------------------

4. Results (How to Reproduce)

1) 의존성 설치
pip install -U numpy pandas matplotlib scikit-learn

2) 실행
python src/train_compare.py

실행 결과로 다음이 생성된다.

- results/metrics.csv
- results/comparison_plot.png

--------------------------------------------------

5. Discussion (Expected Findings)

- Linear Regression은 단순하여 과적합 위험이 낮지만
  비선형 관계를 충분히 설명하지 못할 수 있다.

- Decision Tree는 학습 데이터를 매우 잘 맞출 수 있으나
  깊이가 커질수록 train 성능은 좋아지고 test 성능은 악화되는
  과적합 패턴이 나타날 수 있다.

- Random Forest는 여러 트리를 평균화하여 분산을 줄이므로
  단일 트리 대비 test 성능이 더 안정적으로 개선되는 경향이 있다.

--------------------------------------------------

6. Conclusion

본 프로젝트는 모델 복잡도가 증가할수록
표현력은 증가하지만 과적합 위험도 함께 증가할 수 있음을 확인한다.

또한 앙상블(Random Forest)은
단일 고복잡도 모델(Decision Tree)의 분산 문제를 완화하여
일반화 성능을 개선할 수 있음을 실험적으로 보여준다.
