# Scikit-Learn Machine Learning Study Repository

이 리포지토리는 Scikit-Learn을 활용한 머신러닝 학습 및 실습 코드를 포함하고 있습니다.
기초적인 데이터 전처리부터 분류, 회귀, 앙상블 모델링, 그리고 하이퍼파라미터 튜닝(Optuna)까지 다양한 주제를 다루고 있습니다.

## 📂 주요 목차 (Curriculum)

이 프로젝트는 학습 순서에 따라 번호가 매겨진 노트북 파일들로 구성되어 있습니다.

### 1. 기초 및 데이터 전처리
- **1_sklearn_start.ipynb**: Scikit-Learn 기초 시작하기
- **2_ModelSelection.ipynb**: 교차 검증 및 모델 선택 방법
- **4_sklearn_PreProcess.ipynb**: 데이터 스케일링, 인코딩 등 전처리 기법
- **8_polynominal_Feature.ipynb**: 다항 특성 추가 (Feature Engineering)

### 2. 지도 학습 - 분류 (Classification)
- **3_SVM.ipynb**: Support Vector Machine (SVM) 모델 학습
- **5_sklearn_classification.ipynb**: 다양한 분류 알고리즘 실습
- **6_classification_Optuna.ipynb**: Optuna를 활용한 분류 모델 하이퍼파라미터 튜닝
- **Plus_1_sklearn_wine_classification.ipynb**: 와인 데이터셋 분류 실습

### 3. 지도 학습 - 회귀 (Regression)
- **9_LinearRegressionModel.ipynb**: 선형 회귀 모델 기초
- **Plus_5_LinearRegression_polyNorm.ipynb**: 다항 회귀 및 정규화
- **Plus_6_LinearRegressionModel.ipynb**: 심화 선형 회귀 실습

### 4. 앙상블 학습 (Ensemble Learning)
- **10_ensemble.ipynb**: 보팅(Voting), 배깅(Bagging), 부스팅(Boosting) 기초
- **11_ensemble_Optuna.ipynb**: 앙상블 모델 튜닝
- **Plus_7_ensemble.ipynb**: 다양한 앙상블 기법 종합 실습
- **Plus_7_ensemble_gemini.ipynb**: AutoML(AutoGluon) 등을 활용한 앙상블 실험

### 5. 비지도 학습
- **13_unsupervisedLearning.ipynb**: 군집화(Clustering) 및 차원 축소 등

## 📊 실전 프로젝트 (Projects)

### Titanic Survival Prediction
타이타닉 승객 생존 예측을 위한 데이터 분석 및 모델링 프로젝트입니다.
- **7_Titanic.ipynb**: 기본 타이타닉 분석
- **titanic-81-1-leader-board-score-guaranteed.ipynb**: 고득점 달성을 위한 심화 모델링
- **colab_titanic-*.ipynb**: Colab 환경에서의 실습 파일들

### Red Wine Quality Analysis
레드 와인 품질 데이터셋을 활용한 분석 프로젝트입니다.
- **Plus_2_Red_wine_quality_analysis.ipynb**: 와인 품질 데이터 EDA 및 분석
- **Plus_4_sklearn_red_wine_quality.ipynb**: Scikit-Learn을 활용한 품질 예측 모델링

## 🛠 유틸리티 및 기타
- **AutoML/**: AutoGluon 등을 활용한 자동화된 머신러닝 실험 폴더
- **webML/**: 웹 애플리케이션 연동 관련 코드 (예상)
- **folium_visualization_colored.ipynb**: Folium을 활용한 지도 시각화
- 다양한 `.py` 스크립트: 노트북 유지보수 및 데이터 처리 보조 스크립트

## 🚀 시작하기 (Getting Started)

### 필수 라이브러리 설치
본 프로젝트를 실행하기 위해 필요한 주요 라이브러리는 다음과 같습니다.
```bash
pip install scikit-learn pandas numpy matplotlib seaborn optuna autogluon
```

### 실행 방법
Jupyter Notebook 또는 Jupyter Lab을 실행하여 `.ipynb` 파일을 열어 실습을 진행할 수 있습니다.
```bash
jupyter lab
```

---
*Created by [Your Name/Team Name]*
