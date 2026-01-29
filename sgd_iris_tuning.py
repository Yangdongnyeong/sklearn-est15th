import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 2. 파이프라인 구축 (SGD는 스케일링이 필수적입니다)
pipeline = make_pipeline(
    StandardScaler(),
    SGDClassifier(random_state=42)
)

# 3. 하이퍼파라미터 그리드 설정
param_grid = {
    # 손실 함수: 경사하강법이 최적화할 대상
    # 'hinge': 선형 SVM (기본값)
    # 'log_loss': 로지스틱 회귀 (확률 출력 가능)
    # 'modified_huber': 이상치에 좀 더 강건함
    'sgdclassifier__loss': ['hinge', 'log_loss', 'modified_huber'],

    # 규제 (Regularization) 방식
    'sgdclassifier__penalty': ['l2', 'l1', 'elasticnet'],

    # 규제 강도 (alpha): 값이 클수록 규제가 강해짐 (과대적합 방지)
    'sgdclassifier__alpha': [0.0001, 0.001, 0.01, 0.1],

    # 학습률 스케줄링
    # 'optimal': 기본값, 효율적
    # 'adaptive': 학습이 정체되면 학습률을 감소시킴 (eta0 필요)
    'sgdclassifier__learning_rate': ['optimal', 'adaptive'],
    
    # 초기 학습률 (adaptive 사용 시 중요)
    'sgdclassifier__eta0': [0.01, 0.1],

    # 최대 반복 횟수 (충분히 주어야 수렴함)
    'sgdclassifier__max_iter': [1000, 2000]
}

if __name__ == "__main__":
    # 4. 그리드 서치 수행
    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X, y)

    # 5. 결과 출력
    print(f"최적의 파라미터:\n{grid.best_params_}")
    print(f"최고 교차 검증 점수: {grid.best_score_:.4f}")

    # 테스트
    print(f"전체 데이터셋 점수: {grid.score(X, y):.4f}")
