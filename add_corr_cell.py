import json
import os

notebook_path = r'c:\Users\Brain\github\DataSicence\scikit-learn\타이타닉생존자예측-모델링-테스트.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the index where "3. 전처리(Preprocessing)" markdown cell is
target_idx = 11
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and '3. 전처리' in ''.join(cell['source']):
        target_idx = i
        break

# Create the correlation cell
corr_cell_md = {
    "cell_type": "markdown",
    "id": "corr-explanation",
    "metadata": {},
    "source": [
        "### 상관 관계 분석\n",
        "Pandas의 `corr()` 메서드를 사용하여 변수 간의 상관 관계를 확인합니다. \n",
        "**주의:** 최신 버전의 Pandas(2.0 이상)에서는 수치형 데이터가 아닌 컬럼이 포함된 경우 `numeric_only=True` 옵션을 명시해야 에러가 발생하지 않습니다."
    ]
}

corr_cell_code = {
    "cell_type": "code",
    "execution_count": None,
    "id": "corr-code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# 상관 관계 계산 (수치형 데이터만 선택)\n",
        "train_df.corr(numeric_only=True)"
    ]
}

# Insert before Section 3
nb['cells'].insert(target_idx, corr_cell_md)
nb['cells'].insert(target_idx + 1, corr_cell_code)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Added correlation analysis cells to {os.path.basename(notebook_path)}")
