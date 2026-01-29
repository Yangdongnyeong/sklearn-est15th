import json

notebook_path = r'c:\Users\Brain\github\DataSicence\scikit-learn\타이타닉생존자예측-데이터전처리.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "markdown",
    "id": "titanic-description",
    "metadata": {},
    "source": [
        "# 타이타닉 생존자 예측 데이터셋 (Titanic: Machine Learning from Disaster)\n",
        "\n",
        "이 데이터셋은 1912년 침몰한 타이타닉호의 승객 정보를 담고 있습니다. 주요 목표는 승객의 여러 특성을 바탕으로 생존 여부(Survived)를 예측하는 것입니다.\n",
        "\n",
        "#### 데이터 사전 (Data Dictionary)\n",
        "\n",
        "| 변수명 (Variable) | 정의 (Definition) | 키 (Key) |\n",
        "| :--- | :--- | :--- |\n",
        "| **Survival** | 생존 여부 | 0 = No, 1 = Yes |\n",
        "| **Pclass** | 티켓 등급 (사회-경제적 지위) | 1 = 1st, 2 = 2nd, 3 = 3rd |\n",
        "| **Sex** | 성별 | |\n",
        "| **Age** | 나이 | |\n",
        "| **SibSp** | 타이타닉호에 동승한 형제/배우자 수 | |\n",
        "| **Parch** | 타이타닉호에 동승한 부모/자녀 수 | |\n",
        "| **Ticket** | 티켓 번호 | |\n",
        "| **Fare** | 운임 (승객 요금) | |\n",
        "| **Cabin** | 객실 번호 | |\n",
        "| **Embarked** | 중간 정착 항구 | C = Cherbourg, Q = Queenstown, S = Southampton |\n",
        "\n",
        "#### 변수 참고 사항 (Variable Notes)\n",
        "\n",
        "* **pclass**: 사회-경제적 지위(SES)의 대리 지표\n",
        "    * 1st = 상류층 (Upper)\n",
        "    * 2nd = 중산층 (Middle)\n",
        "    * 3rd = 하류층 (Lower)\n",
        "* **age**: 나이가 1세 미만인 경우 분수(fractional)로 표시됩니다. 나이가 추정치인 경우 xx.5 형태로 표시됩니다.\n",
        "* **sibsp**: 가족 관계를 다음과 같이 정의합니다.\n",
        "    * Sibling: 형제, 자매, 의붓형제, 의붓자매\n",
        "    * Spouse: 남편, 아내 (내연녀, 약혼자는 무시됨)\n",
        "* **parch**: 가족 관계를 다음과 같이 정의합니다.\n",
        "    * Parent: 어머니, 아버지\n",
        "    * Child: 딸, 아들, 의붓딸, 의붓아들\n",
        "    * 참고: 유모(nanny)와 함께 여행한 어린이는 parch=0으로 표시됩니다."
    ]
}

nb['cells'].insert(0, new_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook updated successfully.")
