import json

notebook_path = r'c:\Users\Brain\github\DataSicence\scikit-learn\타이타닉생존자예측-모델링-테스트 강사님꺼.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "train_test_split(train_df2.drop('Survived', axis=1)" in line:
                source[i] = line.replace("train_df2.drop('Survived', axis=1)", "train_df2.drop(['Survived', 'Name'], axis=1)")
                print(f"Updated line: {source[i].strip()}")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
