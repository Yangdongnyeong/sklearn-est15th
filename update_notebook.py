import json
from pathlib import Path

notebook_path = r'c:\Users\Brain\github\DataSicence\scikit-learn\Plus_6_LinearRegressionModel.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 찾아야 할 셀 ID
cell_id_1 = "a5c7c035"
cell_id_2 = "12947c9f"

code_1 = [
    "import urllib.request\n",
    "from pathlib import Path\n",
    "\n",
    "# 캘리포니아 지도 다운로드\n",
    "IMAGES_PATH = Path() / \"images\"\n",
    "IMAGES_PATH.mkdir(parents=True, exist_ok=True)\n",
    "filename = \"california.png\"\n",
    "if not (IMAGES_PATH / filename).is_file():\n",
    "    homl3_root = \"https://github.com/ageron/handson-ml3/raw/main/\"\n",
    "    url = homl3_root + \"images/end_to_end_project/\" + filename\n",
    "    print(\"Downloading\", filename)\n",
    "    urllib.request.urlretrieve(url, IMAGES_PATH / filename)"
]

code_2 = [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# 위도/경도를 이용한 구역별 인구 및 집값 시각화\n",
    "ax = df.plot(kind=\"scatter\", x=\"Longitude\", y=\"Latitude\", \n",
    "             s=df[\"Population\"] / 100, label=\"Population\", \n",
    "             c=\"MedHouseVal\", cmap=\"turbo\", colorbar=False, \n",
    "             figsize=(10, 7), alpha=0.4)\n",
    "\n",
    "# 다운로드된 캘리포니아 지도 표시\n",
    "california_img = plt.imread(str(IMAGES_PATH / filename))\n",
    "axis = -124.55, -113.8, 32.45, 42.05 # x축, y축 범위 설정\n",
    "plt.axis(axis)\n",
    "plt.imshow(california_img, extent=axis)\n",
    "\n",
    "# 컬러바 추가\n",
    "cbar = plt.colorbar(ax.collections[0], label=\"Median House Value\", orientation='vertical', fraction=0.046, pad=0.04)\n",
    "\n",
    "plt.xlabel(\"Longitude\")\n",
    "plt.ylabel(\"Latitude\")\n",
    "plt.legend()\n",
    "plt.show()"
]

for cell in nb['cells']:
    if cell['id'] == cell_id_1:
        cell['source'] = code_1
    elif cell['id'] == cell_id_2:
        cell['source'] = code_2

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print("Notebook updated successfully.")
