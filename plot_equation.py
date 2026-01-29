import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_plane():
    # 1. X1, X2 범위 설정
    x1 = np.linspace(-10, 10, 50)
    x2 = np.linspace(-10, 10, 50)
    
    # 2. 격자(Grid) 생성
    X1, X2 = np.meshgrid(x1, x2)
    
    # 3. Y 값 계산: y = 2*X1 + 3*X2 + 1
    Y = 2 * X1 + 3 * X2 + 1
    
    # 4. 그래프 생성
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # 5. 평면 그리기 (surface plot)
    surf = ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.8)
    
    # 6. 라벨 및 타이틀 설정
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.set_title('$y = 2X_1 + 3X_2 + 1$')
    
    # 컬러바 추가
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.show()

if __name__ == "__main__":
    plot_3d_plane()
