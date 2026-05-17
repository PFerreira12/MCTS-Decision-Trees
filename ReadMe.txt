# 1. Criar o ambiente virtual
python -m venv venv

# 2. Ativar (Git Bash)
source venv/Scripts/activate

# 3. Instalar dependencias
pip install numpy pandas matplotlib seaborn scipy pygame jupyter

# 4. Treinar a Decision Tree ID3 implementada de raiz
python train_id3_popout.py

# 5. Treinar/validar ID3 no Iris com thresholds manuais
python train_id3_iris.py

# 6. Iniciar Jupyter
jupyter notebook PopOut_MCTS_DecisionTrees.ipynb

# 7. Abrir a interface grafica
python gui.py
7.1 MCTS vs ID3 - aqui

# Nota
# Se "python gui.py" disser que nao encontra pygame, usa o Python do Anaconda:
C:\Users\pavfe\anaconda3\python.exe gui.py
