# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import accuracy_score

# Figures inline and set visualization style
try:
    # Enable inline plotting when running inside IPython/Jupyter notebooks
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("matplotlib", "inline")
except Exception:
    pass
sns.set()
# Import test and train datasets
df_train = pd.read_csv(
    "/Users/maneshss/Desktop/Study/Project/Mangithub/Mangithub/MachineLearning/titanic/train.csv"
)
df_test = pd.read_csv(
    "/Users/maneshss/Desktop/Study/Project/Mangithub/Mangithub/MachineLearning/titanic/test.csv"
)

# View first lines of training data
df_train.head(n=4)
# View first lines of training data
print(df_train.head(n=4))
