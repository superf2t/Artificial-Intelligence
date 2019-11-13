# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

full = train.append( test , ignore_index = True )
titanic = full[ :891 ]
def CovertSex(x):
    if(x == "Male"):
        return  0
    else:
        return  1
train["Sex"] = train["Sex"].apply(CovertSex)
print(train["Sex"])

print(full.shape)
embarked = pd.get_dummies( full.Embarked , prefix='Embarked')
full = pd.concat((full, embarked), axis=1)
print(full.shape)