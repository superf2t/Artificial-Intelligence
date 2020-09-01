import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df2['UserupdateInfo1'] = df2.UserupdateInfo1.map(lambda x:x.lower())
