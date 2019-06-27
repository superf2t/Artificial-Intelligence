import matplotlib.pyplot as plt
from sklearn.metrics import  roc_curve, roc_auc_score

def plot_roc_curve(y_test, y_predicts, model_names):
    plt.figure()
    for i in range(len(y_predicts)):
        fp, tp, _ = roc_curve(y_test, y_predicts[i])
        roc_auc = roc_auc_score(y_test, y_predicts[i])
        plt.plot(fp, tp, lw=1, alpha=0.3,
                 label='ROC fold %s (AUC = %0.3f)' % (model_names[i], roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    return plt
