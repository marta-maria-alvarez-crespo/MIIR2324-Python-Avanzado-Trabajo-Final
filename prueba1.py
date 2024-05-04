from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, classification_report
import numpy as np

y_true= ["Marta", "Marta", "Marta", "Marta"]
y_pred= ["Marta", "Marta", "Marta", "Maria"]

print(classification_report(y_true, y_pred, zero_division= np.nan))