import common
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

model = common.prepare_model()
model.load_weights(common.WeightsFile)
test_df = pd.read_csv(common.TestFile)
lookup_df = pd.read_csv(common.IdLookupTableFile)
test_imgs, y = common.process_img(test_df['Image'])
predictions = model.predict(test_imgs)
plt.imshow(test_imgs[0].reshape(96, 96), cmap='gray')
plt.scatter(predictions[0][0::2], predictions[0][1::2], c='red', marker='x')
plt.show()


locations = []
rows = []
train_df = pd.read_csv(common.TrainingFile)
labels = train_df.drop('Image', axis=1)
FEATURES = list(labels.columns)
for row_id, img_id, feature_name, loc in lookup_df.values:
    fi = FEATURES.index(feature_name)
    loc = predictions[img_id - 1][fi]
    locations.append(loc)
    rows.append(row_id)
row_id_series = pd.Series(rows, name='RowId')
loc_series = pd.Series(locations, name='Location')
sub_csv = pd.concat([row_id_series, loc_series], axis=1)
sub_csv.to_csv('face_key_detection_submission.csv', index=False)