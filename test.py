import tensorflow as tf
import pandas as pd

CSV_COLUMNS = ["ad_provider", "gold", "success"]
FEATURES = ["ad_provider", "gold"]
TYPES = [tf.string, tf.float32]
LABEL = "success"


def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)

def main(unused_argv):
  # Load datasets
  training_set = pd.read_csv("train.csv", skipinitialspace=True,
                             skiprows=0, names=CSV_COLUMNS)
  test_set = pd.read_csv("test.csv", skipinitialspace=True,
                         skiprows=0, names=CSV_COLUMNS)
  # Feature cols
  ad_provider = tf.feature_column.categorical_column_with_vocabulary_list(
      "ad_provider", ["Organic","Apple Search Ads","googleadwords_int",
                      "Facebook Ads","website"]  )

  gold = tf.feature_column.numeric_column("gold")
  video_success = tf.feature_column.numeric_column("video_success")

  feature_columns = [
    tf.feature_column.indicator_column(ad_provider),
    tf.feature_column.numeric_column(key="gold"),
  ]

  # Train
  regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
                                        hidden_units=[40, 30, 20],
                                        model_dir="model1",
                                        optimizer='RMSProp'
                                        )
  regressor.fit(input_fn=get_input_fn(training_set), steps=5000)

  def json_serving_input_fn():
    """Build the serving inputs."""
    inputs = {}
    for feat, dtype in zip(FEATURES, TYPES):
      inputs[feat] = tf.placeholder(shape=[None], dtype=dtype)

    features = {
      key: tf.expand_dims(tensor, -1)
      for key, tensor in inputs.items()
    }
    print(inputs)
    print(features)
    return tf.contrib.learn.InputFnOps(features, None, inputs)

  regressor.export_savedmodel("test",json_serving_input_fn)

if __name__ == "__main__":
  tf.app.run()

