"""constant dataset."""
import numpy as np
import tensorflow_datasets as tfds


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for constant dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(constant): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        # supervised_keys=('image', 'label'),  # Set to `None` to disable
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(constant): Downloads the data and defines the splits
    arr = np.load('/opt/home3/swyoon/energy-based-autoencoder/src/datasets/const_img.npy')

    # TODO(constant): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'test': self._generate_examples(arr),
    }

  def _generate_examples(self, arr):
    """Yields examples."""
    # TODO(constant): Yields (key, example) tuples from the dataset
    for i, f in enumerate(arr):
      yield i, {
          'image': (f * np.ones((32, 32, 3))).astype(np.uint8),
      }
