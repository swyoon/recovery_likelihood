"""Manual Building of CelebA dataset"""

import tensorflow_datasets as tfds
import glob
import os


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for mydatset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(mydatset): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(178, 218, 3)),
            # 'label': tfds.features.ClassLabel(names=['no', 'yes']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        # supervised_keys=('image', 'label'),  # Set to `None` to disable
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def read_split_file(self, root_dir):
    split_path = '/opt/home3/swyoon/energy-based-autoencoder/src/datasets/CelebA/Eval/list_eval_partition.txt'
    d_split = {0:[], 1:[], 2:[]}
    with open(split_path) as f:
        for line in f:
            fname, setnum = line.strip().split()
            d_split[int(setnum)].append(fname)
    return d_split

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(mydatset): Downloads the data and defines the splits
    path = '/opt/home3/swyoon/energy-based-autoencoder/src/datasets/CelebA/Img/img_align_celeba'
    d_split = self.read_split_file(path)

    # TODO(mydatset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path, d_split, 0),
        'eval': self._generate_examples(path, d_split, 2),
    }

  def _generate_examples(self, path, d_split, split_key):
    """Yields examples."""
    # TODO(mydatset): Yields (key, example) tuples from the dataset
    # for f in glob.glob(path + '/*.jpg'):
    for f in d_split[split_key]:
      f = os.path.join(path, f)
      yield f, {
          'image': f,
          # 'label': 'yes',
      }
