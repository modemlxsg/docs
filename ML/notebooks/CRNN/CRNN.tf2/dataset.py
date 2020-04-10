import tensorflow as tf
import os
import yaml
import utils


class Mj_Dataset:

    def __init__(self, mode, item_num=100):
        config_file = open('config.yaml', 'r', encoding='utf-8')
        config = config_file.read()
        config_file.close()
        self.config = yaml.full_load(config)
        self.root_dir = self.config['dataset']['SynthText']['root_dir']
        self.item_num = self.config['dataset']['SynthText'][f'{mode}_num']
        self.mode = mode

        lexicon = [x for x in self.config['lexicon']['chars']]
        self.lexicon = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(lexicon, tf.range(len(lexicon))), -1
        )

        self.paths = self.getPaths()

    def getDS(self):
        ds = tf.data.Dataset.from_tensor_slices(self.paths)
        ds = ds.map(self.readData)
        return ds

    def readData(self, path):
        label = tf.strings.split(path, '/')[-1]
        label = tf.strings.split(label, '_')[1]
        im = tf.io.read_file(path)
        im = tf.io.decode_jpeg(im, channels=1)
        im = tf.image.convert_image_dtype(im, tf.float32)
        im = tf.image.resize(im, [32, 100])

        return im, label

    def encode(self, batch_labels):
        chars = tf.strings.unicode_split(batch_labels, input_encoding="UTF-8")
        mapped_label = tf.ragged.map_flat_values(self.lexicon.lookup, chars)
        sparse_label = mapped_label.to_sparse()
        sparse_label = tf.cast(sparse_label, tf.int32)

        return sparse_label


    def getPaths(self):
        with open(os.path.join(self.root_dir, f"annotation_{self.mode}.txt")) as f:
            lines = f.readlines()

        i, items = 0, []
        for line in lines:
            if i >= self.item_num:
                break
            fullpath = os.path.join(self.root_dir, line.split(' ')[0])
            if os.path.exists(fullpath):
                items.append(fullpath)
                i += 1
        return items

if __name__ == "__main__":
    dataset = Mj_Dataset('train')
    print(dataset.lexicon)
    
    ds = dataset.getDS()
    ds = ds.batch(2)

    for imgs, labels in ds:
        
        print(imgs.shape, labels)

        imgs, labels = dataset.encode(imgs, labels)
        print(imgs.shape, labels)
        break

