import json
import tensorflow as tf
import fire
import pickle
from model.net import SenCNN
from model.utils import PreProcessor
from pathlib import Path
from konlpy.tag import Okt
from tqdm import tqdm


def create_dataset(filepath, batch_size, shuffle=True, drop_remainder=True):
    ds = tf.data.TextLineDataset(filepath)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return ds

def main(cfgpath):
    # parsing config.json
    proj_dir = Path.cwd()
    params = json.load((proj_dir / cfgpath).open())

    # create dataset
    batch_size = params['training'].get('batch_size')
    tr_filepath = params['filepath'].get('tr')
    val_filepath = params['filepath'].get('val')
    tr_ds = create_dataset(tr_filepath, batch_size, True)
    val_ds = create_dataset(val_filepath, batch_size, False)

    # create pre_processor
    vocab = pickle.load((proj_dir / params['filepath'].get('vocab')).open(mode='rb'))
    pre_processor = PreProcessor(vocab=vocab, tokenizer=Okt)

    # create model
    model =SenCNN(num_classes=2, vocab=vocab)

    # create optimizer & loss_fn
    epochs = params['training'].get('epochs')
    learning_rate = params['training'].get('learning_rate')
    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.losses.SparseCategoricalCrossentropy()

    # training

    for epoch in tqdm(range(epochs), desc='epochs'):
        tr_loss = 0
        tf.keras.backend.set_learning_phase(1)

        for step, mb in tqdm(enumerate(tr_ds), desc='steps'):
            x_mb, y_mb = pre_processor.convert2idx(mb)
            with tf.GradientTape() as tape:
                mb_loss = loss_fn(y_mb, model(x_mb))
            grads = tape.gradient(target=mb_loss, sources=model.trainable_variables)
            opt.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
            tr_loss += mb_loss.numpy()
        else:
            tr_loss /= (step + 1)

        tf.keras.backend.set_learning_phase(0)
        val_loss = 0
        for step, mb in tqdm(enumerate(val_ds), desc='steps'):
            x_mb, y_mb = pre_processor.convert2idx(mb)
            mb_loss = loss_fn(y_mb, model(x_mb))
            val_loss += mb_loss.numpy()
        else:
            val_loss /= (step + 1)

        tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch + 1, tr_loss, val_loss))


if __name__ == '__main__':
    fire.Fire(main)
