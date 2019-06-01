#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import html
import model as model
import sample as sample
import encoder as encoder

model_name = 'checkpoint/pushshift-bigbatch'
seed = None
nsamples = 1
length = None
temperature = 0.9
top_k = 60
top_p = 0.9
batch_size = 1
gpus = 1

hpdir = '117M'

enc = encoder.get_encoder(hpdir)
hparams = model.default_hparams()
with open(os.path.join('models', hpdir, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

if length is None:
    length = hparams.n_ctx // 2
elif length > hparams.n_ctx:
    raise ValueError(
        "Can't get samples longer than window size: %s" % hparams.n_ctx)
g = tf.Graph()

conf = tf.ConfigProto(
    device_count={'GPU': gpus}
)
sess = tf.Session(graph=g, config=conf)
with g.as_default():
    context = tf.placeholder(tf.int32, [batch_size, None])
    np.random.seed(seed)
    tf.set_random_seed(seed)
    output = sample.sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k, top_p=top_p
    )

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
    print(ckpt)
    saver.restore(sess, ckpt)


def gen_model(prompts):
    with tf.Session(graph=tf.Graph(), config=tf.ConfigProto(
        device_count = {'GPU': 0}
    )) as sess:

        context_tokens = []
        for prompt in prompts:
            context_tokens.append(enc.encode(prompt.replace('\\n', '\n')))

        out = sess.run(output, feed_dict={
            context: context_tokens
        })

        texts = []
        for i, res in enumerate(out):
            text = enc.decode(res[len(context_tokens[i]):])
            text = html.unescape(text)
            texts.append(text)

        return texts
