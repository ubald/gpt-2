#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import model, sample, encoder
from http.server import BaseHTTPRequestHandler, HTTPServer
from termcolor import colored

def interact_model(
    model_name='124M',
    seed=None,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
):
    """
    Interactively run the model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [1, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams,
            length=length,
            context=context,
            batch_size=1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        print(colored("Priming stuff", 'cyan'))
        context_tokens = enc.encode("Customer support bot")
        out = sess.run(output, feed_dict={context: [context_tokens]})[:, len(context_tokens):]
        text = enc.decode(out[0])
        print(colored(text, color='white', attrs=['dark']))

        def respond(prompt):
            context_tokens = enc.encode(prompt)
            out = sess.run(output, feed_dict={context: [context_tokens]})[:, len(context_tokens):]
            return str.strip(enc.decode(out[0]))


        class HTTPHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                prompt = post_data.decode('utf-8')

                print(f"{colored('Prompt', 'yellow')}: {colored(prompt, color='white', attrs=['dark'])}")

                text = respond(prompt)

                print(f"{colored('Generated', 'yellow')}: {colored(text, color='white', attrs=['dark'])}")

                self.send_response(200)
                self.send_header('Content-type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(text.encode('utf-8'))

        httpd = HTTPServer(('', 8000), HTTPHandler)
        print(colored("Starting web server", 'cyan'))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        httpd.server_close()


if __name__ == '__main__':
    fire.Fire(interact_model)


