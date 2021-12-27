import numpy as np
import theano
import theano.tensor as T
from matplotlib import pyplot as plt
from tqdm import tqdm

from iwae import get_samples


def train(model, dataset, optimizer, minibatch_size, n_epochs, srng, **kwargs):
    print("training for {} epochs with {} learning rate".format(n_epochs, optimizer.learning_rate))
    num_minibatches = dataset.get_n_examples('train') / minibatch_size

    index = T.lscalar('i')
    minibatch = dataset.minibatchIindex_minibatch_size(index, minibatch_size, srng=srng, subdataset='train')
    L_k_q, grad = model.gradIminibatch_srng(minibatch, srng, **kwargs)
    updates = optimizer.updatesIgrad_model(grad, model)
    train_step = theano.function([index], L_k_q, updates=updates)

    for e in range(n_epochs):
        ls = []
        pbar = tqdm(range(num_minibatches))
        for i in pbar:
            _L_k_q = train_step(i)
            ls.append(_L_k_q)
            pbar.set_description("[e|{j:03d}/{n_epochs:03d}][l|{ls:.03f}] "
                                 .format(j=e, n_epochs=n_epochs, ls=np.mean(ls)))
        # After each epoch preview some samples
        if e % 10 == 0:
            samples = get_samples(model, 100)
            plt.imshow(samples, cmap='Greys')
            plt.axis('off')
            plt.title('[{mt}] 100 samples after epoch={e:03d}'.format(mt=kwargs["model_type"].upper(), e=e))
            plt.show()
    return model
