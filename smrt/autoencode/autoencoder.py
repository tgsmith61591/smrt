# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The autoencoder(s)

from __future__ import division, absolute_import, division
import time
import numpy as np
import tensorflow as tf
from sklearn.externals import six
from sklearn.utils import gen_batches, check_array
from sklearn.utils.validation import check_is_fitted
from abc import ABCMeta, abstractmethod

from .layer import SymmetricalAutoEncoderTopography, SymmetricalVAETopography, _BaseDenseLayer
from .base import BaseAutoEncoder, ReconstructiveMixin, GenerativeMixin, _validate_float, DTYPE, NPDTYPE
from ..utils import overrides, get_random_state
from ._ae_utils import cross_entropy, kullback_leibler

__all__ = [
    'AutoEncoder',
    'VariationalAutoEncoder'
]


DEFAULT_L2 = 0.0001
DEFAULT_DROPOUT = 1.

# this dict maps all the supported activation functions to the tensorflow
# equivalent functions for the session model training. It also maps all the
# supported activation functions to the local variants for offline scoring operations.
PERMITTED_ACTIVATIONS = {
    'elu': tf.nn.elu,
    'identity': tf.identity,
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh
}

# this dict maps all the supported optimizer classes to the tensorflow
# callables. For now, only strings are supported for learning_function
PERMITTED_OPTIMIZERS = {
    'adadelta': tf.train.AdadeltaOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'adagrad-da': tf.train.AdagradDAOptimizer,
    'adam': tf.train.AdamOptimizer,
    'momentum': tf.train.MomentumOptimizer,
    'proximal-sgd': tf.train.ProximalGradientDescentOptimizer,
    'proximal-adagrad': tf.train.ProximalAdagradOptimizer,
    'rms_prop': tf.train.RMSPropOptimizer,
    'sgd': tf.train.GradientDescentOptimizer
}


def _validate_activation_optimization(activation_function, learning_function):
    """Given the keys for the activation function and the learning function
    get the appropriate TF callable. The reason we store and pass around strings
    is so the models can be more easily pickled (and don't attempt to pickle a
    non-instance method)

    Parameters
    ----------
    activation_function : str
        The key for the activation function

    learning_function : str
        The key for the learning function.

    Returns
    -------
    activation : callable
        The activation function

    learning : callable
        The learning function.
    """
    if isinstance(activation_function, six.string_types):
        activation = PERMITTED_ACTIVATIONS.get(activation_function, None)
        if activation is None:
            raise ValueError('Permitted activation functions: %r' % list(PERMITTED_ACTIVATIONS.keys()))
    else:
        raise TypeError('Activation function must be a string')

    # validation optimization function:
    if isinstance(learning_function, six.string_types):
        learning = PERMITTED_OPTIMIZERS.get(learning_function, None)
        if learning is None:
            raise ValueError('Permitted learning functions: %r' % list(PERMITTED_OPTIMIZERS.keys()))
    else:
        raise TypeError('Learning function must be a string')

    return activation, learning


class _SymmetricAutoEncoder(six.with_metaclass(ABCMeta, BaseAutoEncoder)):
    """Base class for the two provided autoencoders, which are architecturally symmetric
    in terms of hidden layers. The encode/decode functions will not work for non-symmetrically-architected
    neural networks.
    """
    def __init__(self, n_hidden, activation_function, learning_rate, n_epochs, batch_size, min_change, verbose,
                 display_step, learning_function, early_stopping, bias_strategy, random_state, layer_type,
                 dropout, l2_penalty, gclip_min, gclip_max, clip):

        super(_SymmetricAutoEncoder, self).__init__(activation_function=activation_function,
                                                    learning_rate=learning_rate, n_epochs=n_epochs,
                                                    batch_size=batch_size, n_hidden=n_hidden,
                                                    min_change=min_change, verbose=verbose,
                                                    display_step=display_step,
                                                    learning_function=learning_function,
                                                    early_stopping=early_stopping,
                                                    bias_strategy=bias_strategy,
                                                    random_state=random_state,
                                                    layer_type=layer_type, dropout=dropout,
                                                    l2_penalty=l2_penalty, gclip_min=gclip_min,
                                                    gclip_max=gclip_max, clip=clip)

    @abstractmethod
    def _initialize_graph(self, X, y):
        """Should be called in the ``fit`` method. This initializes the placeholder variables"""

    def _add_regularization(self, cost_function, topography):
        if self.l2_penalty is not None:
            penalties = [tf.nn.l2_loss(w) for w in topography.get_weights_biases()[0]]
            l2_reg = self.l2_penalty * tf.add_n(penalties)
            cost_function += l2_reg

        return cost_function

    def _clip_or_minimize(self, learning_function, rate, cost):
        # https://stackoverflow.com/questions/36498127/how-to-effectively-apply-gradient-clipping-in-tensor-flow
        if not self.clip:
            return learning_function(rate).minimize(cost)
        else:
            global_step = tf.Variable(0, trainable=False)
            optimizer = learning_function(rate)
            grads = optimizer.compute_gradients(cost, tf.trainable_variables())
            clipped = [
                (tf.clip_by_value(grad, self.gclip_min, self.gclip_max), var)
                for grad, var in grads
                if grad is not None
            ]

            return optimizer.apply_gradients(clipped, global_step=global_step)

    def fit(self, X, y=None, **run_args):
        """Train the neural network.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors as real numbers, where ``n_samples`` is the number of
            samples and ``n_features`` is the number of input features.

        y : array-like, optional (default=None)
            None. Pass-through for pipe-lining.

        **run_args : dict, optional
            A key-word dictionary of arguments to be passed to the :func:`_train` method.
        """
        # validate array before graph init
        X = check_array(X, accept_sparse=False, force_all_finite=True, ensure_2d=True, dtype=NPDTYPE)

        # set the TF seed
        rs = get_random_state(self.random_state)
        tf.set_random_seed(rs.get_state()[1][0])

        # assign X to tf as a placeholder before graph init
        self.X_placeholder = tf.placeholder(DTYPE, [None, X.shape[1]])

        # initialize the graph for each re-fit
        X, cost_function, optimizer, dropout = self._initialize_graph(X, y)

        # do training
        return self._train(self.X_placeholder, X, cost_function, optimizer, dropout, **run_args)

    def _train(self, X_placeholder, X_original, cost_function, optimizer, dropout, **run_args):
        # initialize global vars for tf - replace them if they already exist
        init = tf.global_variables_initializer()
        self.clean_session()
        sess = self.sess = tf.InteractiveSession()

        # run the training session
        sess.run(init)
        epoch_times = []
        costs = []
        last_cost = None

        # generate the batches in a generator from sklearn, but store
        # in a list so we don't have to re-gen (since the generator will be
        # empty by the end of the epoch)
        n_samples = X_original.shape[0]
        batches = list(gen_batches(n_samples, self.batch_size))

        # training cycle. For each epoch
        for epoch in range(self.n_epochs):
            # track epoch time
            start_time = time.time()

            # loop batches
            for batch in batches:

                # extract the chunk given the slice, and assert it's not length 0 or anything weird...
                chunk = X_original[batch, :]
                m, _ = chunk.shape
                assert m <= self.batch_size and m != 0  # sanity check

                # train the batch - runs optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost_function],
                                feed_dict={X_placeholder: chunk, dropout: self.dropout},
                                **run_args)

            # add the time to the times array to compute average later
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
            costs.append(c)

            # Display logs if display_step and verbose
            if epoch % self.display_step == 0 and self.verbose > 1:
                print('Epoch: %i, cost=%.6f, time=%.4f (sec)' % (epoch + 1, c, epoch_time))

            # update last_cost, and if it meets the stopping criteria, break.
            # we only do this if we've enabled early_stopping, though.
            if self.early_stopping:
                if last_cost is None:
                    last_cost = c
                else:
                    delta = abs(last_cost - c)
                    if delta <= self.min_change:
                        if self.verbose:
                            print('Convergence reached at epoch %i, stopping early' % epoch)
                        break
                    else:
                        last_cost = c

        # set instance vars
        self.train_cost_ = c
        self.epoch_times_ = epoch_times
        self.epoch_costs_ = costs

        if self.verbose:
            print('Optimization complete after %i epoch(s). Average epoch time: %.4f seconds'
                  % (len(epoch_times), np.average(epoch_times)))

        return self


class AutoEncoder(_SymmetricAutoEncoder, ReconstructiveMixin):
    """An AutoEncoder is a special case of a feed-forward neural network that attempts to learn
    a compressed feature space of the input tensor, and whose output layer seeks to reconstruct
    the original input. It is, therefore, a dimensionality reduction technique, on one hand, but
    can also be used for such tasks as de-noising and anomaly detection. It can be crudely thought
    of as similar to a "non-linear PCA."

    The ``AutoEncoder`` class learns to reconstruct its input, minimizing the MSE between
    training examples and the reconstructed output thereof.


    Parameters
    ----------
    n_hidden : int or list
        The hidden layer structure. If an int is provided, a single hidden layer is constructed,
        with ``n_hidden`` neurons. If ``n_hidden`` is an iterable, ``len(n_hidden)`` hidden layers
        are constructed, with as many neurons as correspond to each index, respectively.

    activation_function : str, optional (default='sigmoid')
        The activation function. Should be one of PERMITTED_ACTIVATIONS:
        ('elu', 'identity', 'relu', 'sigmoid', 'tanh')

    learning_rate : float, optional (default=0.05)
        The algorithm learning rate.

    n_epochs : int, optional (default=20)
        An epoch is one forward pass and one backward pass of *all* training examples. ``n_epochs``,
        then, is the number of full passes over the training data. The algorithm will stop early if
        the cost delta between iterations diminishes below ``min_change`` between epochs and if
        ``early_stopping`` is enabled.

    batch_size : int, optional (default=128)
        The number of training examples in a single forward/backward pass. As ``batch_size``
        increases, the memory required will also increase.

    min_change : float, optional (default=1e-3)
        An early stopping criterion. If the delta between the last cost and the new cost
        is less than ``min_change``, the network will stop fitting early (``early_stopping``
        must also be enabled for this feature to work).

    verbose : int, optional (default=0)
        The level of verbosity. If 0, no stdout will be produced. Varying levels of
        output will increase with an increasing value of ``verbose``.

    display_step : int, optional (default=5)
        The interval of epochs at which to update the user if ``verbose`` mode is enabled.

    learning_function : str, optional (default='rms_prop')
        The optimizing function for training. Default is ``'rms_prop'``, which will use
        the ``tf.train.RMSPropOptimizer``. Can be one of { ``'adadelta'``, ``'adagrad'``,
        ``'adagrad-da'``, ``'adam'``, ``'momentum'``, ``'proximal-sgd'``, ``'proximal-adagrad'``,
        ``'rms_prop'``, ``'sgd'``}

    early_stopping : bool, optional (default=False)
        If this is set to True, and the delta between the last cost and the new cost
        is less than ``min_change``, the network will stop fitting early.

    bias_strategy : str, optional (default='zeros')
        The strategy for initializing the bias vector. Default is 'zeros' and will
        initialize all bias values as zeros. The alternative is 'ones', which will
        initialize all bias values as ones.

    random_state : int, ``np.random.RandomState`` or None, optional (default=None)
        The numpy random state for seeding random TensorFlow variables in weight initialization.

    layer_type : str
        The type of layer, i.e., 'xavier'. This is the type of layer that
        will be generated. One of {'xavier', 'gaussian'}

    dropout : float, optional (default=1.0)
        Dropout is a mechanism to prevent over-fitting a network. Dropout functions
        by randomly dropping hidden units (and their connections) during training.
        This prevents units from co-adapting too much.

    l2_penalty : float or None, optional (default=0.0001)
        The L2 penalty (regularization term) parameter.

    gclip_min : float, optional (default=-5.)
        The minimum value at which to clip the gradient. Gradient clipping can be
        necessary for preventing vanishing or exploding gradients. Only necessary when
        ``clip`` is True.

    gclip_max : float, optional (default=5.)
        The maximum value at which to clip the gradient. Gradient clipping can be
        necessary for preventing vanishing or exploding gradients. Only necessary when
        ``clip`` is True.

    clip : bool, optional (default=True)
        Whether or not to clip the gradient in ``[gclip_min, gclip_max]``. Gradient
        clipping can be necessary for preventing vanishing or exploding gradients.


    Notes
    -----
    This class is based loosely on an example located at:
    https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py


    Attributes
    ----------
    topography_ : ``smrt.autoencode.layer._BaseSymmetricalTopography``
        The structure of hidden layers, weights and biases.

    train_cost_ : float
        The final cost as a result of the training procedure on the training examples.


    References
    ----------
    [1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document
        recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.

    [2] http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    """

    def __init__(self, n_hidden, activation_function='sigmoid', learning_rate=0.05, n_epochs=20,
                 batch_size=128, min_change=1e-3, verbose=0, display_step=5, learning_function='rms_prop',
                 early_stopping=False, bias_strategy='zeros', random_state=None, layer_type='xavier',
                 dropout=DEFAULT_DROPOUT, l2_penalty=DEFAULT_L2, gclip_min=-5., gclip_max=5., clip=True):

        super(AutoEncoder, self).__init__(activation_function=activation_function,
                                          learning_rate=learning_rate, n_epochs=n_epochs,
                                          batch_size=batch_size, n_hidden=n_hidden,
                                          min_change=min_change, verbose=verbose,
                                          display_step=display_step,
                                          learning_function=learning_function,
                                          early_stopping=early_stopping,
                                          bias_strategy=bias_strategy,
                                          random_state=random_state,
                                          layer_type=layer_type, dropout=dropout,
                                          l2_penalty=l2_penalty, gclip_min=gclip_min,
                                          gclip_max=gclip_max, clip=clip)

    @overrides(_SymmetricAutoEncoder)
    def _initialize_graph(self, X, y):
        # validate X, then make it into TF structure
        n_samples, n_features = X.shape

        # validate floats and other params...
        self._validate_for_fit()

        dropout = tf.placeholder_with_default(DEFAULT_DROPOUT, shape=[], name='dropout')  # make placeholder
        activation, learning_function = _validate_activation_optimization(self.activation_function,
                                                                          self.learning_function)

        # set up our weight matrix. This needs to be re-initialized for every fit, since (like sklearn)
        # we want to allow for model/transformer re-fits. IF we don't reinitialize, the next input
        # either gets a warm-start or a potentially already grand-mothered weight matrix.
        self.topography_ = SymmetricalAutoEncoderTopography(X_placeholder=self.X_placeholder,
                                                            n_hidden=self.n_hidden,
                                                            input_shape=n_features,
                                                            activation=activation,
                                                            layer_type=self.layer_type,
                                                            dropout=dropout,
                                                            bias_strategy=self.bias_strategy,
                                                            random_state=self.random_state)

        # define the encoder, decoder functions
        y_pred, y_true = self.topography_.decode, self.X_placeholder

        # get loss + regularization and optimizer, minimize MSE
        cost_function = self._add_regularization(tf.reduce_mean(tf.pow(y_true - y_pred, 2)), self.topography_)
        optimizer = self._clip_or_minimize(learning_function, self.learning_rate, cost_function)

        return X, cost_function, optimizer, dropout

    @overrides(BaseAutoEncoder)
    def transform(self, X):
        check_is_fitted(self, 'topography_')
        t = self.topography_
        return self.sess.run(t.encode, feed_dict={self.X_placeholder: X})

    @overrides(ReconstructiveMixin)
    def reconstruct(self, X):
        check_is_fitted(self, 'topography_')
        t = self.topography_
        return self.sess.run(t.decode, feed_dict={self.X_placeholder: X})


class VariationalAutoEncoder(_SymmetricAutoEncoder, GenerativeMixin):
    """An AutoEncoder is a special case of a feed-forward neural network that attempts to learn
    a compressed feature space of the input tensor, and whose output layer seeks to reconstruct
    the original input. It is, therefore, a dimensionality reduction technique, on one hand, but
    can also be used for such tasks as de-noising and anomaly detection. It can be crudely thought
    of as similar to a "non-linear PCA."

    The ``VariationalAutoEncoder`` class, as it is intended in ``smrt``, is used to ultimately identify
    the more archetypal minority-class training examples to generate observations.


    Parameters
    ----------
    n_hidden : int or list
        The hidden layer structure. If an int is provided, a single hidden layer is constructed,
        with ``n_hidden`` neurons. If ``n_hidden`` is an iterable, ``len(n_hidden)`` hidden layers
        are constructed, with as many neurons as correspond to each index, respectively.

    n_latent_factors : int or float
        The size of the latent factor layer learned by the ``VariationalAutoEncoder``

    activation_function : str, optional (default='sigmoid')
        The activation function. Should be one of PERMITTED_ACTIVATIONS:
        ('elu', 'identity', 'relu', 'sigmoid', 'tanh')

    learning_rate : float, optional (default=0.05)
        The algorithm learning rate.

    n_epochs : int, optional (default=20)
        An epoch is one forward pass and one backward pass of *all* training examples. ``n_epochs``,
        then, is the number of full passes over the training data. The algorithm will stop early if
        the cost delta between iterations diminishes below ``min_change`` between epochs and if
        ``early_stopping`` is enabled.

    batch_size : int, optional (default=128)
        The number of training examples in a single forward/backward pass. As ``batch_size``
        increases, the memory required will also increase.

    min_change : float, optional (default=1e-3)
        An early stopping criterion. If the delta between the last cost and the new cost
        is less than ``min_change``, the network will stop fitting early (``early_stopping``
        must also be enabled for this feature to work).

    verbose : int, optional (default=0)
        The level of verbosity. If 0, no stdout will be produced. Varying levels of
        output will increase with an increasing value of ``verbose``.

    display_step : int, optional (default=5)
        The interval of epochs at which to update the user if ``verbose`` mode is enabled.

    learning_function : str, optional (default='rms_prop')
        The optimizing function for training. Default is ``'rms_prop'``, which will use
        the ``tf.train.RMSPropOptimizer``. Can be one of { ``'adadelta'``, ``'adagrad'``,
        ``'adagrad-da'``, ``'adam'``, ``'momentum'``, ``'proximal-sgd'``, ``'proximal-adagrad'``,
        ``'rms_prop'``, ``'sgd'``}

    early_stopping : bool, optional (default=False)
        If this is set to True, and the delta between the last cost and the new cost
        is less than ``min_change``, the network will stop fitting early.

    bias_strategy : str, optional (default='zeros')
        The strategy for initializing the bias vector. Default is 'zeros' and will
        initialize all bias values as zeros. The alternative is 'ones', which will
        initialize all bias values as ones.

    random_state : int, ``np.random.RandomState`` or None, optional (default=None)
        The numpy random state for seeding random TensorFlow variables in weight initialization.

    layer_type : str
        The type of layer, i.e., 'xavier'. This is the type of layer that
        will be generated. One of {'xavier', 'gaussian'}

    dropout : float, optional (default=1.0)
        Dropout is a mechanism to prevent over-fitting a network. Dropout functions
        by randomly dropping hidden units (and their connections) during training.
        This prevents units from co-adapting too much.

    l2_penalty : float or None, optional (default=0.0001)
        The L2 penalty (regularization term) parameter.

    eps : float, optional (default=1e-10)
        A small amount of noise to add to the loss to avoid a potential computation of
        ``log(0)``.

    gclip_min : float, optional (default=-5.)
        The minimum value at which to clip the gradient. Gradient clipping can be
        necessary for preventing vanishing or exploding gradients. Only necessary when
        ``clip`` is True.

    gclip_max : float, optional (default=5.)
        The maximum value at which to clip the gradient. Gradient clipping can be
        necessary for preventing vanishing or exploding gradients. Only necessary when
        ``clip`` is True.

    clip : bool, optional (default=True)
        Whether or not to clip the gradient in ``[gclip_min, gclip_max]``. Gradient
        clipping can be necessary for preventing vanishing or exploding gradients.


    Notes
    -----
    This class is based loosely on the following examples:
        * http://kvfrans.com/variational-autoencoders-explained/
        * http://jmetzen.github.io/2015-11-27/vae.html


    Attributes
    ----------
    topography_ : ``smrt.autoencode.layer._BaseSymmetricalTopography``
        The structure of hidden layers, weights and biases.

    train_cost_ : float
        The final cost as a result of the training procedure on the training examples.


    References
    ----------
    [1] http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

    [2] http://jmetzen.github.io/2015-11-27/vae.html

    [3] http://blog.fastforwardlabs.com/2016/08/12/introducing-variational-autoencoders-in-prose-and.html
    """
    def __init__(self, n_hidden, n_latent_factors, activation_function='sigmoid', learning_rate=0.05,
                 n_epochs=20, batch_size=128, min_change=1e-3, verbose=0, display_step=5, learning_function='rms_prop',
                 early_stopping=False, bias_strategy='zeros', random_state=None, layer_type='xavier',
                 dropout=DEFAULT_DROPOUT, l2_penalty=DEFAULT_L2, eps=1e-10, gclip_min=-5., gclip_max=5.,
                 clip=True):

        super(VariationalAutoEncoder, self).__init__(activation_function=activation_function,
                                                     learning_rate=learning_rate, n_epochs=n_epochs,
                                                     batch_size=batch_size, n_hidden=n_hidden,
                                                     min_change=min_change, verbose=verbose,
                                                     display_step=display_step,
                                                     learning_function=learning_function,
                                                     early_stopping=early_stopping,
                                                     bias_strategy=bias_strategy,
                                                     random_state=random_state,
                                                     layer_type=layer_type, dropout=dropout,
                                                     l2_penalty=l2_penalty, gclip_min=gclip_min,
                                                     gclip_max=gclip_max, clip=clip)

        # the only VAE-specific params
        self.n_latent_factors = n_latent_factors
        self.eps = eps

    @overrides(_SymmetricAutoEncoder)
    def _initialize_graph(self, X, y):
        # validate X, then make it into TF structure
        n_samples, n_features = X.shape

        # validate floats and other params...
        self._validate_for_fit()
        eps = _validate_float(self, 'eps')

        dropout = tf.placeholder_with_default(DEFAULT_DROPOUT, shape=[], name='dropout')  # make placeholder
        activation, learning_function = _validate_activation_optimization(self.activation_function,
                                                                          self.learning_function)

        # set up our weight matrix. This needs to be re-initialized for every fit, since (like sklearn)
        # we want to allow for model/transformer re-fits. IF we don't reinitialize, the next input
        # either gets a warm-start or a potentially already grand-mothered weight matrix.
        self.topography_ = SymmetricalVAETopography(X_placeholder=self.X_placeholder,
                                                    n_hidden=self.n_hidden,
                                                    input_shape=n_features,
                                                    activation=activation,
                                                    n_latent_factors=self.n_latent_factors,
                                                    layer_type=self.layer_type,
                                                    dropout=dropout,
                                                    bias_strategy=self.bias_strategy,
                                                    random_state=self.random_state)

        # Create the loss function optimizer. This dual-part loss function is adapted from code found at [2]
        # 1.) The reconstruction loss (the negative log probability of the input under the reconstructed
        #     Bernoulli distribution induced by the decoder in the data space). This can be interpreted as the number
        #     of "nats" required for reconstructing the input when the activation in latent is given.
        decoder = self.topography_.decode
        reconstruction_loss = cross_entropy(self.X_placeholder, decoder, eps=eps)

        # 2.) The latent loss, which is defined as the Kullback Leibler divergence between the distribution in
        #     latent space induced by the encoder on the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required for transmitting the the latent space
        #     distribution given the prior.
        latent_loss = kullback_leibler(self.topography_.z_mean_, self.topography_.z_log_sigma_)

        # define cost function and add regularization
        cost_function = tf.reduce_mean(reconstruction_loss + latent_loss, name='VAE_cost')
        cost_function = self._add_regularization(cost_function, self.topography_)

        # define the optimizer
        optimizer = self._clip_or_minimize(learning_function, self.learning_rate, cost_function)

        # return to actually do training
        return X, cost_function, optimizer, dropout

    @overrides(BaseAutoEncoder)
    def transform(self, X):
        """The encode task. Given input samples, ``X``, transform
        the observations using the inferential MLP.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input samples.
        """
        check_is_fitted(self, 'topography_')
        t = self.topography_
        return self.sess.run([t.z_mean_, t.z_log_sigma_], feed_dict={self.X_placeholder: X})

    @overrides(GenerativeMixin)
    def decode(self, X):
        """Given an encoded set of samples, ``X``, decode them from latent space

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input data.
        """
        check_is_fitted(self, 'topography_')
        t = self.topography_

        X = check_array(X, force_all_finite=True, dtype=NPDTYPE, ensure_2d=False)

        # z ~ N(0, I)
        return self.sess.run(t.decode, feed_dict={t.z_: X})

    @overrides(GenerativeMixin)
    def generate_from_sample(self, X, **nrm_args):
        """Given a sample, ``X``, encode and draw from the unit
        Gaussian posterior to generate the new examples.


        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            The input data.

        **nrm_args : dict, optional
            A keyword argument dictionary to be passed to the
            ``numpy.random.normal`` function.


        References
        ----------
        [1] https://github.com/fastforwardlabs/vae-tf/blob/master/vae.py#L210
        """
        check_is_fitted(self, 'topography_')

        X = check_array(X, force_all_finite=True, dtype=NPDTYPE, ensure_2d=True, accept_sparse=False)
        z_mu, log_sigma = self.encode(X)  # calls transform

        # sample from the unit Gaussian:
        rs = get_random_state(self.random_state)
        eps = rs.normal(size=log_sigma.shape, **nrm_args)
        z_mu += eps * np.exp(log_sigma)

        return self.decode(z_mu)

    @overrides(GenerativeMixin)
    def generate(self, n=1, **nrm_args):
        """Given a sample, ``X``, encode and draw from the unit
        Gaussian posterior to generate the new examples.


        Parameters
        ----------
        n : int, optional (default=1)
            The number of synthetic samples to create.

        **nrm_args : dict, optional
            A keyword argument dictionary to be passed to the
            ``numpy.random.normal`` function.


        References
        ----------
        [1] https://github.com/fastforwardlabs/vae-tf/blob/master/vae.py#L194
        """
        check_is_fitted(self, 'topography_')

        # see https://github.com/fastforwardlabs/vae-tf/blob/master/vae.py#L194
        z_mu = get_random_state(self.random_state).normal(size=(n, self.n_latent_factors), **nrm_args)
        return self.decode(z_mu)
