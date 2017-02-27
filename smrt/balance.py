# -*- coding: utf-8 -*-
#
# Authors: Taylor Smith <taylor.smith@alkaline-ml.com>
#          Jason White <jason.m.white5@gmail.com>
#
# The SMRT and SMOTE balancers

from __future__ import division, absolute_import, division
from sklearn.utils import column_or_1d
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array
from sklearn.neighbors import NearestNeighbors
from .autoencode import AutoEncoder
from .utils import get_random_state, validate_float
import numpy as np

__all__ = [
    'smote_balance',
    'smrt_balance'
]

DEFAULT_SEED = 42
MAX_N_CLASSES = 100  # max unique classes in y
MIN_N_SAMPLES = 2  # min allowed ever.


def _validate_X_y_ratio_classes(X, y, ratio):
    # validate the cheap stuff before copying arrays around...
    validate_float(ratio, 'balance_ratio')

    # validate arrays
    X = check_array(X, accept_sparse=False, dtype=np.float32)
    y = check_array(y, accept_sparse=False, ensure_2d=False, dtype=None)
    y = column_or_1d(y, warn=False)

    # get n classes in y, ensure they are <= MAX_N_CLASSES, but first ensure these are actually
    # class labels and not floats or anything...
    y_type = type_of_target(y)
    supported_types = {'multiclass', 'binary'}
    if y_type not in supported_types:
        raise ValueError('balancers only support %r, but got %r'
                         % ("(" + ', '.join(supported_types) + ")", y_type))

    present_classes, counts = np.unique(y, return_counts=True)
    n_classes = len(present_classes)

    # ensure <= MAX_N_CLASSES
    if n_classes > MAX_N_CLASSES:
        raise ValueError('balancers currently only support a maximum of %i '
                         'unique class labels, but %i were identified.' % (MAX_N_CLASSES, n_classes))

    # get the majority class label, and its count:
    majority_count_idx = np.argmax(counts, axis=0)
    majority_label, majority_count = present_classes[majority_count_idx], counts[majority_count_idx]
    target_count = max(int(ratio * majority_count), 1)

    #define a min_n_samples based on the sample ratio to max_class
    # required = {target_count - counts[i] for i, v in enumerate(present_classes) if v != majority_label}

    # THIS WAS OUR ORIGINAL LOGIC:
    #   * If there were any instances where the number of synthetic examples required for a class
    #     outweighed the number that existed in the class to begin with, we would end up having to
    #     potentially sample from the synthetic examples. We didn't want to have to do that.
    #
    # But it seems like a totally valid use-case. If we're detecting breast cancer, it might be a rare
    # event that needs lots of bolstering. We should allow that, even though we may discourage it.

    # if any counts < MIN_N_SAMPLES, raise:
    if any(i < MIN_N_SAMPLES for i in counts):
        raise ValueError('All label counts must be >= %i' % MIN_N_SAMPLES)

    return X, y, n_classes, present_classes, counts, majority_label, target_count


def _perturb(consider_vector, nearest, random_state):
    return (consider_vector - nearest) * random_state.rand(nearest.shape[0], 1) + consider_vector


def _interpolate(consider_vector, nearest, _):
    # the danger here is that if there are not enough samples, we'll interpolate
    # with the same value. Hrmm. Maybe add some entropy? # todo
    return np.average((consider_vector, nearest), axis=0)


# Define after the _perturb and _interpolate functions are defined
STRATEGIES = {
    'perturb': _perturb,
    'interpolate': _interpolate
}


def _nearest_neighbors_for_class(X, label, label_encoder, y_transform, majority_label, target_count, random_state,
                                 strategy, n_neighbors, algorithm, leaf_size, p, metric, metric_params, n_jobs):
    if strategy not in STRATEGIES:
        raise ValueError('strategy must be one of %r' % STRATEGIES)
    func = STRATEGIES[strategy]

    # start the iteration...
    if label == majority_label:
        return X, y_transform, None

    # transform the label, get the subset
    transformed_label = label_encoder.transform([label])[0]
    X_sub = X[y_transform == transformed_label, :]

    # if the count >= the ratio, skip this label
    count = X_sub.shape[0]
    if count >= target_count:
        return X, y_transform, None

    # define the nearest neighbors model
    model = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, p=p,
                             metric=metric, metric_params=metric_params, n_jobs=n_jobs)

    # get the observations that map to the transformed label
    amt_required = target_count - count

    # fit the model once, query the tree once. n_neighbors MUST
    # be ONE PLUS n_neighbors, since the zero'th index will always
    # be the index of the observation itself (i.e., obs 0 is its own
    # nearest neighbor).
    model.fit(X_sub)
    nearest = model.kneighbors(X_sub, n_neighbors=n_neighbors + 1, return_distance=False)
    indices = np.arange(count)

    # append the labels to y_transform - do this once to avoid the overhead of repeated concatenations
    y_transform = np.concatenate([y_transform, np.ones(amt_required, dtype=np.int16) * transformed_label])

    # sample while under the requisite ratio
    while amt_required > 0:
        # randomly select some observations, take first amt_required
        random_indices = random_state.permutation(indices)[:amt_required]

        # select the random sample, get nearest neighbors for the sample indices.
        synthetic = np.asarray([
            func(X_sub[consideration, :],
                 X_sub[nearest[consideration][1:], :],  # take 0th out (the observation vector under consideration)
                 random_state)
            for consideration in random_indices  # using the language of smote... ("vector under consideration")
        ])

        # append to X
        X = np.vstack([X, synthetic])

        # determine whether we need to recurse for this class (if there were too few samples)
        amt_required -= synthetic.shape[0]

    return X, y_transform, model


def smote_balance(X, y, return_estimators=False, balance_ratio=0.2, strategy='perturb', n_neighbors=5,
                  algorithm='kd_tree', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1,
                  seed=DEFAULT_SEED, shuffle=True, **kwargs):
    """Synthetic Minority Oversampling TEchnique (SMOTE) is a class balancing strategy that samples
    the k-Nearest Neighbors from each minority class, perturbs them with a random value between 0 and 1,
    and adds the difference between the original observation and the perturbation to the original observation
    to generate a synthetic observation. This is repeated until the minority classes are represented at
    a prescribed ratio to the majority class.

    Alternative methods involve interpolating the distance between the original observation and the
    nearest neighbors using either the median or the mean. This strategy can be set using the ``strategy``
    arg (one of 'perturb' or 'interpolate').

    Parameters
    ----------
    X : array-like, shape (n_samples, n_inputs)
        Training vectors as real numbers, where ``n_samples`` is the number of
        samples and ``n_inputs`` is the number of input features.

    y : array-like, shape (n_samples,)
        Training labels as integers, where ``n_samples`` is the number of samples.
        ``n_samples`` should be equal to the ``n_samples`` in ``X``.

    return_estimators : bool, optional (default=False)
        Whether or not to return the dictionary of fit :class:``sklearn.neighbors.NearestNeighbors`` instances.
        If True, the return value will be a tuple, with the first index being the balanced ``X`` matrix,
        the second index being the ``y`` values, and the third index being a dictionary of the fit estimators.
        If False, the return value is simply a tuple of the balanced ``X`` matrix and the corresponding labels.

    balance_ratio : float, optional (default=0.2)
        The minimum acceptable ratio of $MINORITY_CLASS : $MAJORITY_CLASS representation,
        where 0 < ``ratio`` <= 1

    strategy : str, optional (default='perturb')
        The strategy used to construct synthetic examples from existing examples. The original SMOTE
        paper suggests a strategy by which a random value between 0 and 1 scales the difference
        between the nearest neighbors, and the difference is then added to the original vector.
        This is the default strategy, 'perturb'. Valid strategies include:

          * 'perturb' - a random value between 0 and 1 scales the difference
             between the nearest neighbors, and the difference is then added
             to the original vector.

          * 'interpolate' - the ``interpolation_method`` ('mean' or 'median') of the nearest neighbors
            constitutes the synthetic example.

    n_neighbors : int, optional (default=5)
        Number of neighbors to use by default for ``kneighbors`` queries. This parameter
        is passed to each respective ``NearestNeighbors call.``

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional (default='kd_tree')
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use ``sklearn.neighbors.BallTree``
        - 'kd_tree' will use ``sklearn.neighbors.KDtree``
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to ``fit`` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force. This parameter
        is passed to each respective ``NearestNeighbors call.``

    leaf_size : int, optional (default=30)
        Leaf size passed to ``BallTree`` or ``KDTree``.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    p : integer, optional (default=2)
        Parameter for the Minkowski metric from
        ``sklearn.metrics.pairwise.pairwise_distances``. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric : string or callable, optional (default='minkowski')
        Metric to use for distance computation. Any metric from scikit-learn
        or ``scipy.spatial.distance`` can be used.

        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.
        Distance matrices are not supported.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
        - from ``scipy.spatial.distance``: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
          'sqeuclidean', 'yule']

        See the documentation for ``scipy.spatial.distance`` for details on these
        metrics.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Affects only ``kneighbors`` and ``kneighbors_graph`` methods.

    shuffle : bool, optional (default=True)
        Whether to shuffle the output.

    seed : int, optional (default=42)
        An integer. Used to create a random seed for the random minority class selection.
    """
    # validate the cheap stuff before copying arrays around...
    X, y, n_classes, present_classes, \
    counts, majority_label, target_count = _validate_X_y_ratio_classes(X, y, balance_ratio)

    random_state = get_random_state(seed)

    # encode y, in case they are not numeric (we need them to be for np.ones)
    le = LabelEncoder()
    le.fit(present_classes)
    y_transform = le.transform(y)  # make numeric

    # get the nearest neighbor models
    models = dict()
    for label in present_classes:
        # X and y_transform will progressively be updated throughout the runs
        X, y_transform, models[label] = _nearest_neighbors_for_class(X=X, label=label, label_encoder=le,
                                                                     y_transform=y_transform,
                                                                     majority_label=majority_label,
                                                                     target_count=target_count,
                                                                     random_state=random_state,
                                                                     strategy=strategy, n_neighbors=n_neighbors,
                                                                     algorithm=algorithm, leaf_size=leaf_size,
                                                                     p=p, metric=metric, metric_params=metric_params,
                                                                     n_jobs=n_jobs)

    # now that X, y_transform have been assembled, inverse_transform the y_t back to its original state:
    y = le.inverse_transform(y_transform)

    # finally, shuffle both and return
    output_order = np.arange(X.shape[0])
    if shuffle:
        output_order = random_state.permutation(output_order)

    if return_estimators:
        return X[output_order, :], y[output_order], models
    return X[output_order, :], y[output_order]


def smrt_balance(X, y, return_estimators=False, balance_ratio=0.2, strategy='perturb', min_error_sample=0.25,
                 activation_function='relu', learning_rate=0.05, n_epochs=200, batch_size=256, n_hidden=None,
                 compression_ratio=0.6, min_change=1e-6, verbose=0, display_step=5, seed=DEFAULT_SEED,
                 shuffle=True, smote_args={}, **kwargs):
    """SMRT (Sythetic Minority Reconstruction Technique) is the younger, more sophisticated cousin to
    SMOTE (Synthetic Minority Oversampling TEchnique). Using auto-encoders, SMRT learns the parameters
    that best reconstruct the observations in each minority class, and then generates synthetic observations
    until the minority class is represented at a minimum of ``balance_ratio`` * majority_class_size. 

    SMRT avoids one of SMOTE's greatest risks: In SMOTE, when drawing random observations from whose k-nearest
    neighbors to reconstruct, the possibility exists that a "border point," or an observation very close to 
    the decision boundary may be selected. This could result in the synthetically-generated observations lying 
    too close to the decision boundary for reliable classification, and could lead to the degraded performance
    of an estimator. SMRT avoids this risk by ranking observations according to their reconstruction MSE, and
    drawing samples to reconstruct from the lowest-MSE observations (i.e., the most "phenotypical" of a class).
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_inputs)
        Training vectors as real numbers, where ``n_samples`` is the number of
        samples and ``n_inputs`` is the number of input features.

    y : array-like, shape (n_samples,)
        Training labels as integers, where ``n_samples`` is the number of samples.
        ``n_samples`` should be equal to the ``n_samples`` in ``X``.

    return_estimators : bool, optional (default=False)
        Whether or not to return the dictionary of fit :class:``smrt.autoencode.AutoEncoder`` instances.
        If True, the return value will be a tuple, with the first index being the balanced
        ``X`` matrix, the second index being the ``y`` values, and the third index being a 
        dictionary of the fit encoders. If False, the return value is simply the balanced ``X`` 
        matrix and the corresponding labels.

    balance_ratio : float, optional (default=0.2)
        The minimum acceptable ratio of $MINORITY_CLASS : $MAJORITY_CLASS representation,
        where 0 < ``ratio`` <= 1

    min_error_sample : float, optional (default=0.25)
        The ratio of the existing minority records from which to sample. Selects the lowest ``min_error_sample``
        percent (by error) of records for reconstruction.

    activation_function : str or callable, optional (default='relu')
        The activation function. If a str, it should be one of PERMITTED_ACTIVATIONS. If a
        callable, it should be an activation function contained in the ``tensorflow.nn`` module.

    learning_rate : float, optional (default=0.05)
        The algorithm learning rate.

    n_epochs : int, optional (default=20)
        An epoch is one forward pass and one backward pass of *all* training examples. ``n_epochs``,
        then, is the number of full passes over the training data. The algorithm will stop early if
        the cost delta between iterations diminishes below ``min_change`` between epochs.

    batch_size : int, optional (default=256)
        The number of training examples in a single forward/backward pass. As ``batch_size``
        increases, the memory required will also increase.

    n_hidden : int, list or dictionary , optional (default=None)
        The hidden layer structure. If an int is provided, a single hidden layer is constructed,
        with ``n_hidden`` neurons. If ``n_hidden`` is an iterable, ``len(n_hidden)`` hidden layers
        are constructed, with as many neurons as correspond to each index, respectively. If no
        value is passed for ``n_hidden`` (default), the ``AutoEncoder`` defaults to a single hidden
        layer of ``compression_ratio * n_features`` in order to force the network to learn a compressed
        feature space.

    compression_ratio : float, optional (default=0.6)
        If no value is passed for ``n_hidden`` (default), the ``AutoEncoder`` defaults to a single hidden
        layer of ``compression_ratio * n_features`` in order to force the network to learn a compressed
        feature space. Default ``compression_ratio`` is 0.6.

    min_change : float, optional (default=1e-6)
        An early stopping criterion. If the delta between the last cost and the new cost
        is less than ``min_change``, the network will stop fitting early.

    verbose : int, optional (default=0)
        The level of verbosity. If 0, no stdout will be produced. Varying levels of
        output will increase with an increasing value of ``verbose``.

    display_step : int, optional (default=5)
        The interval of epochs at which to update the user if ``verbose`` mode is enabled.

    seed : int, optional (default=42)
        An integer. Used to create a random seed for the weight and bias initialization.

    shuffle : bool, optional (default=True)
        Whether to shuffle the output.
    """
    # validate the cheap stuff before copying arrays around...
    X, y, n_classes, present_classes, \
    counts, majority_label, target_count = _validate_X_y_ratio_classes(X, y, balance_ratio)

    random_state = get_random_state(seed)
    validate_float(min_error_sample, 'min_error_sample')

    # encode y, in case they are not numeric
    le = LabelEncoder()
    le.fit(present_classes)
    y_transform = le.transform(y)  # make numeric (we need them to be for np.ones)

    # create X copy on which to append. We do this because we do not want to augment
    # synthetic examples of already-reconstructed examples...
    X_copy = X[:, :]

    # start the iteration...
    encoders = dict()  # map the label to the fit encoder
    for i, label in enumerate(present_classes):
        if label == majority_label:
            continue

        # if the count >= the ratio, skip this label
        count = counts[i]
        if count >= target_count:
            encoders[label] = None
            continue

        # fit the autoencoder
        encoder = AutoEncoder(activation_function=activation_function, learning_rate=learning_rate, n_epochs=n_epochs,
                              batch_size=batch_size, n_hidden=n_hidden, compression_ratio=compression_ratio,
                              min_change=min_change, verbose=verbose, display_step=display_step, seed=seed,
                              **kwargs)

        # transform label
        transformed_label = le.transform([label])[0]
        X_sub = X[y_transform == transformed_label, :]

        # fit the model, store it
        encoder.fit(X_sub)
        encoders[label] = encoder

        # transform X_sub, rank it
        reconstructed = encoder.feed_forward(X_sub)
        mse = np.asarray([
            mean_squared_error(X_sub[i, :], reconstructed[i, :])
            for i in range(X_sub.shape[0])
        ])

        # rank order:
        mse_order = np.argsort(mse)
        ordered = X_sub[mse_order, :]  # order asc by reconstr error
        reconstructed_ordered = reconstructed[mse_order, :]

        # sample_count = target_count - X_sub.shape[0]  # todo: redo this--pull a subset of these..
        sample_count = int(round(min_error_sample * X_sub.shape[0]))  # the num rows to select from bottom

        # the number of obs we need
        obs_req = target_count - X_sub.shape[0]

        # sample while under the requisite ratio
        while obs_req > 0:  # shouldn't be less than... but just in case

            # if obs_req is lower than the sample_count, go with the min
            sample_count = min(obs_req, sample_count)
            perturb_sample = ordered[:sample_count]  # makes a copy we can perturb
            reconst_sample = reconstructed_ordered[:sample_count]  # the corresponding reconstruction samples

            # perturb the sample. Subtract the reconstruction matrix from the original sample,
            # then We create a random matrix, M x N, (bound in [0, 1]), multiply it by the difference matrix,
            # and add it to the original sample.
            perturb_sample += ((perturb_sample - reconst_sample) * random_state.rand(*perturb_sample.shape))

            # append to X, y_transform
            X_copy = np.vstack([X_copy, perturb_sample])
            y_transform = np.concatenate([y_transform,
                                          np.ones(sample_count, dtype=np.int16) * transformed_label])

            # determine whether we need to recurse for this class (if there were too few samples)
            # update the required amount
            obs_req -= sample_count

    # now that X, y_transform have been assembled, inverse_transform the y_t back to its original state:
    y = le.inverse_transform(y_transform)

    # finally, shuffle both and return
    if shuffle:
        output_order = random_state.shuffle(np.arange(X_copy.shape[0]))
    else:
        output_order = np.arange(X_copy.shape[0])

    if return_estimators:
        return X_copy[output_order, :], y[output_order], encoders
    return X_copy[output_order, :], y[output_order]
