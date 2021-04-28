import logging
import os
import pickle

import gin
import numpy as np
import tensorflow as tf
from backend.ml_model.helpers import choice_or_all, arr_of_dicts_to_dict_of_arrays, convert_to_tf
from backend.ml_model.preference_aggregation import PreferencePredictor, MedianPreferenceAggregator
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from backend.ml_model.preference_aggregation import print_memory, tqdmem

tf.compat.v1.enable_eager_execution()

Adam = gin.external_configurable(tf.keras.optimizers.Adam)
SGD = gin.external_configurable(tf.keras.optimizers.SGD)
Zeros = gin.external_configurable(tf.keras.initializers.Zeros)


def variable_index_layer_call(tensor, inputs):
    """Get variable at indices."""
    return tf.gather_nd(tensor, inputs)


@gin.configurable
class VariableIndexLayer(tf.keras.layers.Layer):
    """Layer which outputs a trainable variable on a call."""

    NEED_INDICES = False

    def __init__(self, shape, name="index_layer", initializer=None, **kwargs):
        super(VariableIndexLayer, self).__init__(name=name)
        self.v = self.add_weight(
            shape=shape, initializer=initializer,
            trainable=True, name="var/" + name,
            dtype=tf.keras.backend.floatx())

    @tf.function
    def call(self, inputs, **kwargs):
        # print("INPUT SHAPE", inputs.shape, "WEIGHT SHAPE", self.v.shape)

        out = variable_index_layer_call(self.v, inputs)
        # print("INPUT SHAPE", inputs.shape, "WEIGHT SHAPE", self.v.shape, "OUT SHAPE", out.shape)
        return out

class HashMap(object):
    """Maps objects from keys to values and vice-versa."""
    def __init__(self):
        self.key_to_value = {}
        self.value_to_key = {}
        
    def set(self, key, value):
        self.key_to_value[key] = value
        self.value_to_key[value] = key
        
    def get_value(self, key):
        return self.key_to_value[key]
    
    def get_key(self, value):
        return self.value_to_key[value]
    
    def get_keys(self, values):
        """Get multiple keys given values. Can be parallelized!"""
        return [self.get_key(value) for value in values]

@gin.configurable
class SparseVariableIndexLayer(tf.keras.layers.Layer):
    """Layer which outputs a trainable variable on a call, sparse storage."""
    
    NEED_INDICES = True

    def __init__(self, shape, indices_list, name="index_layer", initializer=None):
        super(SparseVariableIndexLayer, self).__init__(name=name)
        
        assert isinstance(indices_list, list), "indices_list must be a list"
        assert indices_list, "List of indices must be non-empty!"
        assert all([isinstance(x, tuple) for x in indices_list]), \
          "all items in indices_list must be tuples"
          
        # checking tuple length
        assert [len(shape) == len(idx) for idx in indices_list], "All tuple lengths must be equal"
        
        self.num_items = len(indices_list)
        # tf.sparse.SparseTensor didn't work with GradientTape, so using own implementation
        self.v = self.add_weight(shape=(self.num_items,), initializer=initializer,
                                 trainable=True, name="var_sparse_values/" + name,
                                 dtype=tf.keras.backend.floatx())
        self.idx = HashMap()
        
        # storing indices
        # TODO: use vectorized implementation
        for i, idx in enumerate(indices_list):
            self.idx.set(i, idx)


    def call(self, inputs, **kwargs):
        # print("INPUT SHAPE", inputs.shape, "WEIGHT SHAPE", self.v.shape)
        
        indices_flat = self.idx.get_keys(inputs)
        return tf.gather(self.v, indices_flat)


@gin.configurable
class AllRatingsWithCommon(object):
    COMMON_EXPERT = "__aggregate_expert__"

    """Stores a tensor will ALL ratings, including the common model."""

    def __init__(
            self,
            experts,
            objects,
            output_features,
            name,
            var_init_cls=VariableIndexLayer):

        print_memory('ARWC:init')

        # experts
        self.name = name
        self.experts = list(experts) + [self.COMMON_EXPERT]
        self.experts_set = set(self.experts)
        self.aggregate_index = len(self.experts) - 1
        assert len(
            self.experts) == len(
            self.experts_set), "Duplicate experts are not allowed"
        self.experts_reverse = {
            expert: i for i, expert in enumerate(self.experts)}

        # features
        self.output_features = list(output_features)
        self.output_dim = len(output_features)

        # objects
        self.objects = list(objects)
        self.objects_set = set(self.objects)
        assert len(
            self.objects_set) == len(
            self.objects), "Duplicate objects are not allowed."
        self.objects_reverse = {obj: i for i, obj in enumerate(self.objects)}

        # outputs
        self.layer = None
        self.var_init_cls = var_init_cls
        self.indices_list = []
        self.variables = []

        self.reset_model()
        
    def add_indices(self, indices):
        self.indices_list += indices

    def reset_model(self):
        if self.var_init_cls.NEED_INDICES and not self.indices_list:
            logging.warning("Variable needs indices, and they are not yet available. Skip init...")
            return
        
        logging.warning("Indices: %d" % len(self.indices_list))
        
        self.layer = self.var_init_cls(
            shape=(len(self.experts), len(self.objects), self.output_dim),
            indices_list=self.indices_list)
        self.variables = [self.layer.v]
        # print(self.output_dim)

    def _save_path(self, directory):
        path = os.path.join(
            directory,
            f'{self.name}_alldata_featureless_onetensor.pkl')
        return path

    def save(self, directory):
        """Save weights."""
        result = {
            'name': self.name,
            'experts': self.experts,
            'objects': self.objects,
            'data': self.layer.v.numpy(),
            'features': self.output_features,
        }
        assert result['data'].shape == (len(result['experts']), len(
            result['objects']), len(result['features']))
        path = self._save_path(directory=directory)
        pickle.dump(result, open(path, 'wb'))

    def load(self, directory):
        """Load weights."""

        # print("Load weights")

        print_memory('ARWC:load_start')

        path = self._save_path(directory=directory)
        result = pickle.load(open(path, 'rb'))

        print_memory('ARWC:pickle_loaded')

        # setting zero weights
        self.reset_model()

        print_memory('ARWC:model_reset_loaded')

        old_object_indices = {
            obj: idx for idx, obj in enumerate(
                result['objects'])}
        old_feature_indices = {
            feature: idx for idx, feature in enumerate(
                result['features'])}
        old_expert_indices = {
            expert: idx for idx, expert in enumerate(
                result['experts'])}

        restored_items = 0

        
        print_memory('ARWC:old_indices_loaded')
        # print("experts", len(self.experts), "objects", len(self.objects),
        #       "features", len(self.output_features))

        to_assign_idx = []
        to_assign_vals = []
        
        print_memory('ARWC:start_assign_append_loop')

        for new_expert_idx, expert in enumerate(tqdmem(self.experts, desc="rating_load_expert_loop",
                                             leave=True)):
            old_expert_idx = old_expert_indices.get(expert, None)
            for new_obj_idx, obj in enumerate(tqdmem(self.objects, desc="rating_load_object_loop",
                                              leave=False, disable=True)):
                old_obj_idx = old_object_indices.get(obj, None)
                for new_f_idx, feature in enumerate(self.output_features):
                    old_f_idx = old_feature_indices.get(feature, None)

                    if all([x is not None for x in [old_expert_idx, old_obj_idx, old_f_idx]]):
                        val = result['data'][old_expert_idx, old_obj_idx, old_f_idx]

                        if not np.isnan(val):
                            to_assign_idx.append((new_expert_idx, new_obj_idx, new_f_idx))
                            to_assign_vals.append(val)
                            restored_items += 1
                            
        print_memory('ARWC:finish_assign_append_loop')

        if to_assign_idx:
            print_memory('ARWC:start_create_layer_variable')
            self.layer.v = tf.Variable(
                tf.tensor_scatter_nd_update(self.layer.v,
                                            to_assign_idx,
                                            to_assign_vals),
                trainable=True)
            print_memory('ARWC:finish_create_layer_variable')

        # print(to_assign_idx, to_assign_vals)
        # print(self.layer.v)
        
        print_memory('ARWC:alive')

        return {'restored_items': restored_items}

    def get_internal_ids(self, objects, experts):
        """Get integer IDs for objects."""
        assert len(objects) == len(experts)
        assert all([expert in self.experts_set for expert in experts])
        assert all([obj in self.objects_set for obj in objects])
        return [(self.experts_reverse[expert], self.objects_reverse[obj])
                for expert, obj in zip(experts, objects)]


@gin.configurable
class FeaturelessPreferenceLearningModel(PreferencePredictor):
    """A model for preference learning."""

    def __init__(self, expert=None, all_ratings=None):
        assert isinstance(all_ratings, AllRatingsWithCommon)
        self.all_ratings = all_ratings
        self.objects = self.all_ratings.objects
        self.used_object_feature_ids = set()
        assert expert in self.all_ratings.experts
        self.expert = expert
        self.expert_id = self.all_ratings.experts_reverse[expert]
        self.output_features = self.all_ratings.output_features
        self.clear_data()
        self.ratings = []
        self.input_dim = -1
        self.output_dim = len(self.output_features)
        super(
            FeaturelessPreferenceLearningModel,
            self).__init__(
            model=self.__call__)

    def __call__(self, objects):
        internal_ids = self.all_ratings.get_internal_ids(
            objects, experts=[self.expert] * len(objects))
        result = self.all_ratings.model(
            np.array(internal_ids, dtype=np.int64))
        return result

    def ratings_with_object(self, obj):
        """Number of ratings with a particular object."""
        M_ev = 0
        for r in self.ratings:
            if r['o1'] == obj:
                M_ev += 1
            elif r['o2'] == obj:
                M_ev += 1
        return M_ev

    def clear_data(self):
        """Remove all preferences from the training dataset."""
        self.ratings = []

    def register_preference(self, o1, o2, p1_vs_p2, weights):
        """Save data when preference is available."""
        assert p1_vs_p2.shape == (self.output_dim,), "Wrong preference shape."
        assert o1 in self.objects, "Wrong object %s" % o1
        assert o2 in self.objects, "Wrong object %s" % o2
        val = p1_vs_p2
        val_no_nan = val[~np.isnan(val)]
        assert np.all(
            val_no_nan >= 0) and np.all(
            val_no_nan <= 1), "Preferences should be in [0,1]"

        # indices for features that are used
        used_feature_indices = [i for i in range(len(weights)) if weights[i] > 0]
        
        for f_idx in used_feature_indices:
            self.used_object_feature_ids.add((self.all_ratings.objects_reverse[o1], f_idx))
            self.used_object_feature_ids.add((self.all_ratings.objects_reverse[o2], f_idx))

        self.ratings.append({'o1': o1, 'o2': o2,
                             'ratings': 2 * (np.array(p1_vs_p2) - 0.5),
                             'weights': weights})
            
    def on_dataset_end(self):
        #print("Expert ID", self.expert_id)
        #print("Used objects", self.used_object_feature_ids)
        
        if self.expert_id == self.all_ratings.aggregate_index:
            # common model
            indices = [(self.expert_id, obj_id, feature_id)
                       for obj_id in self.all_ratings.objects_reverse.values()
                       for feature_id in range(self.all_ratings.output_dim)]
        else:
            # individual model, order is fixed here
            indices = [(self.expert_id, obj_id, feature_id)
                       for obj_id, feature_id in self.used_object_feature_ids]
        
        #print("Appending indices", indices)
        
        self.all_ratings.add_indices(indices)


@tf.function(experimental_relax_shapes=True)
def loss_fcn_gradient_hessian(video_indices, **kwargs):
    """Compute the loss function, gradient and the Hessian."""
    variable = kwargs['model_tensor']
    loss = loss_fcn(**kwargs)['loss']
    g = tf.gradients(loss, variable)[0]
    g = tf.gather(g, axis=1, indices=video_indices)
    h = tf.hessians(loss, variable)[0]
    h = tf.gather(h, axis=1, indices=video_indices)
    h = tf.gather(h, axis=4, indices=video_indices)
    return {'loss': loss, 'gradient': g, 'hessian': h}


@tf.function(experimental_relax_shapes=True)
def transform_grad(g):
    """Replace nan with 0."""
    idx_non_finite = tf.where(~tf.math.is_finite(g))
    zeros = tf.zeros(len(idx_non_finite), dtype=g.dtype)
    return tf.tensor_scatter_nd_update(g, idx_non_finite, zeros)


@gin.configurable
class FeaturelessMedianPreferenceAverageRegularizationAggregator(MedianPreferenceAggregator):
    """Aggregate with median, train with an average set of parameters."""

    def __init__(
            self,
            models,
            epochs=20,
            optimizer=Adam(),
            hypers=None,
            callback=None,
            batch_params=None):
        assert models, "Models cannot be empty."
        assert all([isinstance(m, FeaturelessPreferenceLearningModel)
                    for m in models]), "Wrong model type."
        super(
            FeaturelessMedianPreferenceAverageRegularizationAggregator,
            self).__init__(models)

        self.all_ratings = self.models[0].all_ratings
        self.losses = []
        self.optimizer = optimizer
        self.epochs = epochs
        self.hypers = hypers if hypers else {}
        self.hypers['aggregate_index'] = self.all_ratings.aggregate_index
        self.batch_params = batch_params if batch_params else {}
        self.callback = callback

        # creating the optimized tf loss function
        assert len(self.all_ratings.variables) == 1, "Only support 1-variable models!"
        self.loss_fcn = self.build_loss_fcn(**self.hypers)

        self.minibatch = None

    def __call__(self, x):
        """Return predictions for the s_qv."""
        ids = self.all_ratings.get_internal_ids(
            x, experts=[self.all_ratings.COMMON_EXPERT] * len(x))
        result = self.all_ratings.model(np.array(ids, dtype=np.int64))
        return result

    def loss_fcn_kwargs(self, **kwargs):
        """Get keyword arguments for the loss_fcn function."""
        kwargs_subset = {'model_tensor', 'aggregate_index', 'lambda_', 'mu', 'C',
                         'default_score_value'}
        if 'ignore_vals' in kwargs:
            kwargs_subset = kwargs_subset.difference(kwargs['ignore_vals'])
        kwargs_subset_dict = {}
        for k in kwargs_subset.intersection(kwargs.keys()):
            val = kwargs[k]
            if isinstance(val, int):
                val = tf.constant(val, dtype=tf.int64)
            elif isinstance(val, float):
                val = tf.constant(val, dtype=tf.float32)
            kwargs_subset_dict[k] = val

        return kwargs_subset_dict

    def build_loss_fcn(self, **kwargs):
        """Create the loss function."""
        kwargs0 = self.loss_fcn_kwargs(**kwargs)

        def fcn(**kwargs1):
            if 'model_tensor' not in kwargs0:
                kwargs0['model_tensor'] = self.all_ratings.variables[0]
            return loss_fcn(**kwargs0, **kwargs1)

        return fcn

    def save(self, directory):
        self.all_ratings.save(directory)

    def load(self, directory):
        try:
            print_memory('FMPARA:pre_rating_load')
            result = self.all_ratings.load(directory)
            print_memory('FMPARA:post_rating_load')
        except FileNotFoundError as e:
            logging.warning(f"No model restore data {e}")
            result = {'status': str(e)}
        return result

    def plot_loss(self, *args, **kwargs):
        """Plot the losses."""
        losses = arr_of_dicts_to_dict_of_arrays(self.losses)

        plt.figure(figsize=(15, 10))
        plot_h = len(losses)

        for i, key in enumerate(losses.keys(), 1):
            plt.subplot(1, plot_h, i)
            plt.title(key)
            plt.plot(losses[key])

    def fit(self, epochs=None, callback=None):
        """Fit with multiple epochs."""
        if epochs is None:
            epochs = self.epochs
        if callback is None:
            callback = self.callback
        with tqdmem(total=epochs, desc="fit_loop") as pbar:
            for i in range(epochs):
                if i % self.hypers.get('sample_every', 1) == 0:
                    self.minibatch = self.sample_minibatch(**self.batch_params)

                losses = self.fit_step()
                self.losses.append(losses)
                pbar.set_postfix(**losses)  # loss=losses['loss']
                pbar.update(1)
                if callable(callback):
                    callback(self, epoch=i, **losses)
        if self.losses:
            return self.losses[-1]
        else:
            return {}

    def sample_minibatch(
            self,
            sample_experts=50,
            sample_ratings_per_expert=50,
            sample_objects_per_expert=50):
        """Get one mini-batch.

        Args:
            sample_experts: number of experts to sample at each mini-batch
                for RATINGS/REGULARIZATION
            sample_ratings_per_expert: number of ratings to sample at each mini-batch for RATINGS
            sample_objects_per_expert: number of objects to sample at each mini-batch
                for REGULARIZATION

        Returns:
            dict with tensors
        """
        # sampled mini-batch
        experts_rating, objects_rating_v1, objects_rating_v2, cmp, weights = [], [], [], [], []
        experts_all, objects_all, num_ratings_all = [], [], []
        # sampled_objects = []

        # sampling the mini-batch: ratings
        # ignoring the common expert here
        experts_to_sample = self.all_ratings.experts[:-1]
        sampled_experts = choice_or_all(experts_to_sample, sample_experts)

        # print("EXPERTS", experts_all, sampled_experts)

        for expert in sampled_experts:
            expert_id = self.all_ratings.experts_reverse[expert]
            # print("EXPERT", expert, self.models)
            sampled_ratings = choice_or_all(
                self.models[expert_id].ratings,
                sample_ratings_per_expert)
            for rating in sampled_ratings:
                # format: ({'o1': o1, 'o2': o2, 'ratings': p1_vs_p2, 'weights': weights})

                experts_rating.append(expert_id)
                objects_rating_v1.append(
                    self.all_ratings.objects_reverse[rating['o1']])
                objects_rating_v2.append(
                    self.all_ratings.objects_reverse[rating['o2']])
                cmp.append(rating['ratings'])
                weights.append(rating['weights'])

        # sampling the mini-batch: regularization
        sampled_objects = choice_or_all(
            self.all_ratings.objects,
            sample_objects_per_expert)

        # print("EXPERTS", experts_all, sampled_experts)

        if hasattr(self, 'certification_status') and len(
                self.certification_status) == len(experts_to_sample):
            certified_experts = [x for i, x in enumerate(
                experts_to_sample) if self.certification_status[i]]
        else:
            logging.warning("List of certified experts not found")
            certified_experts = experts_to_sample

        sampled_certified_experts = choice_or_all(
            certified_experts, sample_experts)

        # print("SAMPLED CERTIFIED", sampled_certified_experts)

        for expert in sampled_certified_experts:
            # print(expert, self.all_ratings.experts_reverse)

            expert_id = self.all_ratings.experts_reverse[expert]
            for obj in sampled_objects:
                object_id = self.all_ratings.objects_reverse[obj]
                experts_all.append(expert_id)
                objects_all.append(object_id)
                num_ratings_all.append(
                    self.models[expert_id].ratings_with_object(obj))

        # videos to plug into the "common_to_1" loss term
        sampled_object_ids = [self.all_ratings.objects_reverse[obj]
                              for obj in sampled_objects]

        # total number of ratings
        # total_ratings = sum([len(m.ratings) for m in self.models])

        if not experts_rating:
            logging.warning("No data to train.")
            return {}

        kwargs = {
            'experts_rating': np.array(experts_rating, dtype=np.int64),
            'objects_rating_v1': np.array(objects_rating_v1, dtype=np.int64),
            'objects_rating_v2': np.array(objects_rating_v2, dtype=np.int64),
            'cmp': np.array(cmp, dtype=np.float32),
            'weights': np.array(weights, dtype=np.float32),
            'experts_all': np.array(experts_all, dtype=np.int64),
            'objects_all': np.array(objects_all, dtype=np.int64),
            'num_ratings_all': np.array(num_ratings_all, dtype=np.float32),
            'objects_common_to_1': np.array(sampled_object_ids, dtype=np.int64)
        }

        kwargs = convert_to_tf(kwargs)

        # print(kwargs)

        return kwargs
    
    def fit_step(self):
        """Fit using the custom magic loss...

        Returns:
            dict of losses (numpy)
        """
        optimizer = self.optimizer
        minibatch = self.minibatch

        if not minibatch:
            return {}

        # mem: +32mb

        # computing the custom loss
        with tf.GradientTape() as tape:
            losses = self.loss_fcn(**minibatch)

        # mem: +135mb

        # doing optimization (1 step)
        all_variables = self.all_ratings.variables
        grads = tape.gradient(losses['loss'], all_variables,
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # mem: +180mb
        
        grads = [transform_grad(g) for g in grads]

        # mem: +250mb

        optimizer.apply_gradients(zip(grads, all_variables))

        # mem: +500mb
        
        out = {}
        out.update(losses)
        out.update({f"grad{i}": tf.linalg.norm(g)
                    for i, g in enumerate(grads)})

        # returning original losses
        return {x: y.numpy() for x, y, in out.items()}
