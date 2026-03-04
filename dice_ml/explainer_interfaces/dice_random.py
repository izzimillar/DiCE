
"""
Module to generate diverse counterfactual explanations based on random sampling.
A simple implementation.
"""
import random
import timeit

import numpy as np
import pandas as pd

from dice_ml import diverse_counterfactuals as exp
from dice_ml.constants import ModelTypes
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase


class DiceRandom(ExplainerBase):

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        """
        super().__init__(data_interface)  # initiating data related parameters

        self.model = model_interface
        self.model.load_model()  # loading pickled trained model if applicable
        self.model.transformer.feed_data_params(data_interface)
        self.model.transformer.initialize_transform_func()

        self.precisions = self.data_interface.get_decimal_precisions(output_type="dict")
        if self.data_interface.outcome_name in self.precisions:
            self.outcome_precision = [self.precisions[self.data_interface.outcome_name]]
        else:
            self.outcome_precision = 0

    def _generate_counterfactuals(self, query_instance, total_CFs, desired_range=None,
                                  desired_class="opposite", permitted_range=None,
                                  features_to_vary="all", stopping_threshold=0.5, posthoc_sparsity_param=0.1,
                                  posthoc_sparsity_algorithm="linear", sample_size=1000, random_seed=None, verbose=False,
                                  limit_steps_ls=10000, causal_constraints=None):
        """Generate counterfactuals by randomly sampling features.

        :param query_instance: Test point of interest. A dictionary of feature names and values or a single row dataframe.
        :param total_CFs: Total number of counterfactuals required.
        :param desired_range: For regression problems. Contains the outcome range to generate counterfactuals in.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome
                              class of query_instance for binary classification.
        :param permitted_range: Dictionary with feature names as keys and permitted range in list as values.
                                Defaults to the range inferred from training data. If None, uses the parameters
                                initialized in data_interface.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large
                                           (for instance, income varying from 10k to 1000k) and only if the features
                                           share a monotonic relationship with predicted outcome in the model.
        :param sample_size: Sampling size
        :param random_seed: Random seed for reproducibility
        :param limit_steps_ls: Defines an upper limit for the linear search step in the posthoc_sparsity_enhancement
        :param causal_constraints: Defines causal constraints between features in the form of a dictionary with type 
                                of constraint as key and list of features names as values. All possible types of constraint: 
                                cannot_increase, cannot_decrease, must_increase_with, must_decrease_with, 
                                inverse_with_increase, inverse_with_decrease.

        :returns: A CounterfactualExamples object that contains the dataframe of generated counterfactuals as an attribute.
        """
        self.features_to_vary = self.setup(features_to_vary, permitted_range, query_instance, feature_weights=None)

        self.causal_constraints = self.validate_constraints(causal_constraints)
        
        if features_to_vary == "all":
            self.fixed_features_values = {}
        else:
            self.fixed_features_values = {}
            for feature in self.data_interface.feature_names:
                if feature not in features_to_vary:
                    self.fixed_features_values[feature] = query_instance[feature].iat[0]

        # Do predictions once on the query_instance and reuse across to reduce the number
        # inferences.
        model_predictions = self.predict_fn(query_instance)
        # number of output nodes of ML model
        self.num_output_nodes = None
        if self.model.model_type == ModelTypes.Classifier:
            self.num_output_nodes = model_predictions.shape[1]

        # query_instance need no transformation for generating CFs using random sampling.
        # find the predicted value of query_instance
        test_pred = model_predictions[0]
        if self.model.model_type == ModelTypes.Classifier:
            self.target_cf_class = self.infer_target_cfs_class(desired_class, test_pred, self.num_output_nodes)
        elif self.model.model_type == ModelTypes.Regressor:
            self.target_cf_range = self.infer_target_cfs_range(desired_range)
        # fixing features that are to be fixed
        self.total_CFs = total_CFs

        self.stopping_threshold = stopping_threshold
        if self.model.model_type == ModelTypes.Classifier:
            # TODO Generalize this for multi-class
            if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
                self.stopping_threshold = 0.25
            elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
                self.stopping_threshold = 0.75

        # get random samples for each feature independently
        start_time = timeit.default_timer()
        random_instances = self.get_samples(
            self.fixed_features_values,
            self.feature_range, 
            query_instance=query_instance,
            causal_constraints=self.causal_constraints,
            sampling_random_seed=random_seed, 
            sampling_size=sample_size, 
        )
        # Generate copies of the query instance that will be changed one feature
        # at a time to encourage sparsity.
        cfs_df = None
        candidate_cfs = pd.DataFrame(
            np.repeat(query_instance.values, sample_size, axis=0), columns=query_instance.columns)
        
        # Loop to change one feature at a time, then two features, and so on.
        for num_features_to_vary in range(1, len(self.features_to_vary)+1):
            # randomly select which features to vary on this round
            selected_features = np.random.choice(self.features_to_vary, (sample_size, 1), replace=True)
            
            for k in range(sample_size):
                feature_to_vary = selected_features[k][0]

                # if feature_to_vary in 
                # edit the original query instance with the random change to each selected feature
                candidate_cfs.at[k, feature_to_vary] = random_instances.at[k, feature_to_vary]
                # ensure candidate_cfs are valid in regard to causal constraints.

                # if 
                # check what the changed instance is, if it has a causal constraint then update this.
            # predict the outcome for each modified instance
            scores = self.predict_fn(candidate_cfs)
            validity = self.decide_cf_validity(scores)

            if sum(validity) > 0:
                # add rows that are valid from this set of candidates
                rows_to_add = candidate_cfs[validity == 1]

                if cfs_df is None:
                    cfs_df = rows_to_add.copy()
                else:
                    cfs_df = pd.concat([cfs_df, rows_to_add])
                cfs_df.drop_duplicates(inplace=True)
                # Always change at least 2 features before stopping
                if num_features_to_vary >= 2 and len(cfs_df) >= total_CFs:
                    break

        self.total_cfs_found = 0
        self.valid_cfs_found = False
        if cfs_df is not None and len(cfs_df) > 0:
            # if more than requested number of cfs were found randomly sample
            if len(cfs_df) > total_CFs:
                cfs_df = cfs_df.sample(total_CFs)
            cfs_df.reset_index(inplace=True, drop=True)
            if len(cfs_df) > 0:
                self.cfs_pred_scores = self.predict_fn(cfs_df)
                cfs_df[self.data_interface.outcome_name] = self.get_model_output_from_scores(self.cfs_pred_scores)
            else:
                if self.model.model_type == ModelTypes.Classifier:
                    self.cfs_pred_scores = [0]*self.num_output_nodes
                else:
                    self.cfs_pred_scores = [0]
            self.total_cfs_found = len(cfs_df)

            self.valid_cfs_found = True if self.total_cfs_found >= self.total_CFs else False
            if len(cfs_df) > 0:
                final_cfs_df = cfs_df[self.data_interface.feature_names + [self.data_interface.outcome_name]]
                final_cfs_df[self.data_interface.outcome_name] = \
                    final_cfs_df[self.data_interface.outcome_name].round(self.outcome_precision)
                self.cfs_preds = final_cfs_df[[self.data_interface.outcome_name]].values
                self.final_cfs = final_cfs_df[self.data_interface.feature_names].values
            else:
                final_cfs_df = None
                self.cfs_preds = None
                self.cfs_pred_scores = None
                self.final_cfs = None
        else:
            final_cfs_df = None
            self.cfs_preds = None
            self.cfs_pred_scores = None
            self.final_cfs = None
        test_instance_df = self.data_interface.prepare_query_instance(query_instance)
        test_instance_df[self.data_interface.outcome_name] = \
            np.array(np.round(self.get_model_output_from_scores((test_pred,)), self.outcome_precision))
        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param is not None and posthoc_sparsity_param > 0 and \
                self.final_cfs is not None and 'data_df' in self.data_interface.__dict__:
            final_cfs_df_sparse = final_cfs_df.copy()
            final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(final_cfs_df_sparse,
                                                                       test_instance_df,
                                                                       posthoc_sparsity_param,
                                                                       posthoc_sparsity_algorithm,
                                                                       limit_steps_ls)
        elif self.final_cfs is not None:
            final_cfs_df_sparse = final_cfs_df.copy()
        else:
            final_cfs_df_sparse = None

        self.elapsed = timeit.default_timer() - start_time
        m, s = divmod(self.elapsed, 60)

        # decoding to original label
        test_instance_df, final_cfs_df, final_cfs_df_sparse = \
            self.decode_to_original_labels(test_instance_df, final_cfs_df, final_cfs_df_sparse)
        if final_cfs_df is not None:
            if verbose:
                print('Diverse Counterfactuals found! total time taken: %02d' %
                      m, 'min %02d' % s, 'sec')
        else:
            if self.total_cfs_found == 0:
                print('No Counterfactuals found for the given configuration, perhaps try with different parameters...',
                      '; total time taken: %02d' % m, 'min %02d' % s, 'sec')
            else:
                print('Only %d (required %d) ' % (self.total_cfs_found, self.total_CFs),
                      'Diverse Counterfactuals found for the given configuration, perhaps try with different parameters...',
                      '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

        desired_class_param = self.decode_model_output(pd.Series(self.target_cf_class))[0] \
            if hasattr(self, 'target_cf_class') else desired_class
        return exp.CounterfactualExamples(data_interface=self.data_interface,
                                          final_cfs_df=final_cfs_df,
                                          test_instance_df=test_instance_df,
                                          final_cfs_df_sparse=final_cfs_df_sparse,
                                          posthoc_sparsity_param=posthoc_sparsity_param,
                                          desired_class=desired_class_param,
                                          desired_range=desired_range,
                                          model_type=self.model.model_type)

    def get_samples(self, fixed_features_values, feature_range, query_instance, causal_constraints, sampling_random_seed, sampling_size):

        # first get required parameters
        precisions = self.data_interface.get_decimal_precisions(output_type="dict")

        if sampling_random_seed is not None:
            random.seed(sampling_random_seed)

        # set ranges based on causal constraints
        if causal_constraints is not None:
            for feature in self.data_interface.feature_names:
                # TODO: does increase/decrease do the same for categorical? this is assuming they're continuous for now
                if feature in causal_constraints["cannot_increase"]:
                    feature_range[feature][1] = min(query_instance[feature].values[0], feature_range[feature][1])
                if feature in causal_constraints["cannot_decrease"]:
                    feature_range[feature][0] = max(query_instance[feature].values[0], feature_range[feature][0])
                    
        samples = []
        for feature in self.data_interface.feature_names:
            if feature in fixed_features_values:
                sample = [fixed_features_values[feature]]*sampling_size
            elif feature in self.data_interface.continuous_feature_names:
                low = feature_range[feature][0]
                high = feature_range[feature][1]
                sample = self.get_continuous_samples(
                    low, high, precisions[feature], size=sampling_size,
                    seed=sampling_random_seed)
            else:
                if sampling_random_seed is not None:
                    random.seed(sampling_random_seed)
                sample = random.choices(feature_range[feature], k=sampling_size)

            samples.append(sample)
        samples = pd.DataFrame(dict(zip(self.data_interface.feature_names, samples)))
        return samples

    def get_continuous_samples(self, low, high, precision, size=1000, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if precision == 0:
            result = np.random.randint(low, high+1, size).tolist()
            result = [float(r) for r in result]
        else:
            result = np.random.uniform(low, high+(10**-precision), size)
            result = [round(r, precision) for r in result]
        return result

    def validate_constraints(self, constraints):
        possible_constraints = ["cannot_increase", "cannot_decrease", "must_increase_with", "must_decrease_with", 
                                "inverse_with_increase", "inverse_with_decrease"]
        if constraints is not None:
            for constraint in possible_constraints:
                if constraint not in constraints:
                    constraints[constraint] = []
        
        return constraints