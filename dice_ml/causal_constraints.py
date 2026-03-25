import graphviz
from raiutils.exceptions import UserConfigValidationException

class CausalConstraints:
    def __init__(self, data_interface):
        self.data_interface = data_interface
        self.feature_names = data_interface.feature_names
        self.continuous_features = data_interface.continuous_feature_names
        self.categorical_features = data_interface.categorical_feature_names
        # values consist of:
        # depends_on : [ feature1, feature2, ... ]
        # depends_on --> feature1
        self.constraints = {
            "increase_with" : {},
            "decrease_with" : {},
            "increase_on_decrease" : {},
            "decrease_on_increase" : {},
        }

        self.single_constraints = {
            "cannot_increase" : [],
            "cannot_decrease" : [],
            "cannot_change" : [],
        }

        self.constraints_config = {
            "increase_with" : {
                "depends_on_change" : 2,
                "dependent_change" : 2,
                "inverse" : "decrease_with",
                "colour" : "red",
            },
            "decrease_with" : {
                "depends_on_change" : 1,
                "dependent_change" : 1,
                "inverse" : "increase_with",
                "colour" : "blue",
            },
            "increase_on_decrease" : {
                "depends_on_change" : 1,
                "dependent_change" : 2,
                "inverse" : "increase_on_decrease",
                "colour" : "orange",
            },
            "decrease_on_increase" : {
                "depends_on_change" : 2,
                "dependent_change" : 1,
                "inverse" : "decrease_on_increase",
                "colour" : "green",
            },

        }
    
    def validate_constraint_features(self, model_features):
        for feature in self.feature_names:
            if feature not in model_features:
                print(
                    "Feature names in the model and constraints do not match. Please ensure that all constraint features " \
                    "are present in the model."
                    )
                return False
        
        return True
    
    def update_data_interface(self, data_interface):
        self.data_interface = data_interface
    
    def add_constraint(self, constraint_type, feature, depends_on, inverse=False):
        """
        Adds a new constraint to the graph.

        constraint_type: the type of relationship between two features.
        feature: the feature in feature_names that the new constraint effects.
        depends_on: the feature that this constraint depends upon.
        """
        # check that user information is valid
        if constraint_type not in self.constraints.keys():
            raise UserConfigValidationException(
                "Not a valid constraint type. Please choose one of the following: " \
                "increase_with, decrease_with, increase_on_decrease, decrease_on_increase."
                )

        if feature not in self.feature_names or depends_on not in self.feature_names:
            raise UserConfigValidationException(
                "Not a valid feature. Please make sure you have added the feature before adding a constraint."
                )


        # user configuration is valid --> we can add the constraint.
        current_constraints = self.constraints[constraint_type]

        if depends_on not in current_constraints:
            current_constraints[depends_on] = []
        
        current_constraints[depends_on].append(feature)

        # add the inverse constraint as well
        if inverse:
            inverse_constraint_type = self.constraints_config[constraint_type]["inverse"]
            if inverse_constraint_type is not None:
                inverse_constraints = self.constraints[inverse_constraint_type]
                if depends_on not in inverse_constraints:
                    inverse_constraints[depends_on] = []
                
                inverse_constraints[depends_on].append(feature)


    def add_single_constraint(self, constraint_type, feature):
        """
        Adds a new constraint to a feature. Currently no visualisation in the graph for this sort of constraint.

        constraint_type: the constraint on the single feature.
        feature: the feature in feature_names that the new constraint effects.
        """
        # check that user information is valid
        if constraint_type not in self.single_constraints.keys():
            raise UserConfigValidationException(
                "Not a valid constraint type. Please choose one of the following: " \
                "increase_with, decrease_with, increase_on_decrease, decrease_on_increase."
                )

        if feature not in self.feature_names:
            raise UserConfigValidationException(
                "Not a valid feature. Please make sure you have added the feature before adding a constraint."
                )
        
        if (feature in self.categorical_features 
            and feature not in self.data_interface.categorical_features_ordering 
            and constraint_type in ["cannot_decrease", "cannot_increase"]
        ):
            raise UserConfigValidationException(
                "This is a categorical feature that does not have an ordering. Please update the data_interface."
                )

        current_constraints = self.single_constraints[constraint_type]

        if feature not in current_constraints:
            current_constraints.append(feature)
    

    # how should a feature change for a certain constraint to be valid for the dependent features to need to change
    def feature_change_for_valid_constraint(self, constraint_type, feature_changed_by):
        depends_on_should_change = self.constraints_config[constraint_type]["depends_on_change"]
        if depends_on_should_change == feature_changed_by:
            return self.constraints_config[constraint_type]["dependent_change"]
        else:
            return 0
    
    def dependencies_to_change(self, feature, feature_changed_by):
        # feature with list of how they need to change
        changes = {}
        # go through all types of constraint
        for constraint in self.constraints:
            # check if the changed feature has any dependencies
            if feature in self.constraints[constraint]:
                # if it does check the impact on the dependent features
                should_change = self.feature_change_for_valid_constraint(constraint, feature_changed_by)
                if should_change > 0:
                    for dependent in self.constraints[constraint][feature]:
                        if dependent not in changes:
                            changes[dependent] = set()
                        changes[dependent].add(should_change)
        return changes
                

        

    def create_constraint_visualisation(self):
        dot = graphviz.Digraph()

        for feature in self.feature_names:
            dot.node(feature)
        
        for constraint in self.constraints:
            all_dependencies = self.constraints[constraint]
            for a in all_dependencies:
                for b in all_dependencies[a]:
                    dot.edge(a, b, color=self.constraints_config[constraint]["colour"])

        return dot
