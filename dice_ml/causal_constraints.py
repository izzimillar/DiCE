import graphviz
from raiutils.exceptions import UserConfigValidationException

class CausalConstraints:
    def __init__(self, features):
        self.feature_names = features
        # values consist of:
        # feature_depends_on : [ feature1, feature2, ... ]
        # depends_on --> feature1
        self.constraints = {
            "increase_with" : {},
            "decrease_with" : {},
            "increase_on_decrease" : {},
            "decrease_on_increase" : {},
        }
        self.constraints_config = {
            "increase_with" : {
                "inverse" : "decrease_with",
                "colour" : "red",
            },
            "decrease_with" : {
                "inverse" : "increase_with",
                "colour" : "blue",
            },
            "increase_on_decrease" : {
                "inverse" : "increase_on_decrease",
                "colour" : "orange",
            },
            "decrease_on_increase" : {
                "inverse" : "decrease_on_increase",
                "colour" : "green",
            },

        } 
    

    def add_node(self, feature_name):
        self.feature_names.append(feature_name)
    
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
