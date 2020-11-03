import copy
import logging


class Optimizer:
    """Performs a hyper-parameter search on the model"""

    def __init__(self, training_data, validation_data, pipeline):
        self._training_data = training_data
        self._validation_data = validation_data
        self._pipeline = pipeline
        self._parameter_values = {}
        self._best_parameter_values = {}
        self._best_fitness = 0

    def _find_hyper_parameters(self):
        """Return available hyper-parameters in model and preprocessor as dict"""

        processor = self._pipeline.feature_preprocessor
        hyper_parameters = self._pipeline.model.hyper_parameters

        while processor is not None:
            hyper_parameters = {**hyper_parameters, **processor.hyper_parameters}
            processor = processor.parent

        return hyper_parameters

    def _set_hyper_parameter_on(self, obj, name, value):
        """Set hyper parameter on obj if it exists, returns true on success"""
        self._parameter_values[name] = value
        if name in obj.hyper_parameters.keys():
            setattr(obj, name, value)
            return True

        return False

    def _set_hyper_parameter(self, name, value):
        """Set hyper parameter of model or preprocessor"""

        if self._set_hyper_parameter_on(self._pipeline.model, name, value):
            return

        processor = self._pipeline.feature_preprocessor
        while processor is not None:
            if self._set_hyper_parameter_on(processor, name, value):
                return
            processor = processor.parent

        raise RuntimeError("hyper-parameter not found")

    def _evaluate(self):
        """Evaluate the fitness of the model for the current settings"""
        error = self._pipeline.learn(copy.deepcopy(self._training_data), copy.deepcopy(self._validation_data))

        if error.fitness > self._best_fitness:
            logging.debug(f"reached new fitness optimum {error.fitness} for parameters {self._parameter_values}")
            self._best_fitness = error.fitness
            self._best_parameter_values = self._parameter_values

    def _search_sub_space(self, hyper_parameters):
        """Recursively search the space of given hyper parameters with a line search"""

        if not hyper_parameters:
            self._evaluate()
            return

        sub_space_parameters = copy.copy(hyper_parameters)
        param_name, param_range = sub_space_parameters.popitem()

        for value in param_range:
            self._set_hyper_parameter(param_name, value)
            self._search_sub_space(sub_space_parameters)

    def grid_search(self):
        """Performs a grid search on all hyper parameters of the pipeline"""
        hyper_params = self._find_hyper_parameters()

        self._search_sub_space(hyper_params)

        logging.info(f"Best fitness is {self._best_fitness}")
        logging.info(f"Optimal values are {self._best_parameter_values}")
        for name, value in self._best_parameter_values.items():
            self._set_hyper_parameter(name, value)
