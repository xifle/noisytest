import copy


class Optimizer:
    """Performs a hyper-parameter search on the model"""

    def __init__(self, training_data, validation_data, model):
        self._training_data = training_data
        self._validation_data = validation_data
        self._model = model
        self._parameter_values = {}
        self._best_parameter_values = {}
        self._best_fitness = 0

    def __find_hyper_parameters(self):

        processor = self._model.preprocessor
        hyper_parameters = self._model.hyper_parameters

        while processor is not None:
            hyper_parameters = {**hyper_parameters, **processor.hyper_parameters}
            processor = processor.parent

        return hyper_parameters

    def __set_hyper_parameter(self, obj, name, value):
        """Set hyper parameter if it exists, returns true on success"""
        self._parameter_values[name] = value
        if name in obj.hyper_parameters.keys():
            setattr(obj, name, value)
            print(name, value)
            return True

        return False

    def __find_and_set_hyper_parameter(self, name, value):
        if self.__set_hyper_parameter(self._model, name, value):
            return

        processor = self._model.preprocessor
        while processor is not None:
            if self.__set_hyper_parameter(processor, name, value):
                return
            processor = processor.parent

        assert False, "Internal error"

    def __evaluate(self):
        self._model.train(copy.deepcopy(self._training_data))
        error = self._model.validate(copy.deepcopy(self._validation_data))
        print("Fittness", error.fitness)

        if error.fitness > self._best_fitness:
            self._best_fitness = error.fitness
            self._best_parameter_values = self._parameter_values

    def __search_sub_space(self, hyper_parameters):

        if not hyper_parameters:
            self.__evaluate()
            return

        sub_space_parameters = copy.copy(hyper_parameters)
        param = sub_space_parameters.popitem()

        for value in param[1]:
            self.__find_and_set_hyper_parameter(param[0], value)
            self.__search_sub_space(sub_space_parameters)

    def grid_search(self):
        hyper_params = self.__find_hyper_parameters()

        self.__search_sub_space(hyper_params)

        for name, value in self._best_parameter_values:
            print("Optimal " + name + ":", value)
            self.__find_and_set_hyper_parameter(name, value)
