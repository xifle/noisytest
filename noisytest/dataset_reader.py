import glob
import os

import noisytest.experiment_reader


class DataSetReader(noisytest.ExperimentReader):

    def read_data_set(self, path):
        data = self._preprocessor.create_empty_input_target_data()

        for filename in glob.iglob(os.path.join(path, "*" + noisytest.NoiseReader.file_extension())):
            exp_name = os.path.splitext(os.path.basename(filename))[0]

            experiment = self.read_experiment(path, exp_name)
            data = self._preprocessor.concat_input_target_data(data, experiment)

        return data
