import glob
import os

import noisytest.experiment_reader


class DataSetReader(noisytest.ExperimentReader):
    """Reader for a whole dataset consisting of multiple experiments in one folder"""

    def read_data_set(self, path):
        """Read a data-set consisting of subfolders with a noise data and meta data file
           Returns a concatenated tensor with the complete data set
        """
        data = self._preprocessor.create_empty_input_target_data()

        for filename in glob.iglob(os.path.join(path, "*" + noisytest.NoiseReader.file_extension())):
            exp_name = os.path.splitext(os.path.basename(filename))[0]

            experiment = self.read_experiment(path, exp_name)
            data = self._preprocessor.concat_input_target_data(data, experiment)

        return data
