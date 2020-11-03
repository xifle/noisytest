import logging
import os

import noisytest.noise_reader
import noisytest.meta_data_reader
import noisytest.preprocessor


class ExperimentReader:
    """Reads in experiment (training or validation) data"""

    def __init__(self, import_preprocessor, do_pad_data=True):
        """Construct from ImportPreprocessor instance. The import_preprocessor is used
        to generate a tensor from raw training / validation data
        """
        self._preprocessor = import_preprocessor
        self._do_pad_data = do_pad_data

    @property
    def do_pad_data(self):
        """Pad the data block when it is smaller than the frame size?

         If false, the data block will be enlarged instead.
         Should be false on validation data"""
        return self._do_pad_data

    @do_pad_data.setter
    def do_pad_data(self, value: bool):
        self._do_pad_data = value

    def read_experiment(self, data_path: str, experiment_name: str):
        """Reads in an experiment consisting of noise data and labels"""

        full_filename = os.path.join(data_path, experiment_name)
        meta_reader = noisytest.MetaDataReader(full_filename)
        noise_reader = noisytest.NoiseReader(full_filename)

        if not self._preprocessor.is_matching_sample_time(noise_reader.sample_time):
            raise ValueError(f"{full_filename}: Sample time does not match preprocessor specs")

        return self._aggregate_data(full_filename, meta_reader.data(), noise_reader.data())

    def _aggregate_data(self, filename, meta_data, noise_data):

        result = self._preprocessor.create_empty_input_target_data()

        for block_range, label in zip(meta_data.block_ranges, meta_data.block_labels):
            (start_time, end_time) = self._start_end_time(filename, noise_data.time, block_range)

            logging.debug(f"{filename}: Aggregating block from t={start_time} to t={end_time}")

            noise = noise_data.noise_estimate[(noise_data.time >= start_time) & (noise_data.time <= end_time)]
            inout_data = noisytest.InputTargetData(noise, label, noise.size, 1)

            processed = self._preprocessor.prepare_input_target_data(inout_data)
            result = self._preprocessor.concat_input_target_data(result, processed)

        return result

    def _start_end_time(self, filename, time_data, block_range):
        """Determine start and end time of a data block. Returns a tuple (start, end) in seconds"""
        start_time = block_range[0]
        end_time = block_range[1]

        if end_time == -1:
            end_time = time_data[-1] + 0.1

        block_duration = end_time - start_time
        if block_duration < self._preprocessor.frame_duration and not self.do_pad_data:
            # Don't pad - use a broader chunk than specified instead
            start_time -= (self._preprocessor.frame_duration - block_duration) / 2.0
            end_time = start_time + self._preprocessor.frame_duration + 1.0 / self._preprocessor.input_data_sample_rate

        if start_time >= end_time:
            raise ValueError(f"{filename}: start time for a block must be less than end time")

        return start_time, end_time
