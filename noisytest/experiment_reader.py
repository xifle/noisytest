import logging
import os

import noisytest.noise_reader
import noisytest.meta_data_reader
import noisytest.preprocessor


class ExperimentReader:
    """Reads in experiment (training or validation) data"""

    def __init__(self, preprocessor):
        self._preprocessor = preprocessor
        self._do_pad_data = True

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

        with noisytest.NoiseReader(full_filename) as noise_reader:
            noise_data = noise_reader.data()
            assert self._preprocessor.is_matching_sample_time(noise_reader.sample_time), \
                "%r: Sample time does not match preprocessor specs" % full_filename

        with noisytest.MetaDataReader(full_filename) as meta_reader:
            meta_data = meta_reader.data()

        data = self._preprocessor.create_empty_input_target_data()

        for block_range, label in zip(meta_data.block_ranges, meta_data.block_labels):
            (start_time, end_time) = self.__start_end_time(full_filename, noise_data.time, block_range)

            logging.info("Processing block from t=", start_time, "to t=", end_time)
           # block_data = self._preprocessor.pad_if_necessary()

          #  block_frames = self._preprocessor.frame_input_target_data(block_data, label)

            noise = noise_data.noise_estimate[(noise_data.time >= start_time) & (noise_data.time <= end_time)]
            inout_data = noisytest.InputTargetData(noise, label, noise.size, 1)
            processed = self._preprocessor.prepare_input_target_data(inout_data)
            data = self._preprocessor.concat_input_target_data(data, processed)

        return data

    def __start_end_time(self, filename, time_data, block_range):
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

        assert start_time < end_time, "%r: start time for a block must be less than end time" % filename
        return start_time, end_time
