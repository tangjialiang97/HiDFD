from nvidia.dali.plugin.pytorch import DALIGenericIterator

class DALIDataloader(DALIGenericIterator):
    def __init__(self, pipeline, data_size, batch_size, output_map=["data", "label"], auto_reset=True, onehot_label=False):
        self.data_size = data_size
        self.batch_size = batch_size
        self.onehot_label = onehot_label
        self.output_map = output_map
        super().__init__(pipelines=pipeline, size=data_size, auto_reset=auto_reset, output_map=output_map)

    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch[0]
            self._first_batch = None
            if self.onehot_label:
                return [batch[self.output_map[0]], batch[self.output_map[1]].squeeze().long()]
            else:
                return [batch[self.output_map[0]], batch[self.output_map[1]]]
        data = super().__next__()[0]
        if self.onehot_label:
            return [data[self.output_map[0]], data[self.output_map[1]].squeeze().long()]
        else:
            return [data[self.output_map[0]], data[self.output_map[1]]]

    def __len__(self):
        if self.data_size % self.batch_size == 0:
            return self.data_size // self.batch_size
        else:
            return self.data_size // self.batch_size + 1