
from neuralmonkey.decorators import tensor


class BeamSearchDecoder(ModelPart):

    def __init__(self,
                 name: str,
                 parent_decoder: Decoder,
                 beam_size: int,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        assert check_argument_types()

        self._parent_decoder = parent_decoder
        self._beam_size = beam_size

        self._cell = self._parent_decoder.get_rnn_cell()



    def step(self):

        att_objects = [self._parent_decoder.get_attention_object(e, False)
                       for e in self._parent_decoder.encoders]
        att_objects = [a for a in att_objects if a is not None]


        assert self._parent_decoder.step_scope.reuse
        logits, state, attns = self._parent_decoder.step()



        return






    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        """Populate the feed dictionary for the decoder object

        Arguments:
            dataset: The dataset to use for the decoder.
            train: Boolean flag, telling whether this is a training run
        """
        assert not train
        assert len(dataset) == 1

        # sentences = cast(Iterable[List[str]],
        #                  dataset.get_series(self.data_id, allow_none=True))

        # if sentences is None and train:
        #     raise ValueError("When training, you must feed "
        #                      "reference sentences")

        # sentences_list = list(sentences) if sentences is not None else None

        # TODO assert ze je to jen jedna veta

        fd = {}  # type: FeedDict

        # if sentences is not None:
        #     # train_mode=False, since we don't want to <unk>ize target words!
        #     inputs, weights = self.vocabulary.sentences_to_tensor(
        #         sentences_list, self.max_output_len, train_mode=False,
        #         add_start_symbol=False, add_end_symbol=True)

        #     assert inputs.shape == (self.max_output_len, len(sentences_list))
        #     assert weights.shape == (self.max_output_len, len(sentences_list))

        #     fd[self.train_inputs] = inputs
        #     fd[self.train_padding] = weights

        return fd
