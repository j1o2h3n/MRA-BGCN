import numpy as np
import torch
import torch.nn as nn

from bgcgru import bgcgru



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, supports, M, args):
        self.supports = supports
        self.M = M
        self.cl_decay_steps = args.cl_decay_steps
        self.num_nodes = args.num_nodes
        self.num_rnn_blocks = args.num_rnn_blocks
        self.rnn_units = args.rnn_units
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, supports, M, args):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, supports, M, args)
        self.args = args
        self.input_dim = args.input_dim
        self.seq_len = args.seq_len
        self.rnn_units = args.rnn_units
        self.mra_bgcn_blocks = nn.ModuleList()
        for i in range(self.num_rnn_blocks):
            if i == 0:
                self.mra_bgcn_blocks.append(bgcgru(self.input_dim, self.rnn_units, supports, M, args))
            else:
                self.mra_bgcn_blocks.append(bgcgru(self.rnn_units, self.rnn_units, supports, M, args))

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_blocks, batch_size, self.args.num_nodes * self.rnn_units)).cuda()

        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.mra_bgcn_blocks):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self,  supports, M, args):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self,  supports, M, args)
        self.args = args
        self.output_dim = args.output_dim
        self.rnn_units = args.rnn_units
        self.horizon = args.horizon # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.mra_bgcn_blocks = nn.ModuleList()
        for i in range(self.num_rnn_blocks):
            if i == 0:
                self.mra_bgcn_blocks.append(bgcgru(self.output_dim, self.rnn_units, supports, M, args))
            else:
                self.mra_bgcn_blocks.append(bgcgru(self.rnn_units, self.rnn_units, supports, M, args))

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.mra_bgcn_blocks):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class mra_bgcn(nn.Module, Seq2SeqAttrs):
    def __init__(self, args, supports, M):
        super().__init__()
        Seq2SeqAttrs.__init__(self, supports, M, args)
        self.encoder_model = EncoderModel( supports, M, args)
        self.decoder_model = DecoderModel( supports, M, args)
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = bool(True)
        self.args = args

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_nodes * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)
        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim)).cuda()
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_nodes * input_dim)
        :param labels: shape (horizon, batch_size, num_nodes * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        inputs = inputs # (batch_size, input_dim, num_nodes, seq_len)
        labels = labels # (batch_size, 1, num_nodes, horizon)
        inputs = inputs.permute(3, 0, 2, 1) # (seq_len, batch_size, num_nodes, input_dim)
        batch_size = inputs.size(1)
        inputs = torch.reshape(inputs, (self.args.seq_len, batch_size, self.args.num_nodes * self.args.input_dim)) # (seq_len, batch_size , num_nodes * input_dim)
        if labels is not None:
            labels = labels.permute(3, 0, 2, 1)  # (horizon, batch_size, num_nodes, 1)
            labels = labels.view(self.args.horizon, batch_size, self.args.num_nodes * self.args.output_dim) # (horizon, batch_size ,num_nodes)

        encoder_hidden_state = self.encoder(inputs)
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)

        outputs = outputs # (horizon, batch_size, num_nodes * output_dim)
        outputs = outputs.view(self.args.horizon, batch_size, self.args.num_nodes, self.args.output_dim) # (horizon, batch_size, num_nodes, output_dim)
        outputs = outputs.permute(1, 3, 2, 0)  # (batch_size, output_dim, num_nodes, horizon)
        outputs = outputs.transpose(1,3) # 64,12,170,1

        return outputs
