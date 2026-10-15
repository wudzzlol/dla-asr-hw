import torch.nn.utils.rnn as rnn_utils
from torch import nn
from torch.nn import Sequential


class DeepSpeech(nn.Module):
    @staticmethod
    def conv_out_len(length, kernel, padding, stride):
        return (length + 2 * padding - kernel) // stride + 1

    def __init__(
        self,
        n_feats,
        n_tokens,
        conv_channels,
        conv_kernel_sizes,
        conv_paddings,
        conv_strides,
        rnn_hidden,
        rnn_layers,
        rnn_batch_norm,
    ):
        super().__init__()

        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_paddings = conv_paddings
        self.conv_strides = conv_strides

        self.conv = Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=tuple(conv_kernel_sizes[0]),
                stride=tuple(conv_strides[0]),
                padding=tuple(conv_paddings[0]),
            ),
            nn.BatchNorm2d(conv_channels),
            nn.Hardtanh(min_val=0, max_val=20),
            nn.Conv2d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=tuple(conv_kernel_sizes[1]),
                stride=tuple(conv_strides[1]),
                padding=tuple(conv_paddings[1]),
            ),
            nn.BatchNorm2d(conv_channels),
            nn.Hardtanh(min_val=0, max_val=20),
        )  # B C F T

        freq_after_conv = self.conv_out_len(
            n_feats, conv_kernel_sizes[0][0], conv_paddings[0][0], conv_strides[0][0]
        )
        freq_after_conv = self.conv_out_len(
            freq_after_conv,
            conv_kernel_sizes[1][0],
            conv_paddings[1][0],
            conv_strides[1][0],
        )

        self.grus = nn.ModuleList()
        for i in range(rnn_layers):
            input_size = conv_channels * freq_after_conv if i == 0 else 2 * rnn_hidden
            self.grus.append(
                nn.GRU(
                    input_size=input_size,
                    hidden_size=rnn_hidden,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )

        self.bns = nn.ModuleList()
        for i in range(rnn_layers):
            self.bns.append(
                nn.BatchNorm1d(2 * rnn_hidden) if rnn_batch_norm else nn.Identity()
            )

        self.linear_log_softmax = Sequential(
            nn.Linear(2 * rnn_hidden, n_tokens), nn.LogSoftmax(dim=-1)
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        x = self.conv(spectrogram.unsqueeze(1))  # B C F T -> B T F C
        B, _, _, T = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B, T, -1)
        transformed_spec_len = self.transform_input_lengths(spectrogram_length)
        transformed_spec_len_cpu = transformed_spec_len.cpu()

        for i in range(len(self.grus)):
            x = rnn_utils.pack_padded_sequence(
                input=x,
                lengths=transformed_spec_len_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output = self.grus[i](x)[0]
            x, _ = rnn_utils.pad_packed_sequence(
                packed_output, batch_first=True, total_length=T
            )
            x = self.bns[i](x.transpose(1, 2)).transpose(1, 2)

        x = self.linear_log_softmax(x)

        return {
            "log_probs": x,
            "log_probs_length": transformed_spec_len,
        }

    def transform_input_lengths(self, input_lengths):
        input_lengths = self.conv_out_len(
            input_lengths,
            self.conv_kernel_sizes[0][1],
            self.conv_paddings[0][1],
            self.conv_strides[0][1],
        )
        input_lengths = self.conv_out_len(
            input_lengths,
            self.conv_kernel_sizes[1][1],
            self.conv_paddings[1][1],
            self.conv_strides[1][1],
        )
        return input_lengths

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
