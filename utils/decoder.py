from utils.processing import TextProcess
# import ctcdecode
from ctc_decoder import beam_search
import torch

textprocess = TextProcess()

with open('labels.txt', 'r') as f:
    lines = f.read().splitlines()
labels = []
for i, line in enumerate(lines):
    if line == "<SPACE>":
        labels.append(" ")
    else:
        labels.append(line)


def DecodeGreedy(output, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2).squeeze(1)
    decode = []
    for i, index in enumerate(arg_maxes):
        if index != blank_label:
            if collapse_repeated and i != 0 and index == arg_maxes[i - 1]:
                continue
            decode.append(index.item())
    return textprocess.int_to_text_sequence(decode)


# class CTCBeamDecoder:
#     def __init__(self, beam_size=100, blank_id=labels.index('_'), kenlm_path=None):
#         print("loading beam search with lm...")
#         self.decoder = ctcdecode.CTCBeamDecoder(
#             labels, alpha=0.522729216841, beta=0.96506699808,
#             beam_width=beam_size, blank_id=labels.index('_'),
#             model_path=kenlm_path)
#         print("finished loading beam search")

#     def __call__(self, output):
#         beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(
#             output)
#         return self.convert_to_string(beam_result[0][0], labels, out_seq_len[0][0])

#     def convert_to_string(self, tokens, vocab, seq_len):
#         return ''.join([vocab[x] for x in tokens[0:seq_len]])

def decoder_test(output):
    char = "".join(labels)
    x = output.numpy()
    x = x.reshape(x.shape[2], x.shape[1])
    print(x.shape)
    print(f'Beam search: "{beam_search(x, char)}"')


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels=labels, blank=28):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])
