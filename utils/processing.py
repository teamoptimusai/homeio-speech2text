import torch


class TextProcess:
    def __init__(self):
        self.char_map = {}
        self.index_map = {}
        with open('labels.txt', 'r') as f:
            lines = f.read().splitlines()
        for i, line in enumerate(lines[:-1]):
            self.char_map[line] = i
            self.index_map[i] = line
        self.index_map[1] = ' '

    def text_to_int_sequence(self, text):
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text_sequence(self, labels):
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')


textprocess = TextProcess()


def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(textprocess.int_to_text_sequence(
            labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j - 1]:
                    continue
                decode.append(index.item())
        decodes.append(textprocess.int_to_text_sequence(decode))
    return decodes, targets
