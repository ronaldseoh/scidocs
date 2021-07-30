import ujson as json
import numpy as np
import torch
from tqdm import tqdm


class SimpleNet(torch.nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, n_hidden=0, dropout_rate=0.5):
        super(SimpleNet, self).__init__()

        # Setup layers
        # Input layer
        self.input = torch.nn.ModuleDict({
            'linear': torch.nn.Linear(input_dim, hidden_dim),
            'dropout': torch.nn.Dropout(dropout_rate),
            'nonlinear': torch.nn.ReLU(),
        })

        # Hidden Layer(s)
        if n_hidden > 0:
            self.hidden_layers = torch.nn.ModuleList()

            for i in range(n_hidden):
                self.hidden_layers.append(
                    torch.nn.ModuleDict({
                        'linear': torch.nn.Linear(hidden_dim, hidden_dim),
                        'dropout': torch.nn.Dropout(dropout_rate),
                        'nonlinear': torch.nn.ReLU(),
                    })
                )

        # Output
        self.output = torch.nn.ModuleDict({
            'linear': torch.nn.Linear(hidden_dim, output_dim),
            'dropout': torch.nn.Dropout(dropout_rate),
        })

    def forward(self, X):
        # Forward through the input layer
        activation = self.input['linear'](X)
        activation = self.input['dropout'](activation)
        activation = self.input['nonlinear'](activation)

        # Forward through hidden layers
        if hasattr(self, 'hidden_layers'):
            for hidden in self.hidden_layers:
                activation = hidden['linear'](activation)
                activation = hidden['dropout'](activation)
                activation = hidden['nonlinear'](activation)

        activation = self.output['linear'](activation)
        activation = self.output['dropout'](activation)

        return activation


def load_embeddings_from_jsonl(embeddings_path):
    """Load embeddings from a jsonl file.
    The file must have one embedding per line in JSON format.
    It must have two keys per line: `paper_id` and `embedding`

    Arguments:
        embeddings_path -- path to the embeddings file

    Returns:
        embeddings -- a dictionary where each key is the paper id
                                   and the value is a numpy array
    """
    embeddings = {}
    with open(embeddings_path, 'r') as f:
        for line in tqdm(f, desc='reading embeddings from file...'):
            line_json = json.loads(line)
            embeddings[line_json['paper_id']] = np.array(line_json['embedding'])
    return embeddings
