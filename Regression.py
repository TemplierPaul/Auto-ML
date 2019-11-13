from Dataset import *


class Regressor(DataSet):

    def __init__(self):
        super(Regressor, self).__init__()
        self.type = type
        self.model = None
        self.true_values = {'train': None, 'validation': None, 'test': None}
        self.predictions = {'train': None, 'validation': None, 'test': None}

    def __str__(self):
        s = 'Supervised '
        if self.predictions['train'] is None:
            s += ' Not'
        s += " Trained"
        s += '\n' + super(Regressor, self).__str__()
        return s