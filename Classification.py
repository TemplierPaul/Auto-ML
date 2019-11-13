from Dataset import *


class Classifier(DataSet):

    def __init__(self):
        super(Classifier, self).__init__()
        self.type = type
        self.model = None
        self.true_labels = {'train': None, 'validation': None, 'test': None}
        self.predictions = {'train': None, 'validation': None, 'test': None}

    def __str__(self):
        s = 'Supervised '
        if self.predictions['train'] is None:
            s += ' Not'
        s += " Trained"
        s += '\n' + super(Classifier, self).__str__()
        return s
