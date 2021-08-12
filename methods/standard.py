class Standard:
    def __init__(self, criterion):
        self.criterion = criterion

        self.name = f"Standard"
        self.num_models = 1

    def loss(self, outputs, target, *args, **kwargs):
        output = outputs[0]
        loss = self.criterion(output, target)
        return [loss], [[]]
