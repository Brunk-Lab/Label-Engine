
class BaseTrainer(object):
    def __init__(self) -> None:
        pass


    def training(self, epoch):
        pass

    def run(self, num_epochs, start_epoch=None, validation=False):
        pass

    def validation(self, epoch):
        pass

    def batch_forward(self, batch_data, validation=False):
        pass

    def add_loss(self, loss_name):
        pass