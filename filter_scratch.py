#! ./venv/bin/python
from RESNET.train import CifarData
import random
from torch.utils.data import Subset
from torch.utils.data import DataLoader


class NewCifarData(CifarData):

    @staticmethod
    def random_indices_in_range(num_to_remove, num_labels):
        """
        param: num_to_remove: the number of labels
        to remove from the dataset

        param: num_labels: the number of labels in
        the dataset
        """
        indices = random.sample(range(0, num_labels-1), num_to_remove)
        return indices

    def filter_dataset(self, batch_size, num_to_remove):
        trainset, valset = self.get_dataset()
        labels_to_remove = self.random_indices_in_range(num_to_remove, 100)
        # get the indices to remove from the torch dataset
        # given the labels to remove
        train_indices_to_remove = [idx for idx, target in enumerate(trainset.targets) if target not in labels_to_remove]
        val_indices_to_remove = [idx for idx, target in enumerate(valset.targets) if target not in labels_to_remove]
        trainloader = DataLoader(Subset(trainset, train_indices_to_remove),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            )
        valloader = DataLoader(Subset(trainset, val_indices_to_remove),
                               batch_size=1,
                               shuffle=True,)
        return trainloader, valloader


cd = NewCifarData()
trn, val = cd.get_data_loader()
trainloader, valloader = cd.filter_dataset(16, 99)
print(len(valloader))
