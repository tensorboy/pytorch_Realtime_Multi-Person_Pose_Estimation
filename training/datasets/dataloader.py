from typing import Generator
from torch.utils.data.dataloader import DataLoader
import logging 

logger = logging.getLogger(__name__)

class sDataLoader(DataLoader):
    def get_stream(self):
        """
        Return a generate that can yield endless data.
        :Example:
        stream = get_stream()
        for i in range(100):
            batch = next(stream)

        :return: stream
        :rtype: Generator
        """
        while True:
            for data in iter(self):
                yield data

    @staticmethod
    def copy(loader):
        """
        Init a sDataloader from an existing Dataloader
        :param loader: an instance of Dataloader
        :type loader: DataLoader
        :return: a new instance of sDataloader
        :rtype: sDataLoader
        """
        if not isinstance(loader, DataLoader):
            logger.warning('loader should be an instance of Dataloader, but got {}'.format(type(loader)))
            return loader

        new_loader = sDataLoader(loader.dataset)
        for k, v in loader.__dict__.items():
            setattr(new_loader, k, v)
        return new_loader
