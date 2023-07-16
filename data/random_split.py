import numpy as np


def main(dataset, train_ratio=0.7):
    labels = np.load('{}/labels.npy'.format(dataset))
    ndata = labels.shape[0]
    lst = np.array(list(range(ndata)))
    for _ in range(5):
        np.random.shuffle(lst)
    train_idx, test_idx = np.split(lst, [round(train_ratio * ndata)])
    np.save('{}/train_idx.npy'.format(dataset), train_idx)
    np.save('{}/test_idx.npy'.format(dataset), test_idx)


if __name__ == '__main__':
    dataset = 'bank'
    train_ratio = 0.8

    main(dataset=dataset,
         train_ratio=train_ratio)
