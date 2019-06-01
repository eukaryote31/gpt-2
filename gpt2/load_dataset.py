import glob
import numpy as np
import os
import tensorflow as tf
import tqdm
import random


def data_paths(path):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)
    return paths


def load_dataset(enc, paths, combine):
    token_chunks = []
    indices = []
    raw_text = ''
    i = 0
    for path in tqdm.tqdm(paths, disable=len(paths) == 1):
        indices.append([])
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                indices[-1].append(i)
                for item in npz.files:
                    token_chunks.append(npz[item])
                    i += 1
                indices[-1].append(i - 1)

        else:
            # Plain text
            indices[-1].append(i)
            with open(path, 'r') as fp:
                raw_text += fp.read()
            if len(raw_text) >= combine:
                tokens = np.stack(enc.encode(raw_text))
                token_chunks.append(tokens)
                i += 1

                raw_text = ''
            else:
                raw_text += '<|endoftext|>'
            indices[-1].append(i - 1)
    if raw_text:
        tokens = np.stack(enc.encode(raw_text))
        token_chunks.append(tokens)
        indices[-1].append(i)
        i += 1
    return token_chunks, indices


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, enc, combine, path, perm_path, num_simultaneous_files=5, seed=None, dataset_loader=load_dataset, arr_paths=False, shuffle=True):
        self.load_dataset = dataset_loader
        self.enc = enc
        self.combine = combine
        print('Loading perma dataset')
        if perm_path is None:
            self.permchunks = []
        else:
            self.permchunks, _ = self.load_dataset(enc, perm_path if arr_paths else data_paths(perm_path), combine)
        self.num_simultaneous_files = num_simultaneous_files

        print('Loading cycling dataset')
        if path is None:
            self.paths = []
        else:
            self.paths = path if arr_paths else data_paths(path)

        self.seed = seed

        if shuffle:
            random.shuffle(self.paths)
        self.chunks, self.chunkindices = self.load_dataset(enc, self.paths[:num_simultaneous_files], combine)
        self.cycleindex = 0
        print('Paths loaded:', self.paths[:num_simultaneous_files])

        self.set_chunks(self.chunks)

    def set_chunks(self, chunks):
        chunks.extend(self.permchunks)
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])
        self.rs = np.random.RandomState(seed=self.seed)

    def cycle_files(self):
        if len(self.paths) - len(self.chunkindices) == 0 or self.num_simultaneous_files == 0:
            # can't cycle :(
            return
        self.chunks = self.chunks[:self.chunkindices[-1][-1] + 1]

#        assert self.chunkindices[0][0] == 0
        # unload first file
        del self.chunks[:self.chunkindices[0][-1] + 1]
        del self.chunkindices[0]
        # shift indices
        newdelta = self.chunkindices[0][0]
        self.chunkindices = [[y - newdelta for y in x] for x in self.chunkindices]

#        assert self.chunkindices[0][0] == 0
#        assert len(self.chunkindices) == 1 or self.chunkindices[1][0] == self.chunkindices[0][-1] + 1
#        print('Unloaded file {}'.format(self.paths[self.cycleindex]))

        sidx = self.cycleindex + len(self.chunkindices) + 1

        sidx %= len(self.paths)

#        print('Loading file {}'.format(self.paths[sidx]))
        nc, ncis = self.load_dataset(self.enc, [self.paths[sidx]], self.combine)

        self.chunks.extend(nc)
        self.chunkindices.extend([[y + self.chunkindices[-1][-1] + 1 for y in x] for x in ncis])

        self.cycleindex += 1
        self.cycleindex %= len(self.paths)
        self.set_chunks(self.chunks)

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(
            length)
        while True:
            index = self.rs.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]
