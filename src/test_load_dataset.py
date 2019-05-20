import load_dataset
import unittest
import numpy as np
import copy


def dummy_dataset(enc, paths, combine):
    ret = []
    indices = []
    i = 0
    for path in paths:
        indices.append([])
        indices[-1].append(i)
        for ind in range(3):
            ret.append(np.array([path, ind], dtype=object))
            i += 1

        indices[-1].append(i - 1)
    return ret, indices


class TestSampler(unittest.TestCase):
    def test_cycle(self):
        samp = load_dataset.Sampler(None, 0, ['c1', 'c2', 'c3', 'c4', 'c5', 'c6'], ['p1', 'p2'], dataset_loader=dummy_dataset, arr_paths=True, num_simultaneous_files=3, shuffle=False)

        chunks = []
        indices = []


        for i in range(7):
            chunks.append(copy.deepcopy(samp.chunks))
            indices.append(copy.deepcopy(samp.chunkindices))

            samp.cycle_files()

        self.assertEqual(indices[0], indices[-1])
        self.assertTrue(all((f == l).all() for f, l in zip(chunks[0], chunks[-1]))) # deep equals

        for i in range(5):
            self.assertFalse(all((f == l).all() for f, l in zip(chunks[i], chunks[i + 1]))) # deep equals


if __name__ == '__main__':
    unittest.main()
