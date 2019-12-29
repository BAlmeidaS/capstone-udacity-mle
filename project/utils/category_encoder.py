import numpy as np

CATEGORIES_MAP = {
    0: 13,   1: 52,   2: 63,   3: 70,   4: 90,   5: 104,  6: 115,  7: 184,
    8: 194,  9: 198, 10: 202, 11: 215, 12: 256, 13: 258, 14: 260, 15: 262,
    16: 263, 17: 265, 18: 266, 19: 267, 20: 268, 21: 269, 22: 270, 23: 300,
    24: 319, 25: 320, 26: 379, 27: 391, 28: 464, 29: 466, 30: 482, 31: 500,
    32: 512, 33: 534, 34: 551, 35: 565, 36: 580, 37: 585, 38: 592
}


class CategoryEncoder():
    def transform(arr: np.array) -> np.array:
        """ Return the useful categories, according to class exploration notebook

        Args:
            arr: np.array

        Returns:
            np.array
        """
        if len(arr.shape) == 1:
            arr = np.expand_dims(arr, axis=0)

        if arr.shape[-1] != 599 and arr.shape[-1] != 603:
            raise RuntimeError('This is not an array valid with all classes defined!')
        elif arr.shape[-1] == 599:
            return arr[:, list(CATEGORIES_MAP.values())]
        else:
            return arr[:, (list(CATEGORIES_MAP.values()) + [-4, -3, -2, -1])]

    def invert_transform(arr: np.array) -> np.array:
        """ Returns all categories, tranforming back from useful categories

        Args:
            arr: np.array

        Returns:
            np.array
        """
        if len(arr.shape) == 1:
            arr = np.expand_dims(arr, axis=0)

        if arr.shape[-1] != 39 and arr.shape[-1] != 43:
            raise RuntimeError('This is not an array valid with valid categories!')
        else:
            temp = np.zeros((arr.shape[0], 599))

            if arr.shape[-1] == 39:
                for i, r in enumerate(arr):
                    for j, c in enumerate(r):
                        temp[i, CATEGORIES_MAP[j]] = c

                return temp
            else:
                for i, r in enumerate(arr[:, :-4]):
                    for j, c in enumerate(r):
                        temp[i, CATEGORIES_MAP[j]] = c
                return np.concatenate([temp, arr[:, -4:]], axis=1)
