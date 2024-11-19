class BiasRegistry:
    def __init__(self):
        self.scanner_type = {'Siemens Skyra': 0, 'Siemens Sonata': 1, 'Siemens Avanto': 2, 'Siemens Trio': 3,
                             'Siemens Verio': 4,
                             'Siemens Trio Tim': 5, 'Siemens Biograph_mMR': 6, 'GE Discovery 750': 7,
                             'Siemens Prisma_fit': 8,
                             'Siemens Prisma': 9}

        self.scanner_vendor = {'Siemens': 0, 'GE': 1}

        self.study = {'UKBB': 0, 'BIOCOG': 1, 'Neurocon': 2, 'Taowu': 3, 'Japan_dataset': 4, 'PD_MCI_PLS': 5,
                      'OASIS': 6,
                      'HAMBURG': 7,
                      'PD_MCI_CALGARY': 8, 'SALD': 9, 'C-BIG': 10, 'UOA': 11}
