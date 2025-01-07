class BiasRegistry:
    def __init__(self):
        self.scanner_type = {'GE Discovery 750': 0, 'Siemens Skyra': 1, 'Siemens Prisma_fit': 2, 'Siemens Avanto': 3,
                             'Siemens Verio': 4, 'Siemens Trio Tim': 5, 'Siemens Trio': 6, 'Siemens Sonata': 7}

        self.study = {'PD_MCI_CALGARY': 0, 'UKBB': 1, 'HAMBURG': 2, 'C-BIG': 3, 'Neurocon': 4, 'Japan_dataset': 5,
                      'PD_MCI_PLS': 6, 'Taowu': 7, 'BIOCOG': 8}
