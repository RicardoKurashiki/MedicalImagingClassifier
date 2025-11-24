import numpy as np

from torch.utils.data import Sampler


class CustomSampler(Sampler):
    def __init__(self, labels, batch_size=32):
        self.labels = labels
        self.classes = np.unique(self.labels)
        self.N = len(self.classes)
        self.m_per_class = batch_size // self.N

        self.S = {c: np.where(self.labels == c)[0].tolist() for c in self.classes}
        self.C = {c: len(self.S[c]) for c in self.classes}
        self.c_max = max(self.C.values())
        self.K = self.c_max // self.m_per_class

    def __len__(self):
        return self.K

    def __iter__(self):
        S_work = {c: list(self.S[c]) for c in self.classes}

        for c in self.classes:
            np.random.shuffle(S_work[c])

        for _ in range(self.K):
            batch = []
            for c in self.classes:
                # Validação para casos onde a classe majoritária não tem o número de imagens necessárias para o batch
                # TODO: Talvez ele faça com que alguns dados da classe minoritária não sejam usados por conta de sempre serem reiniciadas
                # TODO: Uma maneira é talvez criar "batches" da classe minoritária para daí sim refazer esses batches depois (verificar)
                if len(S_work[c]) < self.m_per_class:  # Se não for a classe majoritária
                    S_work[c] = list(self.S[c])  # Reinicia a lista de imagens da classe
                    np.random.shuffle(S_work[c])  # Embaralha a lista novamente
                chosen = S_work[c][: self.m_per_class]
                S_work[c] = S_work[c][self.m_per_class :]
                batch.extend(chosen)
            yield batch
