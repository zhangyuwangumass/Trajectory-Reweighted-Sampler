import numpy as np

class Noiser:
    def error_noise(self, data, label, SAMPLE_RATE=0.3):
        N = data.shape[0]
        M = int(SAMPLE_RATE * N)
        mask = np.random.choice(N, (M,), replace=False)
        label = (label[mask] + np.random.randint(1,10,(M,))) % 10

        return np.hstack((data[mask], label.reshape(M,1)))

    def gaussian_noise(self, data, label, mu=0, var=0.1, SAMPLE_RATE=0.3):
        N = data.shape[0]
        D = data.shape[1]
        M = int(SAMPLE_RATE * N)
        mask = np.random.choice(N, (M,), replace=False)
        noised_data = data[mask] + np.random.normal(mu, var, (M,D))

        return np.hstack((noised_data, label[mask].reshape(M,1)))

    def generate(self, data, label, error_param={'sample_rate':0.4}, gaussian_param={'sample_rate':0.1,'mu':0,'var':0.5}):
        N = data.shape[0]
        M = int(N * error_param['sample_rate'])
        P = int(N * gaussian_param['sample_rate'])
        mu = gaussian_param['mu']
        var = gaussian_param['var']

        error_noise = None
        gaussian_noise = None

        if M > 0:
            error_noise = self.error_noise(data[:M], label[:M], SAMPLE_RATE=1)
            error_index = np.arange(M).reshape(M,1)
            error_noise = np.hstack((error_noise, error_index))

        if P > 0:
            gaussian_noise = self.gaussian_noise(data[M:M+P], label[M:M+P], mu=mu, var=var, SAMPLE_RATE=1)
            gaussian_index = np.arange(M,M+P).reshape(P,1)
            gaussian_noise = np.hstack((gaussian_noise, gaussian_index))


        clean = np.hstack((data[M+P:], label[M+P:].reshape(N-M-P, 1)))
        clean = np.hstack((clean, np.arange(M+P,N).reshape(N-M-P, 1)))

        # 1 means error noise, -1 means gaussian noise, 0 means clean data
        if M > 0 and P > 0:
            data = np.vstack((error_noise, np.vstack((gaussian_noise, clean))))
            class_index = np.vstack((np.vstack((np.ones((M, 1)), -np.ones((P, 1)))), np.zeros((N - M - P, 1))))
        elif M > 0:
            data = np.vstack((error_noise, clean))
            class_index = np.vstack((np.ones((M, 1)), np.zeros((N - M - P, 1))))
        elif P > 0:
            data = np.vstack((gaussian_noise, clean))
            class_index = np.vstack((np.ones((M, 1)), -np.ones((P, 1))))
        else:
            data = clean

        #np.random.shuffle(data)
        return data, class_index

