import numpy as np

def em_algorithm(iterations, data, k, p0, p1):

    for _ in range(iterations):
        p = e_step(data, k, p0, p1) # responsibilities
        k, p0, p1 = m_step(data, p)
    return k, p0, p1

def e_step(data, k, p0, p1):
    p = np.zeros((3, 2)) # Responsibility
    for i, trial_outcome in enumerate(data):
        for j, coin in enumerate([0, 1]): # for j, in range(2):  # 2 coins  C0 and C1
                if j == 0:
                    p[i, j] = k * p0 ** (trial_outcome) * (1 - p0) ** (3 - trial_outcome)
                else:
                    p[i, j] = (1 - k) * p1 ** (trial_outcome) * (1 - p1) ** (3 - trial_outcome)

    p /= np.sum(p, axis=1, keepdims=True)
    #print(p)
    return p

def m_step(data, responsibilities):
    p = responsibilities
    k = np.sum(p[:, 0]) / len(data)
    #p0 = np.sum(np.sum(p * data, axis=0)[:, 0]) / np.sum(p[:, 0])
    #p1 = np.sum(np.sum(p * data, axis=0)[:, 1]) / np.sum(p[:, 1])
    w = (p.T @ data.reshape(-1, 1))
    a = w[0, 0]
    b = w[1, 0]
    p0 = a / (3 * np.sum( p[:, 0] ) )
    p1 = b / (3 * np.sum( p[:, 1] ) )
    print("p0 = ", p0)
    print("p1 = ", p1, "\n")

    return k, p0, p1

 
      


data = np.array([0, 2, 3])
k = 0.5
p0 = 0.6
p1 = 0.1
iterations = 1

k, p0, p1 = em_algorithm(iterations, data, k, p0, p1)
print(k, p0, p1)




