import numpy as np
import matplotlib.pyplot as plt
import json
import sys

def aktivacijska_fn1(z):
    return 2/(1+np.exp(-z)) - 1

def aktivacijska_fn1_derivacija(z):
    return 0.5*(1-np.square(z))

def aktivacijska_fn2(z): 
    return 1/(1+np.exp(-z))
    
def aktivacijska_fn2_derivacija(z):
    return z*(1-z)

class NN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_weights = np.random.rand(hidden_size, input_size)
        self.hidden_bias = np.random.rand(hidden_size)
        self.output_weights = np.random.rand(output_size, hidden_size)
        self.output_bias = np.random.rand(output_size)

    def forward(self, X):
        h = X @ self.hidden_weights.T + self.hidden_bias
        H = aktivacijska_fn1(h) 
        o = H @ self.output_weights.T + self.output_bias
        O = aktivacijska_fn2(o)
        
        return O
    
    def train(self, X, y, iterations = 500, learning_rate = 1, momentum = 0.9):
        loss_history = []

        W1 = self.hidden_weights
        b1 = self.hidden_bias
        W2 = self.output_weights
        b2 = self.output_bias
        
        vW1 = np.zeros_like(W1)
        vW2 = np.zeros_like(W2)
        vb1 = np.zeros_like(b1)
        vb2 = np.zeros_like(b2)

        for _ in range(iterations):
            # forward
            h = X @ W1.T + b1
            H = aktivacijska_fn1(h) 
            o = H @ W2.T + b2
            O = aktivacijska_fn2(o)

            # backprop
            error = y - O 
            do = error * aktivacijska_fn2_derivacija(O)
            dH = do @ W2
            dh = dH * aktivacijska_fn1_derivacija(H)
            dW2 = do.T @ H
            dW1 = dh.T @ X
            db2 = do.sum()
            db1 = dh.mean(axis=0)

            # momentum
            vW2 = learning_rate * dW2 + momentum * vW2
            vW1 = learning_rate * dW1 + momentum * vW1
            vb2 = learning_rate * db2 + momentum * vb2
            vb1 = learning_rate * db1 + momentum * vb1

            # optimization step
            W2 += vW2
            W1 += vW1
            b2 += vb2
            b1 += vb1

            # loss
            MS = np.mean(np.square(error))
            RMS = np.sqrt(MS)
            NRMS = RMS / np.sqrt(np.mean(np.square(O - np.mean(y))))

            loss = RMS
            loss_history.append(loss)


        preds = self.forward(X).tolist()
        print(f'{iterations} iteracija | Broj sakrivenih neurona: {len(self.hidden_bias)} | Stopa/brzina učenja: {learning_rate} | Momentum: {momentum}\n')
        print(f"Konačna greška (RMS): {loss}\n")
        for i, o in enumerate(preds):
            print(f'Ulazna vrijednost: {X[i]} --> Izlazna vrijednost: {o[0]:.3f}')

        plt.plot(range(iterations), loss_history)
        plt.xlabel("Iteracija")
        plt.ylabel("RMS")
        plt.title("Greška modela kroz iteracije")
        plt.show()

with open('podaci.json', 'r') as file1:
    content = file1.read()
    try:
        data = json.loads(content)
    except:
        print('Nije moguće učitat podatke.\nGreška formatiranja JSON file-a (provjeri podaci.json)')
        exit()

X = np.array(data["X"])
y = np.array(data["y"])

if len(X) != len(y):
    print('Veličina ulaznih i izlaznih podataka mora biti ista! (provjeri podaci.json)')
    exit()

args = sys.argv[1:]

if len(args) == 0:
    model = NN(len(X[0]), 2, len(y[0]))
    model.train(X, y)
else:
    if len(args) != 4:
        print('Pogrešan unos argumenata\nPotrebno je unijeti broj sakrivenih neurona, broj iteracija, brzinu učenja i momentum.\nTočan primjer: py main.py 2 500 1 0.9')
        exit()

    model = NN(input_size=len(X[0]), output_size=len(y[0]), hidden_size=int(args[0]))
    model.train(X, y, iterations=int(args[1]), learning_rate=float(args[2]), momentum=float(args[3]))