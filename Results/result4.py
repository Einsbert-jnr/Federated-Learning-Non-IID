import matplotlib.pyplot as plt
import pickle


NUM_CLIENTS = 4
with open('history4.pickle', 'rb') as f:
    history = pickle.load(f)
f.close()

print(history)

var = history.metrics_centralized['Centralised report']
print(type(var))
for i in var:
    for pair in i:
        print(pair)


accuracy = history.metrics_centralized['accuracy']
acc_values = [item[1] for item in accuracy]
loss = history.metrics_centralized['loss']
loss_values = [item[1] for item in loss]

plt.plot(acc_values)
plt.legend(['Accuracy'], loc = 'lower right')
plt.ylabel('Accuracy')
plt.xlabel('Rounds')
plt.title(f"Accuracy curve: Federated learning with {NUM_CLIENTS} clients")
plt.show()