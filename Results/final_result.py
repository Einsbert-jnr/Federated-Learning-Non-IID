
import matplotlib.pyplot as plt
import pickle

with open('history2.pickle', 'rb') as f:
    history2 = pickle.load(f)
f.close()
with open('history3.pickle', 'rb') as f:
    history3 = pickle.load(f)
f.close()
with open('history4.pickle', 'rb') as f:
    history4 = pickle.load(f)
f.close()


accuracy2 = history2.metrics_centralized['accuracy']
acc_values2 = [item[1] for item in accuracy2]
loss2 = history2.metrics_centralized['loss']
loss_values2 = [item[1] for item in loss2]

accuracy3 = history3.metrics_centralized['accuracy']
acc_values3 = [item[1] for item in accuracy3]
loss3 = history3.metrics_centralized['loss']
loss_values3 = [item[1] for item in loss3]

accuracy4 = history4.metrics_centralized['accuracy']
acc_values4 = [item[1] for item in accuracy4]
loss4 = history4.metrics_centralized['loss']
loss_values4 = [item[1] for item in loss4]

plt.plot(acc_values2)
plt.plot(acc_values3)
plt.plot(acc_values4)

plt.legend(['Accuracy'], loc = 'lower right')
plt.ylabel('Accuracy')
plt.xlabel('Rounds')
plt.title(f"Accuracy curve: Federated learning (Non-IID)")
plt.legend(['2 clients', '3 clients', '4 clients'], loc = 'lower right')
# plt.savefig('final_result1.png')
plt.show()