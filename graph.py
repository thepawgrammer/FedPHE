import matplotlib.pyplot as plt

# Epoch별 accuracy / loss / time 리스트
accuracy = [93.17, 93.18, 96.44, 96.44, 97.3, 97.3, 97.76, 97.76, 97.89, 97.91]
loss = [0.237, 0.237, 0.106, 0.106, 0.081, 0.081, 0.066, 0.066, 0.058, 0.057]
time_list = [18.58, 1.05, 4.47, 1.06, 4.69, 1.03, 4.64, 1.07, 4.65, 1.06]
epochs = list(range(1, 11))  # 1~10 epoch

plt.figure(figsize=(10,4))

# Accuracy
plt.subplot(1,2,1)
plt.plot(epochs, accuracy, marker='o')
plt.ylim(90, 100)   # min=90, max=100
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
# 시작점 / 끝점 표시
plt.text(epochs[0], accuracy[0]+0.2, f"{accuracy[0]:.2f}%", ha='center')
plt.text(epochs[-1], accuracy[-1]+0.2, f"{accuracy[-1]:.2f}%", ha='center')

# Loss
plt.subplot(1,2,2)
plt.plot(epochs, loss, marker='o', color='red')
plt.ylim(0, 0.25)
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
# 시작점 / 끝점 표시
plt.text(epochs[0], loss[0]+0.01, f"{loss[0]:.3f}", ha='center')
plt.text(epochs[-1], loss[-1]-0.02, f"{loss[-1]:.3f}", ha='center')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(epochs, time_list, marker='o', color='green')
plt.ylim(0, 20)
plt.title("Epoch Training Time")
plt.xlabel("Epoch")
plt.ylabel("Time (s)")
plt.grid(True)
plt.show()