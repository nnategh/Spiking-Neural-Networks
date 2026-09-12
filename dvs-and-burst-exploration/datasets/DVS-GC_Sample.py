import tonic
import matplotlib.pyplot as plt

dataset = tonic.datasets.DVSGesture(save_to='./data', train=True)

events, label = dataset[0]

# simple visualization
plt.scatter(events['x'], events['y'], s=1)
plt.gca().invert_yaxis()
plt.title("DVS event frame")
plt.axis("off")
plt.show()
