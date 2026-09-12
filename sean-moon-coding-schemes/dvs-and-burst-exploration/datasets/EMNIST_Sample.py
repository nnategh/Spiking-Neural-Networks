from torchvision import datasets, transforms
import matplotlib.pyplot as plt

emnist = datasets.EMNIST(root="./data", split="balanced", train=True, download=True,
                        transform=transforms.ToTensor())

img, label = emnist[0]

plt.imshow(img.squeeze(), cmap="gray")
plt.title("EMNIST sample")
plt.axis("off")
plt.show()