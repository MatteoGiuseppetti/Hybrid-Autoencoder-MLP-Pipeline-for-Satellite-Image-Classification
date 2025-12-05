# Hybrid autoencoder–MLP pipeline for satellite image classification

## Project Goals

The objective of this project is to design and modify a convolutional autoencoder so that it can be effectively adapted to the task of satellite image classification.
The encoder is used as a feature extractor and its latent representations are employed to train a classifier on top of them.
The goal is to evaluate whether the learned latent space provides meaningful and discriminative features for classifying EuroSAT images.

## Dataset description

EuroSAT dataset (https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) is used for land use and land cover classification in geospatial imagery. The goal is to identify the semantic category represented in each satellite image.

The data consist of small RGB image patches captured by the Sentinel-2 satellite, and each patch corresponds to a specific type of land usage such as agricultural fields, forested areas, or urban environments.

The original EuroSAT dataset includes two folders:
- EuroSAT – containing RGB images
- EuroSATallBands – containing multispectral .tif images (all Sentinel-2 bands)

The dataset used in this project corresponds to the EuroSAT RGB subset, which contains 64×64 pixel RGB images derived from the Sentinel-2 multispectral satellite data. Each image has a Ground Sampling Distance (GSD) of 10 meters, meaning each pixel represents a 10×10 meter area on the ground.


```python
from torchvision.datasets import ImageFolder

data_root = r"C:/Users/Matti/Desktop/Progetto DL/Codice/Dataset/EuroSAT"
full_dataset = ImageFolder(
    root=data_root,
    transform=None  
)

# Class
print("Class names:", full_dataset.classes)

# Class-to-index mapping
print("Class-to-index mapping:", full_dataset.class_to_idx)
```

    Class names: ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    Class-to-index mapping: {'AnnualCrop': 0, 'Forest': 1, 'HerbaceousVegetation': 2, 'Highway': 3, 'Industrial': 4, 'Pasture': 5, 'PermanentCrop': 6, 'Residential': 7, 'River': 8, 'SeaLake': 9}
    


```python
import collections
import matplotlib.pyplot as plt

class_counts = collections.Counter([label for _, label in full_dataset])
labels = [full_dataset.classes[i] for i in class_counts.keys()]
counts = list(class_counts.values())

plt.figure(figsize=(12, 5))
plt.bar(labels, counts)
plt.title("Class Distribution in EuroSAT (RGB)")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of images")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()
```


    
![png](output_6_0.png)
    


The plot shows the distribution of the ten classes in the EuroSAT RGB dataset.
The dataset is originally well balanced, with each class containing approximately 2,000 to 3,000 images.
Since the total number of images in the dataset is very large, it is necessary to reduce the number of samples to make the project computationally feasible. Therefore, we kept a maximum of 2,000 images per class, matching the size of the smallest class to preserve class balance.


```python
import torch

per_class = 2000

selected_indices = []

for class_idx in range(len(full_dataset.classes)):
    class_all_idx = [i for i, (_, lbl) in enumerate(full_dataset) if lbl == class_idx]
    
    chosen = torch.randperm(len(class_all_idx))[:per_class]
    chosen = [class_all_idx[i] for i in chosen]
    
    selected_indices.extend(chosen)

reduced_dataset = torch.utils.data.Subset(full_dataset, selected_indices)

class_counts = collections.Counter([label for _, label in reduced_dataset])

labels = [full_dataset.classes[i] for i in class_counts.keys()]
counts = list(class_counts.values())

plt.figure(figsize=(12, 5))
plt.bar(labels, counts)
plt.title("Class Distribution in EuroSAT (2000 per class)")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of images")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()
```


    
![png](output_8_0.png)
    


I display a few sample images to show what the dataset looks like.


```python
import random

plt.figure(figsize=(10, 10))

for i in range(9):
    idx = random.randint(0, len(full_dataset)-1)
    img, label = full_dataset[idx]

    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(full_dataset.classes[label])
    plt.axis("off")

plt.suptitle("Sample Images from EuroSAT (RGB)", fontsize=16)
plt.show()
```


    
![png](output_10_0.png)
    


## Importing Libraries & Device Configuration

I import the libraries required for the project and set the computation device to GPU if available.


```python
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import transforms

from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
```


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

    Using device: cuda
    

## Data Preprocessing & DataLoader

To prepare the EuroSAT dataset for training, the reduced dataset is first split into three subsets:
70% for training, 15% for validation, and 15% for testing.
A fixed random seed ensures that the split is fully reproducible.


```python
train_size = int(0.7 * len(reduced_dataset))
val_size   = int(0.15 * len(reduced_dataset))
test_size  = len(reduced_dataset) - train_size - val_size

train_subset, val_subset, test_subset = random_split(
    reduced_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
```

Since the original ImageFolder transformation was disabled, a custom wrapper class (TransformDataset) is used to apply preprocessing steps after the dataset split. This guarantees that each subset receives the correct transformation without altering the underlying dataset structure.


```python
class TransformDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        img = self.transform(img)
        return img, label
```

A custom transformation (AddGaussianNoise) is implemented to inject random noise into the input images. Gaussian Noise consists of adding a random perturbation to the input data according to a normal distribution. This technique is commonly used to train denoising autoencoders and to improve model robustness. By slightly corrupting the input image, the encoder is forced to extract more stable and meaningful features, which enhances generalization and reduces overfitting. Moreover, the added noise acts as a form of regularization, making the latent space smoother and less sensitive to small variations in the data.


```python
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.03):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise
```

In the implemented preprocessing pipeline, the training set undergoes both data augmentation and noise injection through the following transformations: random horizontal flipping, random cropping conversion to tensors, and the addition of Gaussian noise. Conversely, the validation and test sets use only a minimal preprocessing step (conversion to tensors), ensuring that no augmentation or artificial perturbations affect model evaluation. These transformations are applied through a custom wrapper class (TransformDataset), which assigns the appropriate preprocessing to each subset after the dataset split.


```python
train_transform_ae = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.03)
])

test_val_transform = transforms.Compose([
    transforms.ToTensor()
])

trainset_ae = TransformDataset(train_subset, train_transform_ae)
valset_ae   = TransformDataset(val_subset,   test_val_transform)
testset_ae  = TransformDataset(test_subset,  test_val_transform)
```

DataLoaders are created for each subset with a batch size of 64.
Training data is shuffled at every epoch, while validation and test data are loaded in consistent order.


```python
batch_size = 64

train_loader = DataLoader(trainset_ae, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valset_ae, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(testset_ae, batch_size=batch_size, shuffle=False)
```

## Autoencoder

Autoencoders are neural architectures designed to learn compact and meaningful representations of data through an unsupervised reconstruction task. They consist of two components: an encoder, which compresses the input into a lower-dimensional latent representation, and a decoder, which reconstructs the original data from this latent code. By minimizing the difference between the input and the reconstruction, the network is encouraged to capture the most essential structures and patterns present in the data. When applied to images, convolutional autoencoders are particularly effective because convolutional layers naturally exploit spatial locality and hierarchical feature extraction.

Since the dataset consists of images, I adopt a Convolutional Autoencoder which is trained to reconstruct the input image and during this process it naturally learns strong spatial and semantic patterns.
The goal of this project is to repurpose these learned representations, not only for reconstruction, but also for image classification.

However, a traditional autoencoder optimizes only a reconstruction loss, which does not encourage the latent space to be discriminative for classification.
For this reason, I extend the model into a supervised autoencoder, adding a classification head and combining two losses: the reconstruction loss and the classification loss.
This forces the encoder to learn features that are useful for both reconstruction and class separation.

### Encoder

The encoder is responsible for transforming the input image into a compact latent representation that captures its most important spatial and semantic features. It progressively reduces the image resolution while increasing the number of channels, allowing the model to extract hierarchical patterns: from low-level textures to high-level structures relevant for land-use classification.

The architecture uses four convolutional blocks with stride 2, which reduce the spatial resolution in stages:

$$
64 \to 32 \to 16 \to 8 \to 4
$$

This depth is chosen because it provides a good balance between compression and information preservation. With fewer layers, the latent representation would remain too large and redundant; with more layers, the spatial information could be overly compressed and lost.

The number of channels increases across the layers $32 \to 64 \to 128 \to 256$ to compensate for the decreasing resolution. As the spatial dimension shrinks, the model is given more feature channels to encode richer and more abstract information. The final feature map ($256 \times 4 \times 4 = 4096$ values) provides enough capacity to capture complex patterns while remaining computationally manageable.

Stride-2 convolutions are preferred over max pooling because they perform learnable downsampling, enabling the network to decide how to compress spatial information in a way that remains invertible by the decoder.
Batch Normalization and ReLU activation promote stable training, faster convergence, and robustness to noise.

A fully connected layer then maps the 4096-dimensional tensor into the latent vector, whose dimensionality (latent_dim) can be tuned independently of the convolutional structure.

![image.png](60abfafa-493a-47aa-b7da-f11c73a901b5.png)


```python
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(256*4*4, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)
```

#### ReLU Activations
Each convolutional layer is followed by a ReLU activation:

$$
\text{ReLU}(x) = \max(0, x)
$$

ReLU introduces non-linearity and enables the network to learn flexible, non-linear decision boundaries in the latent space. ReLU also avoids vanishing gradients and keeps training efficient.

#### Batch Normalization
After each convolution, a `BatchNorm1d`, which normalizes the activations across the batch:

$$
\hat{x} = \frac{x - \mu}{\sigma}
$$

Batch normalization provides several key benefits:
* stabilizes and accelerates training;
* reduces internal covariate shift;
* smooths the optimization landscape;
* improves generalization.

Batch normalization is especially useful in convolutional encoders, where it helps maintain stable signal propagation throughout deep feature extraction.

### Decoder

The decoder mirrors the encoder, using transposed convolutions to gradually upsample the latent vector back to the original 64 x 64 x 3 image size.

This symmetric design ensures that each upsampling step in the decoder corresponds to a downsampling step in the encoder, helping the network reconstruct spatial structures at the appropriate scale.

The process involves a progressive reduction in channel depth, reflecting the reverse hierarchy of features:

$$
256 \to 128 \to 64 \to 32 \to 3
$$

This transition moves from abstract high-level representations back to pixel-level detail.

Transposed Convolutions are used to perform learnable upsampling. Batch Normalization and ReLU activation are applied after each intermediate layer to stabilize training and avoid vanishing gradients.
The final layer uses a Sigmoid activation function to constrain the output to the range [0, 1], matching the normalized input image format.

![image.png](1fa976d3-5ca8-4f79-a8d0-99692ec1aa51.png)


```python
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.decoder_input = nn.Linear(latent_dim, 256*4*4)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),    
            nn.Sigmoid()   
        )

    def forward(self, z):
        z = self.decoder_input(z)
        z = self.decoder(z)
        return z
```

#### Transposed Convolution

Transposed convolutions (also called deconvolutions) are used in decoder architectures to increase spatial resolution and reconstruct images from compressed latent representations. While standard convolutions reduce spatial dimensions by aggregating local information, transposed convolutions perform the opposite operation: they expand an input feature map while learning spatial patterns through a trainable kernel.

A transposed convolution can be understood as the transpose of the matrix operation underlying a regular convolution. Instead of sliding the kernel across the input, the kernel is effectively placed around each input element, producing overlapping patches that are summed together to form a larger output. This allows the layer to broadcast information spatially and recover higher-resolution feature maps.

Stride and padding also behave in reverse: stride controls the spacing of the expanded patches (thus enlarging the output) and padding is applied to the output rather than the input, ensuring correct output dimensions.

As in standard convolution, the layer learns a kernel per input channel and a set of kernels per output channel, making transposed convolution a natural counterpart to the encoder’s convolutional layers.

In this project, transposed convolutions are used to progressively upsample the latent vector back to a 64×64×3 RGB image. They provide a principled, learnable alternative to fixed upsampling methods and enable the decoder to reconstruct fine-grained spatial details from the latent space.

### The Latent Space and Classification Head
The latent space is the compressed representation produced by the encoder, where the original image is mapped into a lower dimensional vector that captures its most important structural and semantic features. In this space redundant pixel is removed while the patterns that are most relevant for distinguishing between different land are preserved. Because the latent space is both compact and semantically meaningful, it provides an ideal point for attaching a classifier.

To enable classification directly from this latent representation, a small multilayer perceptron (MLP) is added on top of the latent vector. This MLP consists of two fully connected layers:
$$
\text{latent\_dim} \to 128 \to \text{num\_classes}
$$
The architecture is intentionally lightweight: a shallow MLP is sufficient to model the class boundaries because the encoder already performs most of the feature extraction. Using a small classifier also keeps the model computationally efficient and ensures that most of the learning pressure is placed on the encoder rather than on the classifier itself.
This design encourages the encoder to organize the latent space so that it is not only useful for reconstruction but also discriminative for classification.


```python
class SupervisedAutoencoder(nn.Module):
    def __init__(self, latent_dim, num_classes=10):
        super().__init__()

        self.enc = Encoder(latent_dim)
        self.dec = Decoder(latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        z = self.enc(x)          
        x_hat = self.dec(z)      
        logits = self.classifier(z)  
        return x_hat, logits, z
```

## Loss function

The supervised autoencoder is trained by minimizing a composite loss function that jointly accounts for image reconstruction and class prediction:

$$
L = \alpha \cdot L_{\text{recon}} + L_{\text{class}}
$$

This formulation encourages the model to learn a latent representation that is simultaneously expressive enough to reconstruct the input and discriminative enough to support classification.

### Mean Squared Error
The reconstruction term is computed using Mean Squared Error, which measures the average squared difference between the input image $x$ and its reconstruction $\hat{x}$.

$$
L_{\text{recon}} = \| x - \hat{x} \|_2^2
$$

* A low MSE indicates that the decoder is able to faithfully reproduce fine-grained spatial and color information from the latent space.
* A high MSE reflects blurry or inaccurate reconstructions.

Because MSE penalizes larger errors more strongly than smaller ones, it naturally encourages smooth, noise-free reconstructions and stable learning dynamics.

### CrossEntropy
The classification component is computed using CrossEntropy, the standard objective for multi-class prediction. It quantifies the mismatch between the true class label and the predicted class probabilities produced by the classification head.

$$
L_{\text{class}} = - \sum_{c=1}^{C} y_c \, \log(\hat{p}_c)
$$

CrossEntropy encourages the model to assign high confidence to the correct class and penalizes ambiguous or incorrect predictions. This term ensures that the latent space contains class-discriminative information, complementing the reconstruction objective.

The scalar $\alpha$ controls the trade-off between the two objectives:

* Larger values give greater emphasis to reconstruction.
* Smaller values favor learning a stronger classifier.

A detailed discussion of how $\alpha$ influences the training dynamics and the final latent representation is provided in the following section.

### Estimating the relative scale of MSE and CrossEntropy at initialization

Before tuning the reconstruction weight $\alpha$, it is important to understand the natural scale of the two loss components that form the supervised autoencoder objective:

$$
L = \alpha \cdot L_{\text{recon}} + L_{\text{class}}
$$

Since the reconstruction loss (MSE) and the classification loss (CrossEntropy) operate on different numerical scales, choosing an appropriate value for $\alpha$ requires knowing how large these losses are relative to one another at initialization.

If one term is inherently much larger than the other, the model may favor it by default, making the balancing parameter $\alpha$ ineffective unless chosen accordingly.

To estimate this scale difference, the following experiment is performed:

1.  The supervised autoencoder initialize 1000 times each time with freshly sampled random weights (PyTorch initializes weights randomly).
2.  For each initialization, a single batch of training images is passed through the untrained model.
3.  Both the reconstruction loss (MSE) and the classification loss (CrossEntropy) are computed.
4.  The ratio is recorded:

$$
\text{ratio} = \frac{L_{\text{class}}}{L_{\text{recon}}}
$$

This indicates how many times larger the CrossEntropy is compared to the MSE. After repeating the procedure 1000 times, a histogram of the ratios is plotted, showing their  dominating the optimization process.


```python
ratios = []

for i in range(1000):
    model = SupervisedAutoencoder(latent_dim=128, num_classes=10).to(device)
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()

    imgs, labels = next(iter(train_loader))
    imgs = imgs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        x_hat, logits, _ = model(imgs)
        loss_recon = criterion_recon(x_hat, imgs)
        loss_class = criterion_class(logits, labels)

    ratio = loss_class.item() / loss_recon.item()
    ratios.append(ratio)

plt.hist(ratios, bins=20)
plt.title("Distribution CE/MSE")
plt.xlabel("Ratio CE/MSE")
plt.ylabel("Count")
plt.grid(alpha=0.3)
plt.show()
```


    
![png](output_44_0.png)
    


The distribution is approximately bell-shaped and centered around 30–32, with most values lying in the interval [25, 38]. This result shows that, before training, the classification loss is consistently an order of magnitude larger than the reconstruction loss ($L_{\text{recon}}$).

If alpha=1 were used, the MSE term would contribute very little to the total objective, and the optimization would be dominated by the classification component. To achieve a balanced training process where both reconstruction and classification influence the encoder, alpha should be scaled to approximately match this ratio.

### Hyperparameter Tuning for the Supervised Autoencoder

The behaviour of the supervised autoencoder during training is strongly influenced by two hyperparameters: the weight α assigned to the reconstruction term in the loss function and the learning rate that controls the dynamics of optimization. Although these parameters act on different aspects of the model, together they determine how the latent representation forms and how well the network balances reconstruction and classification.

### Effect of the reconstruction weight α
The parameter α regulates the importance of the reconstruction loss relative to the classification loss.

When α is very large, the training process becomes dominated by the reconstruction objective. In this regime, the decoder learns to reproduce the input images with high fidelity while the classification head receives comparatively weaker gradients. As a consequence, the latent representation tends to encode generic visual information rather than features that are discriminative for the classes, resulting in a high-quality reconstruction but a weak classifier.

When α is very small, the opposite happens. The optimization becomes almost entirely driven by the CrossEntropy loss. The latent space adapts primarily to separate the classes and, as a result, the model becomes effective at classification but significantly worse at reconstructing the inputs. The decoder receives little training signal and the reconstructions degrade, often becoming blurry or structurally inaccurate.

Only when α takes values of the right magnitude do the two objectives complement each other: the latent space retains both geometric structure for reconstruction and discriminative structure for classification, producing an effective compromise between the two tasks.

### Effect on the learning rate
The learning rate affects the model in a different but equally crucial way.

A learning rate that is too high can cause unstable behaviour during optimization. In such cases, the parameter updates become overly aggressive, and the loss may oscillate or diverge rather than decrease smoothly. This instability prevents the autoencoder from forming a coherent latent representation and often results in poor reconstructions and low classification accuracy.

If the learning rate is too small, the training becomes excessively slow and can stagnate. The model may converge toward suboptimal solutions simply because the updates are too small to escape shallow minima or plateaus in the loss landscape.

Only with an appropriately chosen learning rate does the training converge steadily, allowing the model to reduce both losses in a consistent and efficient manner.

## Adam Optimizer

Adam (Adaptive Moment Estimation) is a widely used optimization algorithm in deep learning because it provides stable and efficient training across a broad range of architectures. It combines two key ideas:

* Momentum. Adam maintains an exponentially weighted moving average of past gradients, which smooths the update direction, reduces oscillations, and accelerates convergence compared to standard SGD.

* RMSProp. At the same time, it tracks an exponentially weighted average of squared gradients, allowing each parameter to receive an adaptive learning rate based on the local curvature of the loss surface. This prevents the aggressive learning rate decay.

Adam integrates both mechanisms by estimating the first moment (mean of the gradients) and the second moment (variance). Because these estimates are initially biased toward zero, Adam applies bias-corrected versions of both quantities before computing the final parameter update, ensuring more stable behavior during the early stages of training.

The optimizer is controlled by a small set of hyperparameters:
* α: learning rate (default 0.001)
* β₁: decay rate for the first moment estimate (typically 0.9)
* β₂: decay rate for the second moment estimate (typically 0.999)
* ε: small constant for numerical stability

Adam is effective because it:
* adapts the learning rate individually for each parameter;
* stabilizes training through momentum and variance-based scaling;
* requires minimal hyperparameter tuning compared to SGD-based methods;
* performs reliably on large datasets and complex neural network architectures.

## Autoencoder training with grid search over α and learning rate

To understand how the supervised autoencoder behaves under different training conditions, I performed a grid search over the two hyperparameters. The grid explored spans α values between 20 and 40, guided by the earlier analysis of the CE/MSE scale ratio, and learning rates ranging from $10^{-4}$ to $10^{-1}$. By evaluating every combination, the training process reveals how sensitive the model is to each hyperparameter and which region of the search space leads to stable and well-balanced learning.

The model is trained using the Adam optimizer, which is generally a good default choice for deep neural networks. In practice, this often leads to smoother and faster convergence than plain stochastic gradient descent, especially in architectures that are deep or contain many convolutional layers, such as autoencoders.

For each pair (α,LR) in the grid, the autoencoder starts from scratch with randomly initialized weights. The model is then trained on the training set, while its performance is monitored on the validation set after every epoch. To avoid unnecessary overfitting or wasted computation, early stopping is employed: if the validation loss does not improve for a certain number of consecutive epochs, training is automatically halted. This ensures that each hyperparameter configuration is given enough opportunity to improve, but not so much that optimization drifts into ineffective or unstable regions.

Across all configurations, the validation loss reached during training is recorded. Whenever a model achieves a new lowest validation loss among all runs performed so far, it is marked as the current best model, and its weights are saved. This systematic exploration makes it possible to compare different combinations of reconstruction weight and learning rate in a fair and controlled manner.

At the end of the process, the combination with the lowest validation loss is selected as the optimal configuration. The corresponding model parameters are stored, along with the validation losses for all tested configurations, enabling further analysis of how the hyperparameters influence reconstruction quality, classification performance, and convergence behavior.


```python
#Folder
os.makedirs("models_best", exist_ok=True)

alpha_values = [20, 25, 30, 35, 40]
lr_values = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2, 1e-1]

results_val_losses = {}

# Tracking best model global
global_best_loss = float("inf")
global_best_info = None
global_best_model_state = None

best_train_losses = None
best_val_losses_curve = None

for alpha in alpha_values:
    for learning_rate in lr_values:

        print("\n=====================================")
        print(f"Training AE for α={alpha}, LR={learning_rate}")
        print("=====================================")

        latent_dim = 64
        model = SupervisedAutoencoder(latent_dim=latent_dim, num_classes=10).to(device)

        criterion_recon = nn.MSELoss()
        criterion_class = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        num_epochs = 80
        patience = 15
        counter = 0
        best_val_loss = float("inf")

        train_curve = []
        val_curve = []

        # Train loop
        for epoch in range(num_epochs):

            # Train
            model.train()
            train_loss = 0.0
            n_train = 0

            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                x_hat, logits, _ = model(imgs)

                loss_recon = criterion_recon(x_hat, imgs)
                loss_class = criterion_class(logits, labels)
                loss = alpha * loss_recon + loss_class

                loss.backward()
                optimizer.step()

                bs = imgs.size(0)
                train_loss += loss.item() * bs
                n_train += bs

            train_loss /= n_train
            train_curve.append(train_loss)

            # Validation
            model.eval()
            val_loss = 0.0
            n_val = 0

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)

                    x_hat, logits, _ = model(imgs)

                    loss_recon = criterion_recon(x_hat, imgs)
                    loss_class = criterion_class(logits, labels)
                    loss = alpha * loss_recon + loss_class

                    bs = imgs.size(0)
                    val_loss += loss.item() * bs
                    n_val += bs

            val_loss /= n_val
            val_curve.append(val_loss)

            print(f"[AE α={alpha} LR={learning_rate}] Epoch {epoch+1} | "
                  f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break
                    
        results_val_losses[(alpha, learning_rate)] = best_val_loss

        # Check global best
        if best_val_loss < global_best_loss:
            global_best_loss = best_val_loss
            global_best_info = (alpha, learning_rate)
            global_best_model_state = model.state_dict()

            best_train_losses = train_curve
            best_val_losses_curve = val_curve

            print("\nNew best AE")
            print(f"   α={alpha}, LR={learning_rate}, ValLoss={best_val_loss:.4f}")

# Save best model
best_alpha, best_lr = global_best_info
best_path = "models_best/AE_GLOBAL_BEST.pt"
torch.save(global_best_model_state, best_path)

print(f"   α={best_alpha}, LR={best_lr}")
print(f"   BEST ValLoss={global_best_loss:.4f}")
print(f"   Saved in: {best_path}\n")

# Save results
results_json = {
    f"alpha={a}, lr={lr}": float(v)
    for (a, lr), v in results_val_losses.items()
}

with open("models_best/validation_losses.json", "w") as f:
    json.dump(results_json, f, indent=4)
```

    
    =====================================
    Training AE for α=20, LR=0.0001
    =====================================
    [AE α=20 LR=0.0001] Epoch 1 | TrainLoss=2.1083 | ValLoss=1.6209
    [AE α=20 LR=0.0001] Epoch 2 | TrainLoss=1.2913 | ValLoss=1.5079
    [AE α=20 LR=0.0001] Epoch 3 | TrainLoss=1.0424 | ValLoss=1.8195
    [AE α=20 LR=0.0001] Epoch 4 | TrainLoss=0.9284 | ValLoss=2.0291
    [AE α=20 LR=0.0001] Epoch 5 | TrainLoss=0.8560 | ValLoss=1.8631
    [AE α=20 LR=0.0001] Epoch 6 | TrainLoss=0.8198 | ValLoss=2.0593
    [AE α=20 LR=0.0001] Epoch 7 | TrainLoss=0.7739 | ValLoss=1.9063
    [AE α=20 LR=0.0001] Epoch 8 | TrainLoss=0.7592 | ValLoss=1.8290
    [AE α=20 LR=0.0001] Epoch 9 | TrainLoss=0.7194 | ValLoss=2.2182
    [AE α=20 LR=0.0001] Epoch 10 | TrainLoss=0.6934 | ValLoss=1.9615
    [AE α=20 LR=0.0001] Epoch 11 | TrainLoss=0.6839 | ValLoss=1.9806
    [AE α=20 LR=0.0001] Epoch 12 | TrainLoss=0.6590 | ValLoss=2.3349
    [AE α=20 LR=0.0001] Epoch 13 | TrainLoss=0.6456 | ValLoss=1.8844
    [AE α=20 LR=0.0001] Epoch 14 | TrainLoss=0.6138 | ValLoss=2.0938
    [AE α=20 LR=0.0001] Epoch 15 | TrainLoss=0.5975 | ValLoss=2.2643
    [AE α=20 LR=0.0001] Epoch 16 | TrainLoss=0.5795 | ValLoss=2.1488
    [AE α=20 LR=0.0001] Epoch 17 | TrainLoss=0.5782 | ValLoss=2.4034
    Early stopping triggered.
    
    New best AE
       α=20, LR=0.0001, ValLoss=1.5079
    
    =====================================
    Training AE for α=20, LR=0.0002
    =====================================
    [AE α=20 LR=0.0002] Epoch 1 | TrainLoss=1.8779 | ValLoss=1.5736
    [AE α=20 LR=0.0002] Epoch 2 | TrainLoss=1.1503 | ValLoss=1.9440
    [AE α=20 LR=0.0002] Epoch 3 | TrainLoss=0.9551 | ValLoss=1.9564
    [AE α=20 LR=0.0002] Epoch 4 | TrainLoss=0.8446 | ValLoss=1.7549
    [AE α=20 LR=0.0002] Epoch 5 | TrainLoss=0.7920 | ValLoss=1.7250
    [AE α=20 LR=0.0002] Epoch 6 | TrainLoss=0.7465 | ValLoss=2.5632
    [AE α=20 LR=0.0002] Epoch 7 | TrainLoss=0.7225 | ValLoss=2.8620
    [AE α=20 LR=0.0002] Epoch 8 | TrainLoss=0.6772 | ValLoss=2.0710
    [AE α=20 LR=0.0002] Epoch 9 | TrainLoss=0.6411 | ValLoss=2.4973
    [AE α=20 LR=0.0002] Epoch 10 | TrainLoss=0.6092 | ValLoss=2.3756
    [AE α=20 LR=0.0002] Epoch 11 | TrainLoss=0.5902 | ValLoss=3.9829
    [AE α=20 LR=0.0002] Epoch 12 | TrainLoss=0.5714 | ValLoss=2.6231
    [AE α=20 LR=0.0002] Epoch 13 | TrainLoss=0.5518 | ValLoss=2.2663
    [AE α=20 LR=0.0002] Epoch 14 | TrainLoss=0.5250 | ValLoss=2.5911
    [AE α=20 LR=0.0002] Epoch 15 | TrainLoss=0.5131 | ValLoss=3.5556
    [AE α=20 LR=0.0002] Epoch 16 | TrainLoss=0.5048 | ValLoss=2.1457
    Early stopping triggered.
    
    =====================================
    Training AE for α=20, LR=0.0005
    =====================================
    [AE α=20 LR=0.0005] Epoch 1 | TrainLoss=1.5413 | ValLoss=1.4751
    [AE α=20 LR=0.0005] Epoch 2 | TrainLoss=1.0100 | ValLoss=3.1971
    [AE α=20 LR=0.0005] Epoch 3 | TrainLoss=0.8791 | ValLoss=2.2526
    [AE α=20 LR=0.0005] Epoch 4 | TrainLoss=0.8032 | ValLoss=3.3777
    [AE α=20 LR=0.0005] Epoch 5 | TrainLoss=0.7529 | ValLoss=2.5025
    [AE α=20 LR=0.0005] Epoch 6 | TrainLoss=0.6991 | ValLoss=1.0212
    [AE α=20 LR=0.0005] Epoch 7 | TrainLoss=0.6623 | ValLoss=2.4777
    [AE α=20 LR=0.0005] Epoch 8 | TrainLoss=0.6213 | ValLoss=2.7741
    [AE α=20 LR=0.0005] Epoch 9 | TrainLoss=0.5997 | ValLoss=1.9951
    [AE α=20 LR=0.0005] Epoch 10 | TrainLoss=0.5551 | ValLoss=2.8939
    [AE α=20 LR=0.0005] Epoch 11 | TrainLoss=0.5524 | ValLoss=2.3001
    [AE α=20 LR=0.0005] Epoch 12 | TrainLoss=0.5066 | ValLoss=1.3147
    [AE α=20 LR=0.0005] Epoch 13 | TrainLoss=0.4944 | ValLoss=1.0559
    [AE α=20 LR=0.0005] Epoch 14 | TrainLoss=0.4870 | ValLoss=1.6356
    [AE α=20 LR=0.0005] Epoch 15 | TrainLoss=0.4612 | ValLoss=3.3714
    [AE α=20 LR=0.0005] Epoch 16 | TrainLoss=0.4412 | ValLoss=2.8992
    [AE α=20 LR=0.0005] Epoch 17 | TrainLoss=0.4384 | ValLoss=1.3105
    [AE α=20 LR=0.0005] Epoch 18 | TrainLoss=0.4162 | ValLoss=2.0000
    [AE α=20 LR=0.0005] Epoch 19 | TrainLoss=0.4130 | ValLoss=1.0778
    [AE α=20 LR=0.0005] Epoch 20 | TrainLoss=0.4047 | ValLoss=2.5056
    [AE α=20 LR=0.0005] Epoch 21 | TrainLoss=0.3880 | ValLoss=2.3812
    Early stopping triggered.
    
    New best AE
       α=20, LR=0.0005, ValLoss=1.0212
    
    =====================================
    Training AE for α=20, LR=0.001
    =====================================
    [AE α=20 LR=0.001] Epoch 1 | TrainLoss=1.5505 | ValLoss=2.3492
    [AE α=20 LR=0.001] Epoch 2 | TrainLoss=1.0624 | ValLoss=1.7204
    [AE α=20 LR=0.001] Epoch 3 | TrainLoss=0.8874 | ValLoss=2.5232
    [AE α=20 LR=0.001] Epoch 4 | TrainLoss=0.8117 | ValLoss=1.8084
    [AE α=20 LR=0.001] Epoch 5 | TrainLoss=0.7609 | ValLoss=2.6492
    [AE α=20 LR=0.001] Epoch 6 | TrainLoss=0.7120 | ValLoss=1.1716
    [AE α=20 LR=0.001] Epoch 7 | TrainLoss=0.6605 | ValLoss=1.6905
    [AE α=20 LR=0.001] Epoch 8 | TrainLoss=0.6447 | ValLoss=1.3483
    [AE α=20 LR=0.001] Epoch 9 | TrainLoss=0.5868 | ValLoss=2.4041
    [AE α=20 LR=0.001] Epoch 10 | TrainLoss=0.5684 | ValLoss=1.5514
    [AE α=20 LR=0.001] Epoch 11 | TrainLoss=0.5392 | ValLoss=2.4776
    [AE α=20 LR=0.001] Epoch 12 | TrainLoss=0.5160 | ValLoss=0.9850
    [AE α=20 LR=0.001] Epoch 13 | TrainLoss=0.5191 | ValLoss=2.1727
    [AE α=20 LR=0.001] Epoch 14 | TrainLoss=0.4863 | ValLoss=1.9234
    [AE α=20 LR=0.001] Epoch 15 | TrainLoss=0.4724 | ValLoss=2.2616
    [AE α=20 LR=0.001] Epoch 16 | TrainLoss=0.4575 | ValLoss=1.7378
    [AE α=20 LR=0.001] Epoch 17 | TrainLoss=0.4509 | ValLoss=1.6350
    [AE α=20 LR=0.001] Epoch 18 | TrainLoss=0.4448 | ValLoss=1.5156
    [AE α=20 LR=0.001] Epoch 19 | TrainLoss=0.4259 | ValLoss=2.0892
    [AE α=20 LR=0.001] Epoch 20 | TrainLoss=0.4152 | ValLoss=1.0433
    [AE α=20 LR=0.001] Epoch 21 | TrainLoss=0.4060 | ValLoss=1.8985
    [AE α=20 LR=0.001] Epoch 22 | TrainLoss=0.3929 | ValLoss=1.0861
    [AE α=20 LR=0.001] Epoch 23 | TrainLoss=0.3863 | ValLoss=1.4441
    [AE α=20 LR=0.001] Epoch 24 | TrainLoss=0.3638 | ValLoss=3.6302
    [AE α=20 LR=0.001] Epoch 25 | TrainLoss=0.3775 | ValLoss=0.7853
    [AE α=20 LR=0.001] Epoch 26 | TrainLoss=0.3511 | ValLoss=2.2709
    [AE α=20 LR=0.001] Epoch 27 | TrainLoss=0.3511 | ValLoss=2.5334
    [AE α=20 LR=0.001] Epoch 28 | TrainLoss=0.3370 | ValLoss=1.6439
    [AE α=20 LR=0.001] Epoch 29 | TrainLoss=0.3446 | ValLoss=3.5335
    [AE α=20 LR=0.001] Epoch 30 | TrainLoss=0.3302 | ValLoss=3.3580
    [AE α=20 LR=0.001] Epoch 31 | TrainLoss=0.3246 | ValLoss=1.1107
    [AE α=20 LR=0.001] Epoch 32 | TrainLoss=0.3190 | ValLoss=2.1757
    [AE α=20 LR=0.001] Epoch 33 | TrainLoss=0.3144 | ValLoss=1.4910
    [AE α=20 LR=0.001] Epoch 34 | TrainLoss=0.3045 | ValLoss=1.1456
    [AE α=20 LR=0.001] Epoch 35 | TrainLoss=0.3133 | ValLoss=1.0568
    [AE α=20 LR=0.001] Epoch 36 | TrainLoss=0.3049 | ValLoss=1.7153
    [AE α=20 LR=0.001] Epoch 37 | TrainLoss=0.2976 | ValLoss=0.9218
    [AE α=20 LR=0.001] Epoch 38 | TrainLoss=0.2856 | ValLoss=3.0863
    [AE α=20 LR=0.001] Epoch 39 | TrainLoss=0.2869 | ValLoss=2.1260
    [AE α=20 LR=0.001] Epoch 40 | TrainLoss=0.2831 | ValLoss=1.3834
    Early stopping triggered.
    
    New best AE
       α=20, LR=0.001, ValLoss=0.7853
    
    =====================================
    Training AE for α=20, LR=0.002
    =====================================
    [AE α=20 LR=0.002] Epoch 1 | TrainLoss=1.5881 | ValLoss=3.7484
    [AE α=20 LR=0.002] Epoch 2 | TrainLoss=1.1208 | ValLoss=1.4351
    [AE α=20 LR=0.002] Epoch 3 | TrainLoss=0.9755 | ValLoss=1.1279
    [AE α=20 LR=0.002] Epoch 4 | TrainLoss=0.8629 | ValLoss=1.5457
    [AE α=20 LR=0.002] Epoch 5 | TrainLoss=0.8152 | ValLoss=1.7187
    [AE α=20 LR=0.002] Epoch 6 | TrainLoss=0.7641 | ValLoss=4.3365
    [AE α=20 LR=0.002] Epoch 7 | TrainLoss=0.7223 | ValLoss=1.3131
    [AE α=20 LR=0.002] Epoch 8 | TrainLoss=0.6696 | ValLoss=0.9857
    [AE α=20 LR=0.002] Epoch 9 | TrainLoss=0.6497 | ValLoss=0.6752
    [AE α=20 LR=0.002] Epoch 10 | TrainLoss=0.6436 | ValLoss=1.4343
    [AE α=20 LR=0.002] Epoch 11 | TrainLoss=0.5977 | ValLoss=1.2203
    [AE α=20 LR=0.002] Epoch 12 | TrainLoss=0.5673 | ValLoss=1.3564
    [AE α=20 LR=0.002] Epoch 13 | TrainLoss=0.5416 | ValLoss=1.1741
    [AE α=20 LR=0.002] Epoch 14 | TrainLoss=0.5335 | ValLoss=1.4473
    [AE α=20 LR=0.002] Epoch 15 | TrainLoss=0.4956 | ValLoss=1.0597
    [AE α=20 LR=0.002] Epoch 16 | TrainLoss=0.5026 | ValLoss=2.0478
    [AE α=20 LR=0.002] Epoch 17 | TrainLoss=0.4742 | ValLoss=1.9796
    [AE α=20 LR=0.002] Epoch 18 | TrainLoss=0.4543 | ValLoss=2.9860
    [AE α=20 LR=0.002] Epoch 19 | TrainLoss=0.4454 | ValLoss=1.2110
    [AE α=20 LR=0.002] Epoch 20 | TrainLoss=0.4366 | ValLoss=0.9363
    [AE α=20 LR=0.002] Epoch 21 | TrainLoss=0.4160 | ValLoss=1.2409
    [AE α=20 LR=0.002] Epoch 22 | TrainLoss=0.4059 | ValLoss=1.5993
    [AE α=20 LR=0.002] Epoch 23 | TrainLoss=0.4036 | ValLoss=0.8467
    [AE α=20 LR=0.002] Epoch 24 | TrainLoss=0.3906 | ValLoss=1.6142
    Early stopping triggered.
    
    New best AE
       α=20, LR=0.002, ValLoss=0.6752
    
    =====================================
    Training AE for α=20, LR=0.005
    =====================================
    [AE α=20 LR=0.005] Epoch 1 | TrainLoss=1.7538 | ValLoss=2.1624
    [AE α=20 LR=0.005] Epoch 2 | TrainLoss=1.2796 | ValLoss=3.3595
    [AE α=20 LR=0.005] Epoch 3 | TrainLoss=1.0543 | ValLoss=1.7898
    [AE α=20 LR=0.005] Epoch 4 | TrainLoss=0.9538 | ValLoss=3.9878
    [AE α=20 LR=0.005] Epoch 5 | TrainLoss=0.8928 | ValLoss=1.1437
    [AE α=20 LR=0.005] Epoch 6 | TrainLoss=0.8297 | ValLoss=0.9097
    [AE α=20 LR=0.005] Epoch 7 | TrainLoss=0.7854 | ValLoss=1.6761
    [AE α=20 LR=0.005] Epoch 8 | TrainLoss=0.7277 | ValLoss=2.0435
    [AE α=20 LR=0.005] Epoch 9 | TrainLoss=0.7289 | ValLoss=1.0243
    [AE α=20 LR=0.005] Epoch 10 | TrainLoss=0.6931 | ValLoss=2.7401
    [AE α=20 LR=0.005] Epoch 11 | TrainLoss=0.6550 | ValLoss=0.6753
    [AE α=20 LR=0.005] Epoch 12 | TrainLoss=0.6405 | ValLoss=0.8883
    [AE α=20 LR=0.005] Epoch 13 | TrainLoss=0.6112 | ValLoss=2.3834
    [AE α=20 LR=0.005] Epoch 14 | TrainLoss=0.5884 | ValLoss=1.0269
    [AE α=20 LR=0.005] Epoch 15 | TrainLoss=0.5487 | ValLoss=0.8223
    [AE α=20 LR=0.005] Epoch 16 | TrainLoss=0.5407 | ValLoss=3.8349
    [AE α=20 LR=0.005] Epoch 17 | TrainLoss=0.5241 | ValLoss=0.8015
    [AE α=20 LR=0.005] Epoch 18 | TrainLoss=0.5155 | ValLoss=1.9002
    [AE α=20 LR=0.005] Epoch 19 | TrainLoss=0.4992 | ValLoss=2.9735
    [AE α=20 LR=0.005] Epoch 20 | TrainLoss=0.4804 | ValLoss=1.2191
    [AE α=20 LR=0.005] Epoch 21 | TrainLoss=0.4907 | ValLoss=3.3074
    [AE α=20 LR=0.005] Epoch 22 | TrainLoss=0.4646 | ValLoss=0.8870
    [AE α=20 LR=0.005] Epoch 23 | TrainLoss=0.4516 | ValLoss=1.9430
    [AE α=20 LR=0.005] Epoch 24 | TrainLoss=0.4573 | ValLoss=2.3373
    [AE α=20 LR=0.005] Epoch 25 | TrainLoss=0.4276 | ValLoss=1.1230
    [AE α=20 LR=0.005] Epoch 26 | TrainLoss=0.4328 | ValLoss=3.1955
    Early stopping triggered.
    
    =====================================
    Training AE for α=20, LR=0.01
    =====================================
    [AE α=20 LR=0.01] Epoch 1 | TrainLoss=2.2634 | ValLoss=2.3065
    [AE α=20 LR=0.01] Epoch 2 | TrainLoss=1.5407 | ValLoss=2.4464
    [AE α=20 LR=0.01] Epoch 3 | TrainLoss=1.2554 | ValLoss=6.6820
    [AE α=20 LR=0.01] Epoch 4 | TrainLoss=1.1143 | ValLoss=2.0007
    [AE α=20 LR=0.01] Epoch 5 | TrainLoss=1.0069 | ValLoss=1.9809
    [AE α=20 LR=0.01] Epoch 6 | TrainLoss=0.9372 | ValLoss=1.7593
    [AE α=20 LR=0.01] Epoch 7 | TrainLoss=0.8935 | ValLoss=0.8082
    [AE α=20 LR=0.01] Epoch 8 | TrainLoss=0.8463 | ValLoss=1.9752
    [AE α=20 LR=0.01] Epoch 9 | TrainLoss=0.8171 | ValLoss=1.4131
    [AE α=20 LR=0.01] Epoch 10 | TrainLoss=0.7882 | ValLoss=0.9746
    [AE α=20 LR=0.01] Epoch 11 | TrainLoss=0.7772 | ValLoss=1.4253
    [AE α=20 LR=0.01] Epoch 12 | TrainLoss=0.7305 | ValLoss=2.9446
    [AE α=20 LR=0.01] Epoch 13 | TrainLoss=0.7101 | ValLoss=0.8316
    [AE α=20 LR=0.01] Epoch 14 | TrainLoss=0.6814 | ValLoss=0.8707
    [AE α=20 LR=0.01] Epoch 15 | TrainLoss=0.6654 | ValLoss=0.6349
    [AE α=20 LR=0.01] Epoch 16 | TrainLoss=0.6374 | ValLoss=1.3099
    [AE α=20 LR=0.01] Epoch 17 | TrainLoss=0.6528 | ValLoss=0.7430
    [AE α=20 LR=0.01] Epoch 18 | TrainLoss=0.6187 | ValLoss=1.2258
    [AE α=20 LR=0.01] Epoch 19 | TrainLoss=0.5874 | ValLoss=0.8045
    [AE α=20 LR=0.01] Epoch 20 | TrainLoss=0.5925 | ValLoss=0.7086
    [AE α=20 LR=0.01] Epoch 21 | TrainLoss=0.5588 | ValLoss=0.7674
    [AE α=20 LR=0.01] Epoch 22 | TrainLoss=0.5566 | ValLoss=0.9957
    [AE α=20 LR=0.01] Epoch 23 | TrainLoss=0.5593 | ValLoss=2.0360
    [AE α=20 LR=0.01] Epoch 24 | TrainLoss=0.5430 | ValLoss=2.6789
    [AE α=20 LR=0.01] Epoch 25 | TrainLoss=0.5303 | ValLoss=1.5626
    [AE α=20 LR=0.01] Epoch 26 | TrainLoss=0.5268 | ValLoss=0.6670
    [AE α=20 LR=0.01] Epoch 27 | TrainLoss=0.5273 | ValLoss=1.0829
    [AE α=20 LR=0.01] Epoch 28 | TrainLoss=0.5135 | ValLoss=1.9028
    [AE α=20 LR=0.01] Epoch 29 | TrainLoss=0.4794 | ValLoss=0.9127
    [AE α=20 LR=0.01] Epoch 30 | TrainLoss=0.4945 | ValLoss=1.0042
    Early stopping triggered.
    
    New best AE
       α=20, LR=0.01, ValLoss=0.6349
    
    =====================================
    Training AE for α=20, LR=0.05
    =====================================
    [AE α=20 LR=0.05] Epoch 1 | TrainLoss=5.5843 | ValLoss=2.5629
    [AE α=20 LR=0.05] Epoch 2 | TrainLoss=2.6524 | ValLoss=2.5273
    [AE α=20 LR=0.05] Epoch 3 | TrainLoss=2.6063 | ValLoss=2.5872
    [AE α=20 LR=0.05] Epoch 4 | TrainLoss=2.5497 | ValLoss=2.4894
    [AE α=20 LR=0.05] Epoch 5 | TrainLoss=2.5058 | ValLoss=2.4589
    [AE α=20 LR=0.05] Epoch 6 | TrainLoss=2.5000 | ValLoss=2.4650
    [AE α=20 LR=0.05] Epoch 7 | TrainLoss=2.4788 | ValLoss=2.4525
    [AE α=20 LR=0.05] Epoch 8 | TrainLoss=2.4730 | ValLoss=2.4429
    [AE α=20 LR=0.05] Epoch 9 | TrainLoss=2.4629 | ValLoss=2.4408
    [AE α=20 LR=0.05] Epoch 10 | TrainLoss=2.4563 | ValLoss=2.4352
    [AE α=20 LR=0.05] Epoch 11 | TrainLoss=2.4805 | ValLoss=2.4471
    [AE α=20 LR=0.05] Epoch 12 | TrainLoss=2.4576 | ValLoss=2.4345
    [AE α=20 LR=0.05] Epoch 13 | TrainLoss=2.4558 | ValLoss=2.4339
    [AE α=20 LR=0.05] Epoch 14 | TrainLoss=2.4503 | ValLoss=2.4258
    [AE α=20 LR=0.05] Epoch 15 | TrainLoss=2.4455 | ValLoss=2.4356
    [AE α=20 LR=0.05] Epoch 16 | TrainLoss=2.4465 | ValLoss=2.4319
    [AE α=20 LR=0.05] Epoch 17 | TrainLoss=2.4471 | ValLoss=2.4402
    [AE α=20 LR=0.05] Epoch 18 | TrainLoss=2.4439 | ValLoss=2.4224
    [AE α=20 LR=0.05] Epoch 19 | TrainLoss=2.6139 | ValLoss=2.5319
    [AE α=20 LR=0.05] Epoch 20 | TrainLoss=2.5559 | ValLoss=2.5462
    [AE α=20 LR=0.05] Epoch 21 | TrainLoss=7.5112 | ValLoss=2.6271
    [AE α=20 LR=0.05] Epoch 22 | TrainLoss=2.5867 | ValLoss=2.4912
    [AE α=20 LR=0.05] Epoch 23 | TrainLoss=2.5102 | ValLoss=2.4817
    [AE α=20 LR=0.05] Epoch 24 | TrainLoss=2.4879 | ValLoss=2.4532
    [AE α=20 LR=0.05] Epoch 25 | TrainLoss=2.4720 | ValLoss=2.4362
    [AE α=20 LR=0.05] Epoch 26 | TrainLoss=2.4647 | ValLoss=2.4334
    [AE α=20 LR=0.05] Epoch 27 | TrainLoss=2.4655 | ValLoss=2.5244
    [AE α=20 LR=0.05] Epoch 28 | TrainLoss=2.4772 | ValLoss=2.4414
    [AE α=20 LR=0.05] Epoch 29 | TrainLoss=2.4556 | ValLoss=2.4299
    [AE α=20 LR=0.05] Epoch 30 | TrainLoss=2.4502 | ValLoss=2.4230
    [AE α=20 LR=0.05] Epoch 31 | TrainLoss=2.4494 | ValLoss=2.4268
    [AE α=20 LR=0.05] Epoch 32 | TrainLoss=2.4453 | ValLoss=2.4226
    [AE α=20 LR=0.05] Epoch 33 | TrainLoss=2.4463 | ValLoss=2.4234
    Early stopping triggered.
    
    =====================================
    Training AE for α=20, LR=0.1
    =====================================
    [AE α=20 LR=0.1] Epoch 1 | TrainLoss=19.0584 | ValLoss=2.7347
    [AE α=20 LR=0.1] Epoch 2 | TrainLoss=2.7652 | ValLoss=2.5769
    [AE α=20 LR=0.1] Epoch 3 | TrainLoss=2.6522 | ValLoss=2.6075
    [AE α=20 LR=0.1] Epoch 4 | TrainLoss=2.6079 | ValLoss=2.5488
    [AE α=20 LR=0.1] Epoch 5 | TrainLoss=2.5682 | ValLoss=2.5511
    [AE α=20 LR=0.1] Epoch 6 | TrainLoss=2.5402 | ValLoss=2.5025
    [AE α=20 LR=0.1] Epoch 7 | TrainLoss=2.5435 | ValLoss=2.4845
    [AE α=20 LR=0.1] Epoch 8 | TrainLoss=2.5099 | ValLoss=2.4802
    [AE α=20 LR=0.1] Epoch 9 | TrainLoss=2.4967 | ValLoss=2.4717
    [AE α=20 LR=0.1] Epoch 10 | TrainLoss=2.4951 | ValLoss=2.4674
    [AE α=20 LR=0.1] Epoch 11 | TrainLoss=2.4900 | ValLoss=2.4752
    [AE α=20 LR=0.1] Epoch 12 | TrainLoss=2.4815 | ValLoss=2.4699
    [AE α=20 LR=0.1] Epoch 13 | TrainLoss=2.4858 | ValLoss=2.4901
    [AE α=20 LR=0.1] Epoch 14 | TrainLoss=2.4897 | ValLoss=2.4629
    [AE α=20 LR=0.1] Epoch 15 | TrainLoss=2.4792 | ValLoss=2.4621
    [AE α=20 LR=0.1] Epoch 16 | TrainLoss=2.5000 | ValLoss=2.4503
    [AE α=20 LR=0.1] Epoch 17 | TrainLoss=2.4740 | ValLoss=2.4754
    [AE α=20 LR=0.1] Epoch 18 | TrainLoss=2.4710 | ValLoss=2.4475
    [AE α=20 LR=0.1] Epoch 19 | TrainLoss=2.4691 | ValLoss=2.4459
    [AE α=20 LR=0.1] Epoch 20 | TrainLoss=2.4666 | ValLoss=2.4525
    [AE α=20 LR=0.1] Epoch 21 | TrainLoss=2.4663 | ValLoss=2.4505
    [AE α=20 LR=0.1] Epoch 22 | TrainLoss=2.4643 | ValLoss=2.4329
    [AE α=20 LR=0.1] Epoch 23 | TrainLoss=3.0864 | ValLoss=2.6759
    [AE α=20 LR=0.1] Epoch 24 | TrainLoss=2.6705 | ValLoss=2.5325
    [AE α=20 LR=0.1] Epoch 25 | TrainLoss=2.5880 | ValLoss=2.5087
    [AE α=20 LR=0.1] Epoch 26 | TrainLoss=2.5464 | ValLoss=2.5443
    [AE α=20 LR=0.1] Epoch 27 | TrainLoss=2.5270 | ValLoss=2.4825
    [AE α=20 LR=0.1] Epoch 28 | TrainLoss=2.5011 | ValLoss=2.4455
    [AE α=20 LR=0.1] Epoch 29 | TrainLoss=2.4983 | ValLoss=2.4799
    [AE α=20 LR=0.1] Epoch 30 | TrainLoss=2.4896 | ValLoss=3.9449
    [AE α=20 LR=0.1] Epoch 31 | TrainLoss=2.5793 | ValLoss=2.5820
    [AE α=20 LR=0.1] Epoch 32 | TrainLoss=2.6320 | ValLoss=2.5236
    [AE α=20 LR=0.1] Epoch 33 | TrainLoss=2.5545 | ValLoss=2.4903
    [AE α=20 LR=0.1] Epoch 34 | TrainLoss=2.5098 | ValLoss=2.4895
    [AE α=20 LR=0.1] Epoch 35 | TrainLoss=2.5001 | ValLoss=2.4514
    [AE α=20 LR=0.1] Epoch 36 | TrainLoss=2.4935 | ValLoss=2.4420
    [AE α=20 LR=0.1] Epoch 37 | TrainLoss=2.4874 | ValLoss=2.4312
    [AE α=20 LR=0.1] Epoch 38 | TrainLoss=2.4791 | ValLoss=2.4423
    [AE α=20 LR=0.1] Epoch 39 | TrainLoss=2.4786 | ValLoss=2.4424
    [AE α=20 LR=0.1] Epoch 40 | TrainLoss=2.4757 | ValLoss=2.4385
    [AE α=20 LR=0.1] Epoch 41 | TrainLoss=2.4779 | ValLoss=2.4242
    [AE α=20 LR=0.1] Epoch 42 | TrainLoss=2.4734 | ValLoss=2.4263
    [AE α=20 LR=0.1] Epoch 43 | TrainLoss=2.4730 | ValLoss=2.4354
    [AE α=20 LR=0.1] Epoch 44 | TrainLoss=2.4718 | ValLoss=2.4355
    [AE α=20 LR=0.1] Epoch 45 | TrainLoss=187.2765 | ValLoss=2.6565
    [AE α=20 LR=0.1] Epoch 46 | TrainLoss=2.6767 | ValLoss=2.6432
    [AE α=20 LR=0.1] Epoch 47 | TrainLoss=2.6675 | ValLoss=2.6263
    [AE α=20 LR=0.1] Epoch 48 | TrainLoss=2.6088 | ValLoss=2.5376
    [AE α=20 LR=0.1] Epoch 49 | TrainLoss=2.5804 | ValLoss=2.5213
    [AE α=20 LR=0.1] Epoch 50 | TrainLoss=2.5602 | ValLoss=2.5854
    [AE α=20 LR=0.1] Epoch 51 | TrainLoss=2.5541 | ValLoss=2.5278
    [AE α=20 LR=0.1] Epoch 52 | TrainLoss=2.5512 | ValLoss=2.5063
    [AE α=20 LR=0.1] Epoch 53 | TrainLoss=2.5480 | ValLoss=2.5118
    [AE α=20 LR=0.1] Epoch 54 | TrainLoss=2.5397 | ValLoss=2.5021
    [AE α=20 LR=0.1] Epoch 55 | TrainLoss=2.5400 | ValLoss=2.5207
    [AE α=20 LR=0.1] Epoch 56 | TrainLoss=2.5430 | ValLoss=2.5347
    Early stopping triggered.
    
    =====================================
    Training AE for α=25, LR=0.0001
    =====================================
    [AE α=25 LR=0.0001] Epoch 1 | TrainLoss=2.1941 | ValLoss=1.6427
    [AE α=25 LR=0.0001] Epoch 2 | TrainLoss=1.3535 | ValLoss=1.9139
    [AE α=25 LR=0.0001] Epoch 3 | TrainLoss=1.1114 | ValLoss=2.3625
    [AE α=25 LR=0.0001] Epoch 4 | TrainLoss=0.9870 | ValLoss=2.0296
    [AE α=25 LR=0.0001] Epoch 5 | TrainLoss=0.9056 | ValLoss=1.9193
    [AE α=25 LR=0.0001] Epoch 6 | TrainLoss=0.8694 | ValLoss=1.9101
    [AE α=25 LR=0.0001] Epoch 7 | TrainLoss=0.8075 | ValLoss=1.5055
    [AE α=25 LR=0.0001] Epoch 8 | TrainLoss=0.8001 | ValLoss=2.4595
    [AE α=25 LR=0.0001] Epoch 9 | TrainLoss=0.7636 | ValLoss=2.1807
    [AE α=25 LR=0.0001] Epoch 10 | TrainLoss=0.7358 | ValLoss=2.4232
    [AE α=25 LR=0.0001] Epoch 11 | TrainLoss=0.7141 | ValLoss=1.8717
    [AE α=25 LR=0.0001] Epoch 12 | TrainLoss=0.6806 | ValLoss=2.1803
    [AE α=25 LR=0.0001] Epoch 13 | TrainLoss=0.6798 | ValLoss=2.2364
    [AE α=25 LR=0.0001] Epoch 14 | TrainLoss=0.6495 | ValLoss=2.3118
    [AE α=25 LR=0.0001] Epoch 15 | TrainLoss=0.6357 | ValLoss=2.4421
    [AE α=25 LR=0.0001] Epoch 16 | TrainLoss=0.6161 | ValLoss=2.0908
    [AE α=25 LR=0.0001] Epoch 17 | TrainLoss=0.6092 | ValLoss=2.9291
    [AE α=25 LR=0.0001] Epoch 18 | TrainLoss=0.5930 | ValLoss=2.3353
    [AE α=25 LR=0.0001] Epoch 19 | TrainLoss=0.5791 | ValLoss=2.6814
    [AE α=25 LR=0.0001] Epoch 20 | TrainLoss=0.5615 | ValLoss=2.8827
    [AE α=25 LR=0.0001] Epoch 21 | TrainLoss=0.5559 | ValLoss=2.8041
    [AE α=25 LR=0.0001] Epoch 22 | TrainLoss=0.5422 | ValLoss=2.3287
    Early stopping triggered.
    
    =====================================
    Training AE for α=25, LR=0.0002
    =====================================
    [AE α=25 LR=0.0002] Epoch 1 | TrainLoss=2.1651 | ValLoss=1.3572
    [AE α=25 LR=0.0002] Epoch 2 | TrainLoss=1.2078 | ValLoss=1.4234
    [AE α=25 LR=0.0002] Epoch 3 | TrainLoss=1.0084 | ValLoss=1.8362
    [AE α=25 LR=0.0002] Epoch 4 | TrainLoss=0.8776 | ValLoss=1.3414
    [AE α=25 LR=0.0002] Epoch 5 | TrainLoss=0.8311 | ValLoss=1.4337
    [AE α=25 LR=0.0002] Epoch 6 | TrainLoss=0.8009 | ValLoss=1.2630
    [AE α=25 LR=0.0002] Epoch 7 | TrainLoss=0.7454 | ValLoss=1.6746
    [AE α=25 LR=0.0002] Epoch 8 | TrainLoss=0.7138 | ValLoss=1.2744
    [AE α=25 LR=0.0002] Epoch 9 | TrainLoss=0.6936 | ValLoss=1.2554
    [AE α=25 LR=0.0002] Epoch 10 | TrainLoss=0.6651 | ValLoss=1.5842
    [AE α=25 LR=0.0002] Epoch 11 | TrainLoss=0.6430 | ValLoss=2.9947
    [AE α=25 LR=0.0002] Epoch 12 | TrainLoss=0.6274 | ValLoss=1.7677
    [AE α=25 LR=0.0002] Epoch 13 | TrainLoss=0.5966 | ValLoss=2.0809
    [AE α=25 LR=0.0002] Epoch 14 | TrainLoss=0.5804 | ValLoss=2.0752
    [AE α=25 LR=0.0002] Epoch 15 | TrainLoss=0.5590 | ValLoss=2.2360
    [AE α=25 LR=0.0002] Epoch 16 | TrainLoss=0.5541 | ValLoss=2.1490
    [AE α=25 LR=0.0002] Epoch 17 | TrainLoss=0.5364 | ValLoss=1.9564
    [AE α=25 LR=0.0002] Epoch 18 | TrainLoss=0.5090 | ValLoss=3.4199
    [AE α=25 LR=0.0002] Epoch 19 | TrainLoss=0.5035 | ValLoss=1.8867
    [AE α=25 LR=0.0002] Epoch 20 | TrainLoss=0.4921 | ValLoss=1.6850
    [AE α=25 LR=0.0002] Epoch 21 | TrainLoss=0.4846 | ValLoss=2.1065
    [AE α=25 LR=0.0002] Epoch 22 | TrainLoss=0.4719 | ValLoss=2.8100
    [AE α=25 LR=0.0002] Epoch 23 | TrainLoss=0.4576 | ValLoss=2.3402
    [AE α=25 LR=0.0002] Epoch 24 | TrainLoss=0.4491 | ValLoss=1.6484
    Early stopping triggered.
    
    =====================================
    Training AE for α=25, LR=0.0005
    =====================================
    [AE α=25 LR=0.0005] Epoch 1 | TrainLoss=1.7899 | ValLoss=1.1055
    [AE α=25 LR=0.0005] Epoch 2 | TrainLoss=1.0990 | ValLoss=1.0659
    [AE α=25 LR=0.0005] Epoch 3 | TrainLoss=0.9171 | ValLoss=1.2473
    [AE α=25 LR=0.0005] Epoch 4 | TrainLoss=0.8255 | ValLoss=1.0443
    [AE α=25 LR=0.0005] Epoch 5 | TrainLoss=0.7530 | ValLoss=1.9138
    [AE α=25 LR=0.0005] Epoch 6 | TrainLoss=0.7245 | ValLoss=0.8925
    [AE α=25 LR=0.0005] Epoch 7 | TrainLoss=0.6809 | ValLoss=1.1084
    [AE α=25 LR=0.0005] Epoch 8 | TrainLoss=0.6620 | ValLoss=1.0146
    [AE α=25 LR=0.0005] Epoch 9 | TrainLoss=0.6263 | ValLoss=1.0017
    [AE α=25 LR=0.0005] Epoch 10 | TrainLoss=0.5931 | ValLoss=0.8435
    [AE α=25 LR=0.0005] Epoch 11 | TrainLoss=0.5538 | ValLoss=1.3052
    [AE α=25 LR=0.0005] Epoch 12 | TrainLoss=0.5470 | ValLoss=0.9986
    [AE α=25 LR=0.0005] Epoch 13 | TrainLoss=0.5363 | ValLoss=1.7142
    [AE α=25 LR=0.0005] Epoch 14 | TrainLoss=0.4998 | ValLoss=1.6511
    [AE α=25 LR=0.0005] Epoch 15 | TrainLoss=0.4955 | ValLoss=1.1500
    [AE α=25 LR=0.0005] Epoch 16 | TrainLoss=0.4896 | ValLoss=0.6993
    [AE α=25 LR=0.0005] Epoch 17 | TrainLoss=0.4692 | ValLoss=1.0670
    [AE α=25 LR=0.0005] Epoch 18 | TrainLoss=0.4428 | ValLoss=1.0940
    [AE α=25 LR=0.0005] Epoch 19 | TrainLoss=0.4361 | ValLoss=1.7393
    [AE α=25 LR=0.0005] Epoch 20 | TrainLoss=0.4415 | ValLoss=1.5329
    [AE α=25 LR=0.0005] Epoch 21 | TrainLoss=0.4233 | ValLoss=1.2158
    [AE α=25 LR=0.0005] Epoch 22 | TrainLoss=0.4233 | ValLoss=1.2929
    [AE α=25 LR=0.0005] Epoch 23 | TrainLoss=0.4173 | ValLoss=1.0966
    [AE α=25 LR=0.0005] Epoch 24 | TrainLoss=0.4037 | ValLoss=0.8458
    [AE α=25 LR=0.0005] Epoch 25 | TrainLoss=0.3935 | ValLoss=1.3121
    [AE α=25 LR=0.0005] Epoch 26 | TrainLoss=0.3866 | ValLoss=1.0049
    [AE α=25 LR=0.0005] Epoch 27 | TrainLoss=0.3720 | ValLoss=2.0998
    [AE α=25 LR=0.0005] Epoch 28 | TrainLoss=0.3715 | ValLoss=1.7113
    [AE α=25 LR=0.0005] Epoch 29 | TrainLoss=0.3600 | ValLoss=2.7298
    [AE α=25 LR=0.0005] Epoch 30 | TrainLoss=0.3523 | ValLoss=1.0932
    [AE α=25 LR=0.0005] Epoch 31 | TrainLoss=0.3482 | ValLoss=1.7542
    Early stopping triggered.
    
    =====================================
    Training AE for α=25, LR=0.001
    =====================================
    [AE α=25 LR=0.001] Epoch 1 | TrainLoss=1.6143 | ValLoss=1.3820
    [AE α=25 LR=0.001] Epoch 2 | TrainLoss=1.0565 | ValLoss=0.9965
    [AE α=25 LR=0.001] Epoch 3 | TrainLoss=0.9254 | ValLoss=1.9151
    [AE α=25 LR=0.001] Epoch 4 | TrainLoss=0.8432 | ValLoss=3.5383
    [AE α=25 LR=0.001] Epoch 5 | TrainLoss=0.8116 | ValLoss=1.5411
    [AE α=25 LR=0.001] Epoch 6 | TrainLoss=0.7409 | ValLoss=1.0533
    [AE α=25 LR=0.001] Epoch 7 | TrainLoss=0.7011 | ValLoss=1.2941
    [AE α=25 LR=0.001] Epoch 8 | TrainLoss=0.6596 | ValLoss=1.3572
    [AE α=25 LR=0.001] Epoch 9 | TrainLoss=0.6555 | ValLoss=2.7707
    [AE α=25 LR=0.001] Epoch 10 | TrainLoss=0.5947 | ValLoss=1.1330
    [AE α=25 LR=0.001] Epoch 11 | TrainLoss=0.5764 | ValLoss=2.9578
    [AE α=25 LR=0.001] Epoch 12 | TrainLoss=0.5463 | ValLoss=2.5396
    [AE α=25 LR=0.001] Epoch 13 | TrainLoss=0.5238 | ValLoss=0.6746
    [AE α=25 LR=0.001] Epoch 14 | TrainLoss=0.5224 | ValLoss=2.5653
    [AE α=25 LR=0.001] Epoch 15 | TrainLoss=0.4924 | ValLoss=2.4357
    [AE α=25 LR=0.001] Epoch 16 | TrainLoss=0.4732 | ValLoss=3.4100
    [AE α=25 LR=0.001] Epoch 17 | TrainLoss=0.4451 | ValLoss=3.8740
    [AE α=25 LR=0.001] Epoch 18 | TrainLoss=0.4455 | ValLoss=1.5396
    [AE α=25 LR=0.001] Epoch 19 | TrainLoss=0.4410 | ValLoss=2.1184
    [AE α=25 LR=0.001] Epoch 20 | TrainLoss=0.4270 | ValLoss=1.7938
    [AE α=25 LR=0.001] Epoch 21 | TrainLoss=0.4091 | ValLoss=1.1460
    [AE α=25 LR=0.001] Epoch 22 | TrainLoss=0.3999 | ValLoss=2.0758
    [AE α=25 LR=0.001] Epoch 23 | TrainLoss=0.3919 | ValLoss=1.6381
    [AE α=25 LR=0.001] Epoch 24 | TrainLoss=0.3861 | ValLoss=2.3632
    [AE α=25 LR=0.001] Epoch 25 | TrainLoss=0.3718 | ValLoss=4.1262
    [AE α=25 LR=0.001] Epoch 26 | TrainLoss=0.3763 | ValLoss=2.2000
    [AE α=25 LR=0.001] Epoch 27 | TrainLoss=0.3643 | ValLoss=2.4936
    [AE α=25 LR=0.001] Epoch 28 | TrainLoss=0.3621 | ValLoss=2.1319
    Early stopping triggered.
    
    =====================================
    Training AE for α=25, LR=0.002
    =====================================
    [AE α=25 LR=0.002] Epoch 1 | TrainLoss=1.7632 | ValLoss=1.5454
    [AE α=25 LR=0.002] Epoch 2 | TrainLoss=1.1961 | ValLoss=1.6524
    [AE α=25 LR=0.002] Epoch 3 | TrainLoss=0.9911 | ValLoss=1.6085
    [AE α=25 LR=0.002] Epoch 4 | TrainLoss=0.8932 | ValLoss=2.0026
    [AE α=25 LR=0.002] Epoch 5 | TrainLoss=0.8656 | ValLoss=1.1118
    [AE α=25 LR=0.002] Epoch 6 | TrainLoss=0.8196 | ValLoss=0.7830
    [AE α=25 LR=0.002] Epoch 7 | TrainLoss=0.7691 | ValLoss=0.9502
    [AE α=25 LR=0.002] Epoch 8 | TrainLoss=0.7249 | ValLoss=2.2229
    [AE α=25 LR=0.002] Epoch 9 | TrainLoss=0.7156 | ValLoss=1.9799
    [AE α=25 LR=0.002] Epoch 10 | TrainLoss=0.6709 | ValLoss=0.8115
    [AE α=25 LR=0.002] Epoch 11 | TrainLoss=0.6589 | ValLoss=1.0532
    [AE α=25 LR=0.002] Epoch 12 | TrainLoss=0.6129 | ValLoss=1.4489
    [AE α=25 LR=0.002] Epoch 13 | TrainLoss=0.5760 | ValLoss=1.1972
    [AE α=25 LR=0.002] Epoch 14 | TrainLoss=0.5796 | ValLoss=1.0661
    [AE α=25 LR=0.002] Epoch 15 | TrainLoss=0.5511 | ValLoss=1.3412
    [AE α=25 LR=0.002] Epoch 16 | TrainLoss=0.5177 | ValLoss=1.2825
    [AE α=25 LR=0.002] Epoch 17 | TrainLoss=0.5194 | ValLoss=1.4271
    [AE α=25 LR=0.002] Epoch 18 | TrainLoss=0.4962 | ValLoss=1.0489
    [AE α=25 LR=0.002] Epoch 19 | TrainLoss=0.4814 | ValLoss=0.8568
    [AE α=25 LR=0.002] Epoch 20 | TrainLoss=0.4924 | ValLoss=1.3381
    [AE α=25 LR=0.002] Epoch 21 | TrainLoss=0.4584 | ValLoss=1.3085
    Early stopping triggered.
    
    =====================================
    Training AE for α=25, LR=0.005
    =====================================
    [AE α=25 LR=0.005] Epoch 1 | TrainLoss=1.8679 | ValLoss=2.7517
    [AE α=25 LR=0.005] Epoch 2 | TrainLoss=1.2963 | ValLoss=3.4950
    [AE α=25 LR=0.005] Epoch 3 | TrainLoss=1.0950 | ValLoss=2.2528
    [AE α=25 LR=0.005] Epoch 4 | TrainLoss=0.9863 | ValLoss=0.8521
    [AE α=25 LR=0.005] Epoch 5 | TrainLoss=0.9244 | ValLoss=1.8467
    [AE α=25 LR=0.005] Epoch 6 | TrainLoss=0.8949 | ValLoss=1.0966
    [AE α=25 LR=0.005] Epoch 7 | TrainLoss=0.8352 | ValLoss=0.8912
    [AE α=25 LR=0.005] Epoch 8 | TrainLoss=0.8095 | ValLoss=1.7900
    [AE α=25 LR=0.005] Epoch 9 | TrainLoss=0.7839 | ValLoss=0.8947
    [AE α=25 LR=0.005] Epoch 10 | TrainLoss=0.7500 | ValLoss=1.0268
    [AE α=25 LR=0.005] Epoch 11 | TrainLoss=0.7308 | ValLoss=0.7547
    [AE α=25 LR=0.005] Epoch 12 | TrainLoss=0.6807 | ValLoss=0.9445
    [AE α=25 LR=0.005] Epoch 13 | TrainLoss=0.6685 | ValLoss=1.1245
    [AE α=25 LR=0.005] Epoch 14 | TrainLoss=0.6333 | ValLoss=3.0332
    [AE α=25 LR=0.005] Epoch 15 | TrainLoss=0.6139 | ValLoss=1.9962
    [AE α=25 LR=0.005] Epoch 16 | TrainLoss=0.6169 | ValLoss=0.8860
    [AE α=25 LR=0.005] Epoch 17 | TrainLoss=0.5770 | ValLoss=1.4197
    [AE α=25 LR=0.005] Epoch 18 | TrainLoss=0.5636 | ValLoss=1.0420
    [AE α=25 LR=0.005] Epoch 19 | TrainLoss=0.5421 | ValLoss=1.0050
    [AE α=25 LR=0.005] Epoch 20 | TrainLoss=0.5470 | ValLoss=2.3434
    [AE α=25 LR=0.005] Epoch 21 | TrainLoss=0.5268 | ValLoss=1.3456
    [AE α=25 LR=0.005] Epoch 22 | TrainLoss=0.5167 | ValLoss=0.6582
    [AE α=25 LR=0.005] Epoch 23 | TrainLoss=0.5143 | ValLoss=1.5924
    [AE α=25 LR=0.005] Epoch 24 | TrainLoss=0.5000 | ValLoss=2.0526
    [AE α=25 LR=0.005] Epoch 25 | TrainLoss=0.4920 | ValLoss=1.3892
    [AE α=25 LR=0.005] Epoch 26 | TrainLoss=0.4789 | ValLoss=2.6358
    [AE α=25 LR=0.005] Epoch 27 | TrainLoss=0.4825 | ValLoss=1.5979
    [AE α=25 LR=0.005] Epoch 28 | TrainLoss=0.4722 | ValLoss=1.8496
    [AE α=25 LR=0.005] Epoch 29 | TrainLoss=0.4378 | ValLoss=1.9636
    [AE α=25 LR=0.005] Epoch 30 | TrainLoss=0.4635 | ValLoss=1.6554
    [AE α=25 LR=0.005] Epoch 31 | TrainLoss=0.4536 | ValLoss=2.0420
    [AE α=25 LR=0.005] Epoch 32 | TrainLoss=0.4369 | ValLoss=0.8836
    [AE α=25 LR=0.005] Epoch 33 | TrainLoss=0.4426 | ValLoss=1.2871
    [AE α=25 LR=0.005] Epoch 34 | TrainLoss=0.4182 | ValLoss=0.9019
    [AE α=25 LR=0.005] Epoch 35 | TrainLoss=0.4229 | ValLoss=2.3765
    [AE α=25 LR=0.005] Epoch 36 | TrainLoss=0.4026 | ValLoss=5.0961
    [AE α=25 LR=0.005] Epoch 37 | TrainLoss=0.4116 | ValLoss=1.6960
    Early stopping triggered.
    
    =====================================
    Training AE for α=25, LR=0.01
    =====================================
    [AE α=25 LR=0.01] Epoch 1 | TrainLoss=2.4019 | ValLoss=1.8147
    [AE α=25 LR=0.01] Epoch 2 | TrainLoss=1.6318 | ValLoss=2.0913
    [AE α=25 LR=0.01] Epoch 3 | TrainLoss=1.3601 | ValLoss=1.6163
    [AE α=25 LR=0.01] Epoch 4 | TrainLoss=1.1323 | ValLoss=2.0195
    [AE α=25 LR=0.01] Epoch 5 | TrainLoss=1.0342 | ValLoss=1.5273
    [AE α=25 LR=0.01] Epoch 6 | TrainLoss=0.9854 | ValLoss=1.4520
    [AE α=25 LR=0.01] Epoch 7 | TrainLoss=0.9242 | ValLoss=1.2350
    [AE α=25 LR=0.01] Epoch 8 | TrainLoss=0.8862 | ValLoss=1.3140
    [AE α=25 LR=0.01] Epoch 9 | TrainLoss=0.8702 | ValLoss=1.5413
    [AE α=25 LR=0.01] Epoch 10 | TrainLoss=0.8061 | ValLoss=0.9194
    [AE α=25 LR=0.01] Epoch 11 | TrainLoss=0.7741 | ValLoss=1.2432
    [AE α=25 LR=0.01] Epoch 12 | TrainLoss=0.7706 | ValLoss=0.9164
    [AE α=25 LR=0.01] Epoch 13 | TrainLoss=0.7323 | ValLoss=1.8237
    [AE α=25 LR=0.01] Epoch 14 | TrainLoss=0.6796 | ValLoss=1.0491
    [AE α=25 LR=0.01] Epoch 15 | TrainLoss=0.6805 | ValLoss=1.0690
    [AE α=25 LR=0.01] Epoch 16 | TrainLoss=0.6577 | ValLoss=1.3318
    [AE α=25 LR=0.01] Epoch 17 | TrainLoss=0.6297 | ValLoss=0.7898
    [AE α=25 LR=0.01] Epoch 18 | TrainLoss=0.6240 | ValLoss=0.9802
    [AE α=25 LR=0.01] Epoch 19 | TrainLoss=0.6416 | ValLoss=1.6081
    [AE α=25 LR=0.01] Epoch 20 | TrainLoss=0.6096 | ValLoss=2.2446
    [AE α=25 LR=0.01] Epoch 21 | TrainLoss=0.6119 | ValLoss=3.2368
    [AE α=25 LR=0.01] Epoch 22 | TrainLoss=0.5997 | ValLoss=1.2770
    [AE α=25 LR=0.01] Epoch 23 | TrainLoss=0.5512 | ValLoss=1.4491
    [AE α=25 LR=0.01] Epoch 24 | TrainLoss=0.5734 | ValLoss=0.8889
    [AE α=25 LR=0.01] Epoch 25 | TrainLoss=0.5712 | ValLoss=1.0945
    [AE α=25 LR=0.01] Epoch 26 | TrainLoss=0.5458 | ValLoss=2.0879
    [AE α=25 LR=0.01] Epoch 27 | TrainLoss=0.5665 | ValLoss=2.9664
    [AE α=25 LR=0.01] Epoch 28 | TrainLoss=0.5430 | ValLoss=2.1198
    [AE α=25 LR=0.01] Epoch 29 | TrainLoss=0.5334 | ValLoss=1.5941
    [AE α=25 LR=0.01] Epoch 30 | TrainLoss=0.5146 | ValLoss=1.9449
    [AE α=25 LR=0.01] Epoch 31 | TrainLoss=0.5058 | ValLoss=1.0944
    [AE α=25 LR=0.01] Epoch 32 | TrainLoss=0.5137 | ValLoss=1.1594
    Early stopping triggered.
    
    =====================================
    Training AE for α=25, LR=0.05
    =====================================
    [AE α=25 LR=0.05] Epoch 1 | TrainLoss=8.0922 | ValLoss=2.2541
    [AE α=25 LR=0.05] Epoch 2 | TrainLoss=2.1901 | ValLoss=2.0388
    [AE α=25 LR=0.05] Epoch 3 | TrainLoss=2.0236 | ValLoss=1.9555
    [AE α=25 LR=0.05] Epoch 4 | TrainLoss=1.8962 | ValLoss=1.9171
    [AE α=25 LR=0.05] Epoch 5 | TrainLoss=1.7888 | ValLoss=1.7301
    [AE α=25 LR=0.05] Epoch 6 | TrainLoss=1.7288 | ValLoss=1.8077
    [AE α=25 LR=0.05] Epoch 7 | TrainLoss=1.6539 | ValLoss=2.2467
    [AE α=25 LR=0.05] Epoch 8 | TrainLoss=1.5319 | ValLoss=2.5051
    [AE α=25 LR=0.05] Epoch 9 | TrainLoss=1.6436 | ValLoss=3.2449
    [AE α=25 LR=0.05] Epoch 10 | TrainLoss=1.4946 | ValLoss=1.4330
    [AE α=25 LR=0.05] Epoch 11 | TrainLoss=1.4170 | ValLoss=2.6366
    [AE α=25 LR=0.05] Epoch 12 | TrainLoss=1.4083 | ValLoss=1.4873
    [AE α=25 LR=0.05] Epoch 13 | TrainLoss=1.3424 | ValLoss=1.7986
    [AE α=25 LR=0.05] Epoch 14 | TrainLoss=1.2774 | ValLoss=2.3901
    [AE α=25 LR=0.05] Epoch 15 | TrainLoss=1.3147 | ValLoss=1.2949
    [AE α=25 LR=0.05] Epoch 16 | TrainLoss=1.2687 | ValLoss=1.2733
    [AE α=25 LR=0.05] Epoch 17 | TrainLoss=1.3527 | ValLoss=1.6573
    [AE α=25 LR=0.05] Epoch 18 | TrainLoss=1.2697 | ValLoss=1.3164
    [AE α=25 LR=0.05] Epoch 19 | TrainLoss=1.2330 | ValLoss=1.8742
    [AE α=25 LR=0.05] Epoch 20 | TrainLoss=1.2491 | ValLoss=1.9703
    [AE α=25 LR=0.05] Epoch 21 | TrainLoss=1.2201 | ValLoss=2.6716
    [AE α=25 LR=0.05] Epoch 22 | TrainLoss=1.2596 | ValLoss=1.2955
    [AE α=25 LR=0.05] Epoch 23 | TrainLoss=1.2002 | ValLoss=1.5603
    [AE α=25 LR=0.05] Epoch 24 | TrainLoss=1.1605 | ValLoss=1.4313
    [AE α=25 LR=0.05] Epoch 25 | TrainLoss=1.1731 | ValLoss=1.5718
    [AE α=25 LR=0.05] Epoch 26 | TrainLoss=1.1681 | ValLoss=1.8620
    [AE α=25 LR=0.05] Epoch 27 | TrainLoss=1.2138 | ValLoss=1.2266
    [AE α=25 LR=0.05] Epoch 28 | TrainLoss=1.1885 | ValLoss=1.2112
    [AE α=25 LR=0.05] Epoch 29 | TrainLoss=1.1736 | ValLoss=1.7044
    [AE α=25 LR=0.05] Epoch 30 | TrainLoss=1.1546 | ValLoss=1.6998
    [AE α=25 LR=0.05] Epoch 31 | TrainLoss=1.1986 | ValLoss=1.4311
    [AE α=25 LR=0.05] Epoch 32 | TrainLoss=1.1494 | ValLoss=1.2231
    [AE α=25 LR=0.05] Epoch 33 | TrainLoss=1.1468 | ValLoss=1.1856
    [AE α=25 LR=0.05] Epoch 34 | TrainLoss=1.1591 | ValLoss=2.1781
    [AE α=25 LR=0.05] Epoch 35 | TrainLoss=1.1397 | ValLoss=1.8044
    [AE α=25 LR=0.05] Epoch 36 | TrainLoss=1.1255 | ValLoss=1.0374
    [AE α=25 LR=0.05] Epoch 37 | TrainLoss=1.1267 | ValLoss=1.2995
    [AE α=25 LR=0.05] Epoch 38 | TrainLoss=1.1415 | ValLoss=1.1198
    [AE α=25 LR=0.05] Epoch 39 | TrainLoss=1.1896 | ValLoss=1.6122
    [AE α=25 LR=0.05] Epoch 40 | TrainLoss=1.1612 | ValLoss=1.7749
    [AE α=25 LR=0.05] Epoch 41 | TrainLoss=1.1972 | ValLoss=2.1877
    [AE α=25 LR=0.05] Epoch 42 | TrainLoss=1.1488 | ValLoss=1.0971
    [AE α=25 LR=0.05] Epoch 43 | TrainLoss=1.0827 | ValLoss=1.3959
    [AE α=25 LR=0.05] Epoch 44 | TrainLoss=1.0880 | ValLoss=1.2163
    [AE α=25 LR=0.05] Epoch 45 | TrainLoss=1.0943 | ValLoss=1.2417
    [AE α=25 LR=0.05] Epoch 46 | TrainLoss=1.1601 | ValLoss=1.7614
    [AE α=25 LR=0.05] Epoch 47 | TrainLoss=1.0698 | ValLoss=1.3591
    [AE α=25 LR=0.05] Epoch 48 | TrainLoss=1.0602 | ValLoss=1.5399
    [AE α=25 LR=0.05] Epoch 49 | TrainLoss=1.1127 | ValLoss=1.2431
    [AE α=25 LR=0.05] Epoch 50 | TrainLoss=1.1209 | ValLoss=1.2533
    [AE α=25 LR=0.05] Epoch 51 | TrainLoss=1.0666 | ValLoss=1.2795
    Early stopping triggered.
    
    =====================================
    Training AE for α=25, LR=0.1
    =====================================
    [AE α=25 LR=0.1] Epoch 1 | TrainLoss=18.5690 | ValLoss=2.5322
    [AE α=25 LR=0.1] Epoch 2 | TrainLoss=2.6376 | ValLoss=2.3638
    [AE α=25 LR=0.1] Epoch 3 | TrainLoss=2.6101 | ValLoss=2.5436
    [AE α=25 LR=0.1] Epoch 4 | TrainLoss=2.4550 | ValLoss=2.1649
    [AE α=25 LR=0.1] Epoch 5 | TrainLoss=2.5273 | ValLoss=3.9469
    [AE α=25 LR=0.1] Epoch 6 | TrainLoss=3.1161 | ValLoss=3.0134
    [AE α=25 LR=0.1] Epoch 7 | TrainLoss=3.0613 | ValLoss=3.0076
    [AE α=25 LR=0.1] Epoch 8 | TrainLoss=3.0620 | ValLoss=3.0290
    [AE α=25 LR=0.1] Epoch 9 | TrainLoss=3.0604 | ValLoss=2.9973
    [AE α=25 LR=0.1] Epoch 10 | TrainLoss=3.0617 | ValLoss=3.0176
    [AE α=25 LR=0.1] Epoch 11 | TrainLoss=3.0604 | ValLoss=2.9653
    [AE α=25 LR=0.1] Epoch 12 | TrainLoss=3.0643 | ValLoss=2.9524
    [AE α=25 LR=0.1] Epoch 13 | TrainLoss=3.0635 | ValLoss=3.0577
    [AE α=25 LR=0.1] Epoch 14 | TrainLoss=3.0613 | ValLoss=2.9815
    [AE α=25 LR=0.1] Epoch 15 | TrainLoss=3.0624 | ValLoss=2.9860
    [AE α=25 LR=0.1] Epoch 16 | TrainLoss=3.0637 | ValLoss=3.0011
    [AE α=25 LR=0.1] Epoch 17 | TrainLoss=3.0627 | ValLoss=2.9953
    [AE α=25 LR=0.1] Epoch 18 | TrainLoss=3.0633 | ValLoss=3.0080
    [AE α=25 LR=0.1] Epoch 19 | TrainLoss=3.0615 | ValLoss=3.0079
    Early stopping triggered.
    
    =====================================
    Training AE for α=30, LR=0.0001
    =====================================
    [AE α=30 LR=0.0001] Epoch 1 | TrainLoss=2.3575 | ValLoss=1.5270
    [AE α=30 LR=0.0001] Epoch 2 | TrainLoss=1.4340 | ValLoss=1.7402
    [AE α=30 LR=0.0001] Epoch 3 | TrainLoss=1.1422 | ValLoss=1.6877
    [AE α=30 LR=0.0001] Epoch 4 | TrainLoss=1.0005 | ValLoss=2.0158
    [AE α=30 LR=0.0001] Epoch 5 | TrainLoss=0.9233 | ValLoss=1.7618
    [AE α=30 LR=0.0001] Epoch 6 | TrainLoss=0.8873 | ValLoss=1.6634
    [AE α=30 LR=0.0001] Epoch 7 | TrainLoss=0.8411 | ValLoss=1.7419
    [AE α=30 LR=0.0001] Epoch 8 | TrainLoss=0.8106 | ValLoss=1.6564
    [AE α=30 LR=0.0001] Epoch 9 | TrainLoss=0.7817 | ValLoss=1.8760
    [AE α=30 LR=0.0001] Epoch 10 | TrainLoss=0.7583 | ValLoss=1.6109
    [AE α=30 LR=0.0001] Epoch 11 | TrainLoss=0.7225 | ValLoss=2.1225
    [AE α=30 LR=0.0001] Epoch 12 | TrainLoss=0.7140 | ValLoss=1.6072
    [AE α=30 LR=0.0001] Epoch 13 | TrainLoss=0.6802 | ValLoss=1.8998
    [AE α=30 LR=0.0001] Epoch 14 | TrainLoss=0.6772 | ValLoss=2.0132
    [AE α=30 LR=0.0001] Epoch 15 | TrainLoss=0.6431 | ValLoss=2.0927
    [AE α=30 LR=0.0001] Epoch 16 | TrainLoss=0.6146 | ValLoss=2.2833
    Early stopping triggered.
    
    =====================================
    Training AE for α=30, LR=0.0002
    =====================================
    [AE α=30 LR=0.0002] Epoch 1 | TrainLoss=1.9533 | ValLoss=1.4385
    [AE α=30 LR=0.0002] Epoch 2 | TrainLoss=1.2044 | ValLoss=1.7931
    [AE α=30 LR=0.0002] Epoch 3 | TrainLoss=1.0020 | ValLoss=1.7098
    [AE α=30 LR=0.0002] Epoch 4 | TrainLoss=0.9214 | ValLoss=2.0732
    [AE α=30 LR=0.0002] Epoch 5 | TrainLoss=0.8602 | ValLoss=1.6412
    [AE α=30 LR=0.0002] Epoch 6 | TrainLoss=0.8079 | ValLoss=2.0423
    [AE α=30 LR=0.0002] Epoch 7 | TrainLoss=0.7556 | ValLoss=1.5950
    [AE α=30 LR=0.0002] Epoch 8 | TrainLoss=0.7405 | ValLoss=2.0157
    [AE α=30 LR=0.0002] Epoch 9 | TrainLoss=0.7109 | ValLoss=2.6712
    [AE α=30 LR=0.0002] Epoch 10 | TrainLoss=0.7012 | ValLoss=2.1484
    [AE α=30 LR=0.0002] Epoch 11 | TrainLoss=0.6637 | ValLoss=1.6766
    [AE α=30 LR=0.0002] Epoch 12 | TrainLoss=0.6397 | ValLoss=2.8756
    [AE α=30 LR=0.0002] Epoch 13 | TrainLoss=0.6285 | ValLoss=2.2796
    [AE α=30 LR=0.0002] Epoch 14 | TrainLoss=0.5962 | ValLoss=1.7594
    [AE α=30 LR=0.0002] Epoch 15 | TrainLoss=0.5909 | ValLoss=2.4406
    [AE α=30 LR=0.0002] Epoch 16 | TrainLoss=0.5685 | ValLoss=2.1131
    Early stopping triggered.
    
    =====================================
    Training AE for α=30, LR=0.0005
    =====================================
    [AE α=30 LR=0.0005] Epoch 1 | TrainLoss=1.8453 | ValLoss=1.8288
    [AE α=30 LR=0.0005] Epoch 2 | TrainLoss=1.1434 | ValLoss=1.9402
    [AE α=30 LR=0.0005] Epoch 3 | TrainLoss=0.9709 | ValLoss=3.1722
    [AE α=30 LR=0.0005] Epoch 4 | TrainLoss=0.8740 | ValLoss=1.4426
    [AE α=30 LR=0.0005] Epoch 5 | TrainLoss=0.8178 | ValLoss=1.7284
    [AE α=30 LR=0.0005] Epoch 6 | TrainLoss=0.7783 | ValLoss=2.3356
    [AE α=30 LR=0.0005] Epoch 7 | TrainLoss=0.7164 | ValLoss=1.8110
    [AE α=30 LR=0.0005] Epoch 8 | TrainLoss=0.6818 | ValLoss=3.2364
    [AE α=30 LR=0.0005] Epoch 9 | TrainLoss=0.6578 | ValLoss=1.2973
    [AE α=30 LR=0.0005] Epoch 10 | TrainLoss=0.6221 | ValLoss=2.0064
    [AE α=30 LR=0.0005] Epoch 11 | TrainLoss=0.6147 | ValLoss=1.4117
    [AE α=30 LR=0.0005] Epoch 12 | TrainLoss=0.5826 | ValLoss=1.9386
    [AE α=30 LR=0.0005] Epoch 13 | TrainLoss=0.5549 | ValLoss=1.5280
    [AE α=30 LR=0.0005] Epoch 14 | TrainLoss=0.5397 | ValLoss=2.3157
    [AE α=30 LR=0.0005] Epoch 15 | TrainLoss=0.5214 | ValLoss=3.3148
    [AE α=30 LR=0.0005] Epoch 16 | TrainLoss=0.5093 | ValLoss=2.1797
    [AE α=30 LR=0.0005] Epoch 17 | TrainLoss=0.4931 | ValLoss=1.2277
    [AE α=30 LR=0.0005] Epoch 18 | TrainLoss=0.4893 | ValLoss=1.9352
    [AE α=30 LR=0.0005] Epoch 19 | TrainLoss=0.4670 | ValLoss=1.7712
    [AE α=30 LR=0.0005] Epoch 20 | TrainLoss=0.4676 | ValLoss=2.6284
    [AE α=30 LR=0.0005] Epoch 21 | TrainLoss=0.4539 | ValLoss=1.6709
    [AE α=30 LR=0.0005] Epoch 22 | TrainLoss=0.4362 | ValLoss=2.7092
    [AE α=30 LR=0.0005] Epoch 23 | TrainLoss=0.4286 | ValLoss=1.8202
    [AE α=30 LR=0.0005] Epoch 24 | TrainLoss=0.4293 | ValLoss=2.5948
    [AE α=30 LR=0.0005] Epoch 25 | TrainLoss=0.4333 | ValLoss=1.9940
    [AE α=30 LR=0.0005] Epoch 26 | TrainLoss=0.4022 | ValLoss=3.0901
    [AE α=30 LR=0.0005] Epoch 27 | TrainLoss=0.3976 | ValLoss=1.6979
    [AE α=30 LR=0.0005] Epoch 28 | TrainLoss=0.4105 | ValLoss=1.9743
    [AE α=30 LR=0.0005] Epoch 29 | TrainLoss=0.3858 | ValLoss=2.1296
    [AE α=30 LR=0.0005] Epoch 30 | TrainLoss=0.3835 | ValLoss=2.6432
    [AE α=30 LR=0.0005] Epoch 31 | TrainLoss=0.3775 | ValLoss=1.5298
    [AE α=30 LR=0.0005] Epoch 32 | TrainLoss=0.3662 | ValLoss=2.5649
    Early stopping triggered.
    
    =====================================
    Training AE for α=30, LR=0.001
    =====================================
    [AE α=30 LR=0.001] Epoch 1 | TrainLoss=1.7047 | ValLoss=1.8266
    [AE α=30 LR=0.001] Epoch 2 | TrainLoss=1.1232 | ValLoss=2.2960
    [AE α=30 LR=0.001] Epoch 3 | TrainLoss=0.9775 | ValLoss=1.3578
    [AE α=30 LR=0.001] Epoch 4 | TrainLoss=0.8905 | ValLoss=1.2676
    [AE α=30 LR=0.001] Epoch 5 | TrainLoss=0.8145 | ValLoss=1.4990
    [AE α=30 LR=0.001] Epoch 6 | TrainLoss=0.7891 | ValLoss=1.6405
    [AE α=30 LR=0.001] Epoch 7 | TrainLoss=0.7337 | ValLoss=1.7862
    [AE α=30 LR=0.001] Epoch 8 | TrainLoss=0.6967 | ValLoss=1.2722
    [AE α=30 LR=0.001] Epoch 9 | TrainLoss=0.6721 | ValLoss=0.7798
    [AE α=30 LR=0.001] Epoch 10 | TrainLoss=0.6441 | ValLoss=1.5860
    [AE α=30 LR=0.001] Epoch 11 | TrainLoss=0.6168 | ValLoss=1.0368
    [AE α=30 LR=0.001] Epoch 12 | TrainLoss=0.5865 | ValLoss=0.9475
    [AE α=30 LR=0.001] Epoch 13 | TrainLoss=0.5772 | ValLoss=0.8401
    [AE α=30 LR=0.001] Epoch 14 | TrainLoss=0.5363 | ValLoss=1.8825
    [AE α=30 LR=0.001] Epoch 15 | TrainLoss=0.5276 | ValLoss=1.2469
    [AE α=30 LR=0.001] Epoch 16 | TrainLoss=0.4993 | ValLoss=2.2560
    [AE α=30 LR=0.001] Epoch 17 | TrainLoss=0.4948 | ValLoss=1.8110
    [AE α=30 LR=0.001] Epoch 18 | TrainLoss=0.4839 | ValLoss=1.1946
    [AE α=30 LR=0.001] Epoch 19 | TrainLoss=0.4713 | ValLoss=1.7031
    [AE α=30 LR=0.001] Epoch 20 | TrainLoss=0.4615 | ValLoss=1.1113
    [AE α=30 LR=0.001] Epoch 21 | TrainLoss=0.4389 | ValLoss=0.7436
    [AE α=30 LR=0.001] Epoch 22 | TrainLoss=0.4314 | ValLoss=1.9203
    [AE α=30 LR=0.001] Epoch 23 | TrainLoss=0.4246 | ValLoss=1.1434
    [AE α=30 LR=0.001] Epoch 24 | TrainLoss=0.4238 | ValLoss=1.3121
    [AE α=30 LR=0.001] Epoch 25 | TrainLoss=0.4080 | ValLoss=1.9195
    [AE α=30 LR=0.001] Epoch 26 | TrainLoss=0.4050 | ValLoss=1.2443
    [AE α=30 LR=0.001] Epoch 27 | TrainLoss=0.3849 | ValLoss=1.6033
    [AE α=30 LR=0.001] Epoch 28 | TrainLoss=0.3833 | ValLoss=2.6476
    [AE α=30 LR=0.001] Epoch 29 | TrainLoss=0.3820 | ValLoss=1.9618
    [AE α=30 LR=0.001] Epoch 30 | TrainLoss=0.3740 | ValLoss=0.8573
    [AE α=30 LR=0.001] Epoch 31 | TrainLoss=0.3702 | ValLoss=1.5636
    [AE α=30 LR=0.001] Epoch 32 | TrainLoss=0.3656 | ValLoss=1.5952
    [AE α=30 LR=0.001] Epoch 33 | TrainLoss=0.3476 | ValLoss=1.2956
    [AE α=30 LR=0.001] Epoch 34 | TrainLoss=0.3396 | ValLoss=1.8822
    [AE α=30 LR=0.001] Epoch 35 | TrainLoss=0.3420 | ValLoss=2.4644
    [AE α=30 LR=0.001] Epoch 36 | TrainLoss=0.3364 | ValLoss=2.0688
    Early stopping triggered.
    
    =====================================
    Training AE for α=30, LR=0.002
    =====================================
    [AE α=30 LR=0.002] Epoch 1 | TrainLoss=1.7935 | ValLoss=2.5236
    [AE α=30 LR=0.002] Epoch 2 | TrainLoss=1.2056 | ValLoss=1.1403
    [AE α=30 LR=0.002] Epoch 3 | TrainLoss=1.0067 | ValLoss=1.4628
    [AE α=30 LR=0.002] Epoch 4 | TrainLoss=0.9341 | ValLoss=1.1937
    [AE α=30 LR=0.002] Epoch 5 | TrainLoss=0.8794 | ValLoss=2.8214
    [AE α=30 LR=0.002] Epoch 6 | TrainLoss=0.8388 | ValLoss=1.8232
    [AE α=30 LR=0.002] Epoch 7 | TrainLoss=0.8054 | ValLoss=2.2875
    [AE α=30 LR=0.002] Epoch 8 | TrainLoss=0.7465 | ValLoss=2.6678
    [AE α=30 LR=0.002] Epoch 9 | TrainLoss=0.7155 | ValLoss=2.3151
    [AE α=30 LR=0.002] Epoch 10 | TrainLoss=0.7077 | ValLoss=0.9400
    [AE α=30 LR=0.002] Epoch 11 | TrainLoss=0.6623 | ValLoss=1.4931
    [AE α=30 LR=0.002] Epoch 12 | TrainLoss=0.6235 | ValLoss=1.6300
    [AE α=30 LR=0.002] Epoch 13 | TrainLoss=0.6093 | ValLoss=0.7555
    [AE α=30 LR=0.002] Epoch 14 | TrainLoss=0.5627 | ValLoss=2.7685
    [AE α=30 LR=0.002] Epoch 15 | TrainLoss=0.5647 | ValLoss=2.4936
    [AE α=30 LR=0.002] Epoch 16 | TrainLoss=0.5505 | ValLoss=1.3015
    [AE α=30 LR=0.002] Epoch 17 | TrainLoss=0.5261 | ValLoss=1.4110
    [AE α=30 LR=0.002] Epoch 18 | TrainLoss=0.5046 | ValLoss=1.3924
    [AE α=30 LR=0.002] Epoch 19 | TrainLoss=0.4863 | ValLoss=1.7224
    [AE α=30 LR=0.002] Epoch 20 | TrainLoss=0.4805 | ValLoss=0.8862
    [AE α=30 LR=0.002] Epoch 21 | TrainLoss=0.4849 | ValLoss=0.6530
    [AE α=30 LR=0.002] Epoch 22 | TrainLoss=0.4532 | ValLoss=2.3134
    [AE α=30 LR=0.002] Epoch 23 | TrainLoss=0.4611 | ValLoss=1.4518
    [AE α=30 LR=0.002] Epoch 24 | TrainLoss=0.4416 | ValLoss=1.5817
    [AE α=30 LR=0.002] Epoch 25 | TrainLoss=0.4386 | ValLoss=0.8208
    [AE α=30 LR=0.002] Epoch 26 | TrainLoss=0.4203 | ValLoss=1.4146
    [AE α=30 LR=0.002] Epoch 27 | TrainLoss=0.4102 | ValLoss=1.1494
    [AE α=30 LR=0.002] Epoch 28 | TrainLoss=0.4093 | ValLoss=1.7168
    [AE α=30 LR=0.002] Epoch 29 | TrainLoss=0.4031 | ValLoss=1.7090
    [AE α=30 LR=0.002] Epoch 30 | TrainLoss=0.3879 | ValLoss=2.1735
    [AE α=30 LR=0.002] Epoch 31 | TrainLoss=0.3915 | ValLoss=2.8475
    [AE α=30 LR=0.002] Epoch 32 | TrainLoss=0.3771 | ValLoss=1.3299
    [AE α=30 LR=0.002] Epoch 33 | TrainLoss=0.3816 | ValLoss=1.8544
    [AE α=30 LR=0.002] Epoch 34 | TrainLoss=0.3638 | ValLoss=2.7911
    [AE α=30 LR=0.002] Epoch 35 | TrainLoss=0.3662 | ValLoss=1.9816
    [AE α=30 LR=0.002] Epoch 36 | TrainLoss=0.3701 | ValLoss=1.5349
    Early stopping triggered.
    
    =====================================
    Training AE for α=30, LR=0.005
    =====================================
    [AE α=30 LR=0.005] Epoch 1 | TrainLoss=2.1056 | ValLoss=1.8741
    [AE α=30 LR=0.005] Epoch 2 | TrainLoss=1.4141 | ValLoss=1.6225
    [AE α=30 LR=0.005] Epoch 3 | TrainLoss=1.1738 | ValLoss=1.5052
    [AE α=30 LR=0.005] Epoch 4 | TrainLoss=1.0502 | ValLoss=1.4114
    [AE α=30 LR=0.005] Epoch 5 | TrainLoss=0.9647 | ValLoss=2.3405
    [AE α=30 LR=0.005] Epoch 6 | TrainLoss=0.9318 | ValLoss=2.5594
    [AE α=30 LR=0.005] Epoch 7 | TrainLoss=0.8609 | ValLoss=2.0665
    [AE α=30 LR=0.005] Epoch 8 | TrainLoss=0.8161 | ValLoss=2.6482
    [AE α=30 LR=0.005] Epoch 9 | TrainLoss=0.7723 | ValLoss=1.5291
    [AE α=30 LR=0.005] Epoch 10 | TrainLoss=0.7372 | ValLoss=2.0488
    [AE α=30 LR=0.005] Epoch 11 | TrainLoss=0.7260 | ValLoss=1.1971
    [AE α=30 LR=0.005] Epoch 12 | TrainLoss=0.6933 | ValLoss=2.7410
    [AE α=30 LR=0.005] Epoch 13 | TrainLoss=0.6559 | ValLoss=1.8033
    [AE α=30 LR=0.005] Epoch 14 | TrainLoss=0.6345 | ValLoss=1.2247
    [AE α=30 LR=0.005] Epoch 15 | TrainLoss=0.6290 | ValLoss=0.7958
    [AE α=30 LR=0.005] Epoch 16 | TrainLoss=0.5883 | ValLoss=0.9852
    [AE α=30 LR=0.005] Epoch 17 | TrainLoss=0.5674 | ValLoss=1.4073
    [AE α=30 LR=0.005] Epoch 18 | TrainLoss=0.5757 | ValLoss=0.6365
    [AE α=30 LR=0.005] Epoch 19 | TrainLoss=0.5518 | ValLoss=1.5249
    [AE α=30 LR=0.005] Epoch 20 | TrainLoss=0.5507 | ValLoss=1.5955
    [AE α=30 LR=0.005] Epoch 21 | TrainLoss=0.5338 | ValLoss=0.8077
    [AE α=30 LR=0.005] Epoch 22 | TrainLoss=0.5050 | ValLoss=1.9144
    [AE α=30 LR=0.005] Epoch 23 | TrainLoss=0.5074 | ValLoss=1.1910
    [AE α=30 LR=0.005] Epoch 24 | TrainLoss=0.4998 | ValLoss=2.0207
    [AE α=30 LR=0.005] Epoch 25 | TrainLoss=0.4874 | ValLoss=2.5895
    [AE α=30 LR=0.005] Epoch 26 | TrainLoss=0.4805 | ValLoss=1.1557
    [AE α=30 LR=0.005] Epoch 27 | TrainLoss=0.4780 | ValLoss=0.8644
    [AE α=30 LR=0.005] Epoch 28 | TrainLoss=0.4654 | ValLoss=1.6413
    [AE α=30 LR=0.005] Epoch 29 | TrainLoss=0.4625 | ValLoss=3.0262
    [AE α=30 LR=0.005] Epoch 30 | TrainLoss=0.4498 | ValLoss=1.2333
    [AE α=30 LR=0.005] Epoch 31 | TrainLoss=0.4577 | ValLoss=1.1528
    [AE α=30 LR=0.005] Epoch 32 | TrainLoss=0.4421 | ValLoss=2.7713
    [AE α=30 LR=0.005] Epoch 33 | TrainLoss=0.4301 | ValLoss=0.6463
    Early stopping triggered.
    
    =====================================
    Training AE for α=30, LR=0.01
    =====================================
    [AE α=30 LR=0.01] Epoch 1 | TrainLoss=2.3990 | ValLoss=2.2037
    [AE α=30 LR=0.01] Epoch 2 | TrainLoss=1.7127 | ValLoss=2.0972
    [AE α=30 LR=0.01] Epoch 3 | TrainLoss=1.4045 | ValLoss=1.6648
    [AE α=30 LR=0.01] Epoch 4 | TrainLoss=1.2042 | ValLoss=1.1182
    [AE α=30 LR=0.01] Epoch 5 | TrainLoss=1.0791 | ValLoss=0.9843
    [AE α=30 LR=0.01] Epoch 6 | TrainLoss=1.0028 | ValLoss=2.1648
    [AE α=30 LR=0.01] Epoch 7 | TrainLoss=0.9705 | ValLoss=1.1819
    [AE α=30 LR=0.01] Epoch 8 | TrainLoss=0.9168 | ValLoss=1.5412
    [AE α=30 LR=0.01] Epoch 9 | TrainLoss=0.8891 | ValLoss=1.3591
    [AE α=30 LR=0.01] Epoch 10 | TrainLoss=0.8181 | ValLoss=1.6757
    [AE α=30 LR=0.01] Epoch 11 | TrainLoss=0.8261 | ValLoss=1.6035
    [AE α=30 LR=0.01] Epoch 12 | TrainLoss=0.7808 | ValLoss=1.0452
    [AE α=30 LR=0.01] Epoch 13 | TrainLoss=0.7674 | ValLoss=1.0745
    [AE α=30 LR=0.01] Epoch 14 | TrainLoss=0.7633 | ValLoss=2.3962
    [AE α=30 LR=0.01] Epoch 15 | TrainLoss=0.7283 | ValLoss=2.0290
    [AE α=30 LR=0.01] Epoch 16 | TrainLoss=0.6999 | ValLoss=1.4905
    [AE α=30 LR=0.01] Epoch 17 | TrainLoss=0.7017 | ValLoss=0.7703
    [AE α=30 LR=0.01] Epoch 18 | TrainLoss=0.6906 | ValLoss=0.7489
    [AE α=30 LR=0.01] Epoch 19 | TrainLoss=0.6656 | ValLoss=1.3090
    [AE α=30 LR=0.01] Epoch 20 | TrainLoss=0.6632 | ValLoss=0.7799
    [AE α=30 LR=0.01] Epoch 21 | TrainLoss=0.6432 | ValLoss=1.0958
    [AE α=30 LR=0.01] Epoch 22 | TrainLoss=0.6200 | ValLoss=0.7300
    [AE α=30 LR=0.01] Epoch 23 | TrainLoss=0.6221 | ValLoss=4.1756
    [AE α=30 LR=0.01] Epoch 24 | TrainLoss=0.5900 | ValLoss=3.0269
    [AE α=30 LR=0.01] Epoch 25 | TrainLoss=0.6060 | ValLoss=0.9181
    [AE α=30 LR=0.01] Epoch 26 | TrainLoss=0.5970 | ValLoss=0.9863
    [AE α=30 LR=0.01] Epoch 27 | TrainLoss=0.5818 | ValLoss=1.1285
    [AE α=30 LR=0.01] Epoch 28 | TrainLoss=0.5460 | ValLoss=3.1085
    [AE α=30 LR=0.01] Epoch 29 | TrainLoss=0.5671 | ValLoss=0.7150
    [AE α=30 LR=0.01] Epoch 30 | TrainLoss=0.5477 | ValLoss=2.0479
    [AE α=30 LR=0.01] Epoch 31 | TrainLoss=0.5360 | ValLoss=2.6171
    [AE α=30 LR=0.01] Epoch 32 | TrainLoss=0.5424 | ValLoss=1.1603
    [AE α=30 LR=0.01] Epoch 33 | TrainLoss=0.5336 | ValLoss=1.4688
    [AE α=30 LR=0.01] Epoch 34 | TrainLoss=0.5222 | ValLoss=1.1537
    [AE α=30 LR=0.01] Epoch 35 | TrainLoss=0.5184 | ValLoss=3.0443
    [AE α=30 LR=0.01] Epoch 36 | TrainLoss=0.5119 | ValLoss=1.7298
    [AE α=30 LR=0.01] Epoch 37 | TrainLoss=0.5122 | ValLoss=1.0189
    [AE α=30 LR=0.01] Epoch 38 | TrainLoss=0.5127 | ValLoss=1.9462
    [AE α=30 LR=0.01] Epoch 39 | TrainLoss=0.4989 | ValLoss=0.6706
    [AE α=30 LR=0.01] Epoch 40 | TrainLoss=0.4826 | ValLoss=1.9597
    [AE α=30 LR=0.01] Epoch 41 | TrainLoss=0.4907 | ValLoss=1.5059
    [AE α=30 LR=0.01] Epoch 42 | TrainLoss=0.4964 | ValLoss=2.1641
    [AE α=30 LR=0.01] Epoch 43 | TrainLoss=0.4805 | ValLoss=3.1383
    [AE α=30 LR=0.01] Epoch 44 | TrainLoss=0.4893 | ValLoss=0.7570
    [AE α=30 LR=0.01] Epoch 45 | TrainLoss=0.4931 | ValLoss=1.1861
    [AE α=30 LR=0.01] Epoch 46 | TrainLoss=0.4675 | ValLoss=1.9299
    [AE α=30 LR=0.01] Epoch 47 | TrainLoss=0.4669 | ValLoss=1.4489
    [AE α=30 LR=0.01] Epoch 48 | TrainLoss=0.4526 | ValLoss=0.9341
    [AE α=30 LR=0.01] Epoch 49 | TrainLoss=0.4716 | ValLoss=0.9199
    [AE α=30 LR=0.01] Epoch 50 | TrainLoss=0.4424 | ValLoss=1.1603
    [AE α=30 LR=0.01] Epoch 51 | TrainLoss=0.4552 | ValLoss=1.4447
    [AE α=30 LR=0.01] Epoch 52 | TrainLoss=0.4509 | ValLoss=0.6655
    [AE α=30 LR=0.01] Epoch 53 | TrainLoss=0.4394 | ValLoss=1.3690
    [AE α=30 LR=0.01] Epoch 54 | TrainLoss=0.4352 | ValLoss=1.2354
    [AE α=30 LR=0.01] Epoch 55 | TrainLoss=0.4367 | ValLoss=1.3643
    [AE α=30 LR=0.01] Epoch 56 | TrainLoss=0.4222 | ValLoss=1.4232
    [AE α=30 LR=0.01] Epoch 57 | TrainLoss=0.4354 | ValLoss=1.4740
    [AE α=30 LR=0.01] Epoch 58 | TrainLoss=0.4301 | ValLoss=4.1774
    [AE α=30 LR=0.01] Epoch 59 | TrainLoss=0.4359 | ValLoss=2.0450
    [AE α=30 LR=0.01] Epoch 60 | TrainLoss=0.4289 | ValLoss=2.6888
    [AE α=30 LR=0.01] Epoch 61 | TrainLoss=0.4285 | ValLoss=2.1162
    [AE α=30 LR=0.01] Epoch 62 | TrainLoss=0.4300 | ValLoss=1.6007
    [AE α=30 LR=0.01] Epoch 63 | TrainLoss=0.4495 | ValLoss=1.0739
    [AE α=30 LR=0.01] Epoch 64 | TrainLoss=0.4239 | ValLoss=1.4952
    [AE α=30 LR=0.01] Epoch 65 | TrainLoss=0.4158 | ValLoss=4.0534
    [AE α=30 LR=0.01] Epoch 66 | TrainLoss=0.4146 | ValLoss=1.0994
    [AE α=30 LR=0.01] Epoch 67 | TrainLoss=0.4095 | ValLoss=2.0307
    Early stopping triggered.
    
    =====================================
    Training AE for α=30, LR=0.05
    =====================================
    [AE α=30 LR=0.05] Epoch 1 | TrainLoss=6.2952 | ValLoss=2.6899
    [AE α=30 LR=0.05] Epoch 2 | TrainLoss=2.7513 | ValLoss=2.4254
    [AE α=30 LR=0.05] Epoch 3 | TrainLoss=2.4210 | ValLoss=2.5209
    [AE α=30 LR=0.05] Epoch 4 | TrainLoss=2.3454 | ValLoss=2.2286
    [AE α=30 LR=0.05] Epoch 5 | TrainLoss=2.3333 | ValLoss=2.2606
    [AE α=30 LR=0.05] Epoch 6 | TrainLoss=2.2742 | ValLoss=2.3515
    [AE α=30 LR=0.05] Epoch 7 | TrainLoss=2.3241 | ValLoss=2.2034
    [AE α=30 LR=0.05] Epoch 8 | TrainLoss=2.2084 | ValLoss=2.2513
    [AE α=30 LR=0.05] Epoch 9 | TrainLoss=2.1002 | ValLoss=2.0299
    [AE α=30 LR=0.05] Epoch 10 | TrainLoss=2.1066 | ValLoss=2.0923
    [AE α=30 LR=0.05] Epoch 11 | TrainLoss=2.1126 | ValLoss=3.8340
    [AE α=30 LR=0.05] Epoch 12 | TrainLoss=2.4234 | ValLoss=2.2746
    [AE α=30 LR=0.05] Epoch 13 | TrainLoss=2.3000 | ValLoss=2.2109
    [AE α=30 LR=0.05] Epoch 14 | TrainLoss=2.2750 | ValLoss=2.1564
    [AE α=30 LR=0.05] Epoch 15 | TrainLoss=2.2023 | ValLoss=2.6047
    [AE α=30 LR=0.05] Epoch 16 | TrainLoss=2.1584 | ValLoss=2.0458
    [AE α=30 LR=0.05] Epoch 17 | TrainLoss=2.1383 | ValLoss=2.0417
    [AE α=30 LR=0.05] Epoch 18 | TrainLoss=2.1110 | ValLoss=2.2094
    [AE α=30 LR=0.05] Epoch 19 | TrainLoss=2.0587 | ValLoss=1.9821
    [AE α=30 LR=0.05] Epoch 20 | TrainLoss=2.0362 | ValLoss=1.9431
    [AE α=30 LR=0.05] Epoch 21 | TrainLoss=2.0212 | ValLoss=2.2003
    [AE α=30 LR=0.05] Epoch 22 | TrainLoss=2.0591 | ValLoss=2.0323
    [AE α=30 LR=0.05] Epoch 23 | TrainLoss=1.9946 | ValLoss=2.3016
    [AE α=30 LR=0.05] Epoch 24 | TrainLoss=2.0973 | ValLoss=3.9811
    [AE α=30 LR=0.05] Epoch 25 | TrainLoss=2.5032 | ValLoss=2.2618
    [AE α=30 LR=0.05] Epoch 26 | TrainLoss=2.5025 | ValLoss=2.2921
    [AE α=30 LR=0.05] Epoch 27 | TrainLoss=2.3569 | ValLoss=2.2966
    [AE α=30 LR=0.05] Epoch 28 | TrainLoss=2.3253 | ValLoss=2.1441
    [AE α=30 LR=0.05] Epoch 29 | TrainLoss=2.2502 | ValLoss=2.2801
    [AE α=30 LR=0.05] Epoch 30 | TrainLoss=2.2654 | ValLoss=2.2065
    [AE α=30 LR=0.05] Epoch 31 | TrainLoss=2.2608 | ValLoss=2.1402
    [AE α=30 LR=0.05] Epoch 32 | TrainLoss=2.2135 | ValLoss=2.1060
    [AE α=30 LR=0.05] Epoch 33 | TrainLoss=2.1472 | ValLoss=2.1357
    [AE α=30 LR=0.05] Epoch 34 | TrainLoss=2.0825 | ValLoss=2.1866
    [AE α=30 LR=0.05] Epoch 35 | TrainLoss=2.0856 | ValLoss=2.2948
    Early stopping triggered.
    
    =====================================
    Training AE for α=30, LR=0.1
    =====================================
    [AE α=30 LR=0.1] Epoch 1 | TrainLoss=14.9226 | ValLoss=2.9786
    [AE α=30 LR=0.1] Epoch 2 | TrainLoss=2.9572 | ValLoss=2.7067
    [AE α=30 LR=0.1] Epoch 3 | TrainLoss=2.7323 | ValLoss=2.7122
    [AE α=30 LR=0.1] Epoch 4 | TrainLoss=2.6775 | ValLoss=2.6203
    [AE α=30 LR=0.1] Epoch 5 | TrainLoss=2.6531 | ValLoss=2.5814
    [AE α=30 LR=0.1] Epoch 6 | TrainLoss=2.6347 | ValLoss=2.5861
    [AE α=30 LR=0.1] Epoch 7 | TrainLoss=2.6283 | ValLoss=2.5642
    [AE α=30 LR=0.1] Epoch 8 | TrainLoss=2.6036 | ValLoss=2.5748
    [AE α=30 LR=0.1] Epoch 9 | TrainLoss=2.6069 | ValLoss=2.7058
    [AE α=30 LR=0.1] Epoch 10 | TrainLoss=2.6005 | ValLoss=2.5358
    [AE α=30 LR=0.1] Epoch 11 | TrainLoss=2.6655 | ValLoss=2.6382
    [AE α=30 LR=0.1] Epoch 12 | TrainLoss=2.6022 | ValLoss=2.5513
    [AE α=30 LR=0.1] Epoch 13 | TrainLoss=2.5876 | ValLoss=2.5376
    [AE α=30 LR=0.1] Epoch 14 | TrainLoss=2.5751 | ValLoss=2.5809
    [AE α=30 LR=0.1] Epoch 15 | TrainLoss=2.5828 | ValLoss=2.5456
    [AE α=30 LR=0.1] Epoch 16 | TrainLoss=2.5732 | ValLoss=2.5272
    [AE α=30 LR=0.1] Epoch 17 | TrainLoss=2.5641 | ValLoss=2.5573
    [AE α=30 LR=0.1] Epoch 18 | TrainLoss=2.6048 | ValLoss=2.7339
    [AE α=30 LR=0.1] Epoch 19 | TrainLoss=2.6122 | ValLoss=2.6778
    [AE α=30 LR=0.1] Epoch 20 | TrainLoss=2.5762 | ValLoss=2.5184
    [AE α=30 LR=0.1] Epoch 21 | TrainLoss=2.5712 | ValLoss=2.5127
    [AE α=30 LR=0.1] Epoch 22 | TrainLoss=2.5690 | ValLoss=2.5157
    [AE α=30 LR=0.1] Epoch 23 | TrainLoss=2.7830 | ValLoss=2.6166
    [AE α=30 LR=0.1] Epoch 24 | TrainLoss=2.6536 | ValLoss=2.5713
    [AE α=30 LR=0.1] Epoch 25 | TrainLoss=2.5996 | ValLoss=2.5265
    [AE α=30 LR=0.1] Epoch 26 | TrainLoss=2.5741 | ValLoss=2.5322
    [AE α=30 LR=0.1] Epoch 27 | TrainLoss=2.5741 | ValLoss=2.5057
    [AE α=30 LR=0.1] Epoch 28 | TrainLoss=134.9080 | ValLoss=2.7452
    [AE α=30 LR=0.1] Epoch 29 | TrainLoss=2.7260 | ValLoss=2.6444
    [AE α=30 LR=0.1] Epoch 30 | TrainLoss=2.6706 | ValLoss=2.5998
    [AE α=30 LR=0.1] Epoch 31 | TrainLoss=2.8597 | ValLoss=3.1257
    [AE α=30 LR=0.1] Epoch 32 | TrainLoss=2.8881 | ValLoss=2.6953
    [AE α=30 LR=0.1] Epoch 33 | TrainLoss=2.6856 | ValLoss=2.7385
    [AE α=30 LR=0.1] Epoch 34 | TrainLoss=2.6541 | ValLoss=2.5900
    [AE α=30 LR=0.1] Epoch 35 | TrainLoss=2.6347 | ValLoss=2.6008
    [AE α=30 LR=0.1] Epoch 36 | TrainLoss=2.6254 | ValLoss=2.5700
    [AE α=30 LR=0.1] Epoch 37 | TrainLoss=2.6051 | ValLoss=2.5600
    [AE α=30 LR=0.1] Epoch 38 | TrainLoss=2.5953 | ValLoss=2.5584
    [AE α=30 LR=0.1] Epoch 39 | TrainLoss=2.5978 | ValLoss=2.5371
    [AE α=30 LR=0.1] Epoch 40 | TrainLoss=2.6363 | ValLoss=2.8980
    [AE α=30 LR=0.1] Epoch 41 | TrainLoss=2.6409 | ValLoss=2.5753
    [AE α=30 LR=0.1] Epoch 42 | TrainLoss=2.5933 | ValLoss=2.5422
    Early stopping triggered.
    
    =====================================
    Training AE for α=35, LR=0.0001
    =====================================
    [AE α=35 LR=0.0001] Epoch 1 | TrainLoss=2.6227 | ValLoss=1.8438
    [AE α=35 LR=0.0001] Epoch 2 | TrainLoss=1.5644 | ValLoss=2.4116
    [AE α=35 LR=0.0001] Epoch 3 | TrainLoss=1.2520 | ValLoss=2.4457
    [AE α=35 LR=0.0001] Epoch 4 | TrainLoss=1.1008 | ValLoss=2.5792
    [AE α=35 LR=0.0001] Epoch 5 | TrainLoss=1.0096 | ValLoss=2.4671
    [AE α=35 LR=0.0001] Epoch 6 | TrainLoss=0.9479 | ValLoss=2.9936
    [AE α=35 LR=0.0001] Epoch 7 | TrainLoss=0.9182 | ValLoss=3.2433
    [AE α=35 LR=0.0001] Epoch 8 | TrainLoss=0.8666 | ValLoss=2.7263
    [AE α=35 LR=0.0001] Epoch 9 | TrainLoss=0.8351 | ValLoss=3.2713
    [AE α=35 LR=0.0001] Epoch 10 | TrainLoss=0.8009 | ValLoss=2.2580
    [AE α=35 LR=0.0001] Epoch 11 | TrainLoss=0.7889 | ValLoss=2.7205
    [AE α=35 LR=0.0001] Epoch 12 | TrainLoss=0.7774 | ValLoss=2.7863
    [AE α=35 LR=0.0001] Epoch 13 | TrainLoss=0.7329 | ValLoss=3.9758
    [AE α=35 LR=0.0001] Epoch 14 | TrainLoss=0.7188 | ValLoss=3.3095
    [AE α=35 LR=0.0001] Epoch 15 | TrainLoss=0.7014 | ValLoss=2.9796
    [AE α=35 LR=0.0001] Epoch 16 | TrainLoss=0.6802 | ValLoss=3.6222
    Early stopping triggered.
    
    =====================================
    Training AE for α=35, LR=0.0002
    =====================================
    [AE α=35 LR=0.0002] Epoch 1 | TrainLoss=2.2508 | ValLoss=2.1421
    [AE α=35 LR=0.0002] Epoch 2 | TrainLoss=1.2930 | ValLoss=1.6118
    [AE α=35 LR=0.0002] Epoch 3 | TrainLoss=1.0848 | ValLoss=2.9902
    [AE α=35 LR=0.0002] Epoch 4 | TrainLoss=0.9739 | ValLoss=1.8271
    [AE α=35 LR=0.0002] Epoch 5 | TrainLoss=0.9106 | ValLoss=1.9848
    [AE α=35 LR=0.0002] Epoch 6 | TrainLoss=0.8576 | ValLoss=2.3756
    [AE α=35 LR=0.0002] Epoch 7 | TrainLoss=0.8195 | ValLoss=1.4766
    [AE α=35 LR=0.0002] Epoch 8 | TrainLoss=0.7822 | ValLoss=1.8044
    [AE α=35 LR=0.0002] Epoch 9 | TrainLoss=0.7592 | ValLoss=2.7174
    [AE α=35 LR=0.0002] Epoch 10 | TrainLoss=0.7150 | ValLoss=3.3188
    [AE α=35 LR=0.0002] Epoch 11 | TrainLoss=0.7091 | ValLoss=2.6825
    [AE α=35 LR=0.0002] Epoch 12 | TrainLoss=0.6591 | ValLoss=2.3004
    [AE α=35 LR=0.0002] Epoch 13 | TrainLoss=0.6402 | ValLoss=2.3564
    [AE α=35 LR=0.0002] Epoch 14 | TrainLoss=0.6171 | ValLoss=2.8625
    [AE α=35 LR=0.0002] Epoch 15 | TrainLoss=0.6047 | ValLoss=3.1559
    [AE α=35 LR=0.0002] Epoch 16 | TrainLoss=0.5981 | ValLoss=2.7297
    [AE α=35 LR=0.0002] Epoch 17 | TrainLoss=0.5756 | ValLoss=2.9829
    [AE α=35 LR=0.0002] Epoch 18 | TrainLoss=0.5618 | ValLoss=2.3691
    [AE α=35 LR=0.0002] Epoch 19 | TrainLoss=0.5501 | ValLoss=1.9478
    [AE α=35 LR=0.0002] Epoch 20 | TrainLoss=0.5426 | ValLoss=2.4680
    [AE α=35 LR=0.0002] Epoch 21 | TrainLoss=0.5126 | ValLoss=3.7022
    [AE α=35 LR=0.0002] Epoch 22 | TrainLoss=0.5114 | ValLoss=2.7847
    Early stopping triggered.
    
    =====================================
    Training AE for α=35, LR=0.0005
    =====================================
    [AE α=35 LR=0.0005] Epoch 1 | TrainLoss=1.8459 | ValLoss=1.6984
    [AE α=35 LR=0.0005] Epoch 2 | TrainLoss=1.1857 | ValLoss=1.2279
    [AE α=35 LR=0.0005] Epoch 3 | TrainLoss=0.9917 | ValLoss=1.1326
    [AE α=35 LR=0.0005] Epoch 4 | TrainLoss=0.9073 | ValLoss=0.9165
    [AE α=35 LR=0.0005] Epoch 5 | TrainLoss=0.8456 | ValLoss=1.6064
    [AE α=35 LR=0.0005] Epoch 6 | TrainLoss=0.7841 | ValLoss=1.7244
    [AE α=35 LR=0.0005] Epoch 7 | TrainLoss=0.7494 | ValLoss=3.1350
    [AE α=35 LR=0.0005] Epoch 8 | TrainLoss=0.7102 | ValLoss=1.7192
    [AE α=35 LR=0.0005] Epoch 9 | TrainLoss=0.6662 | ValLoss=0.9836
    [AE α=35 LR=0.0005] Epoch 10 | TrainLoss=0.6491 | ValLoss=0.9014
    [AE α=35 LR=0.0005] Epoch 11 | TrainLoss=0.6140 | ValLoss=2.5341
    [AE α=35 LR=0.0005] Epoch 12 | TrainLoss=0.5919 | ValLoss=2.1224
    [AE α=35 LR=0.0005] Epoch 13 | TrainLoss=0.5727 | ValLoss=1.2979
    [AE α=35 LR=0.0005] Epoch 14 | TrainLoss=0.5612 | ValLoss=1.2305
    [AE α=35 LR=0.0005] Epoch 15 | TrainLoss=0.5450 | ValLoss=1.0438
    [AE α=35 LR=0.0005] Epoch 16 | TrainLoss=0.5243 | ValLoss=2.1701
    [AE α=35 LR=0.0005] Epoch 17 | TrainLoss=0.5202 | ValLoss=0.7675
    [AE α=35 LR=0.0005] Epoch 18 | TrainLoss=0.5073 | ValLoss=2.6164
    [AE α=35 LR=0.0005] Epoch 19 | TrainLoss=0.4965 | ValLoss=1.7159
    [AE α=35 LR=0.0005] Epoch 20 | TrainLoss=0.5040 | ValLoss=2.3369
    [AE α=35 LR=0.0005] Epoch 21 | TrainLoss=0.4650 | ValLoss=1.3341
    [AE α=35 LR=0.0005] Epoch 22 | TrainLoss=0.4574 | ValLoss=1.0781
    [AE α=35 LR=0.0005] Epoch 23 | TrainLoss=0.4533 | ValLoss=1.4409
    [AE α=35 LR=0.0005] Epoch 24 | TrainLoss=0.4479 | ValLoss=1.0392
    [AE α=35 LR=0.0005] Epoch 25 | TrainLoss=0.4412 | ValLoss=1.6482
    [AE α=35 LR=0.0005] Epoch 26 | TrainLoss=0.4374 | ValLoss=1.3207
    [AE α=35 LR=0.0005] Epoch 27 | TrainLoss=0.4264 | ValLoss=1.0018
    [AE α=35 LR=0.0005] Epoch 28 | TrainLoss=0.4168 | ValLoss=2.6540
    [AE α=35 LR=0.0005] Epoch 29 | TrainLoss=0.4135 | ValLoss=1.8296
    [AE α=35 LR=0.0005] Epoch 30 | TrainLoss=0.4043 | ValLoss=1.3375
    [AE α=35 LR=0.0005] Epoch 31 | TrainLoss=0.3969 | ValLoss=1.9559
    [AE α=35 LR=0.0005] Epoch 32 | TrainLoss=0.3897 | ValLoss=1.8800
    Early stopping triggered.
    
    =====================================
    Training AE for α=35, LR=0.001
    =====================================
    [AE α=35 LR=0.001] Epoch 1 | TrainLoss=1.7898 | ValLoss=1.5469
    [AE α=35 LR=0.001] Epoch 2 | TrainLoss=1.1437 | ValLoss=1.3294
    [AE α=35 LR=0.001] Epoch 3 | TrainLoss=0.9971 | ValLoss=1.3430
    [AE α=35 LR=0.001] Epoch 4 | TrainLoss=0.9092 | ValLoss=1.1613
    [AE α=35 LR=0.001] Epoch 5 | TrainLoss=0.8537 | ValLoss=1.1857
    [AE α=35 LR=0.001] Epoch 6 | TrainLoss=0.8061 | ValLoss=0.8330
    [AE α=35 LR=0.001] Epoch 7 | TrainLoss=0.7400 | ValLoss=1.3181
    [AE α=35 LR=0.001] Epoch 8 | TrainLoss=0.7019 | ValLoss=0.8935
    [AE α=35 LR=0.001] Epoch 9 | TrainLoss=0.6747 | ValLoss=1.6064
    [AE α=35 LR=0.001] Epoch 10 | TrainLoss=0.6301 | ValLoss=1.3083
    [AE α=35 LR=0.001] Epoch 11 | TrainLoss=0.6074 | ValLoss=1.7023
    [AE α=35 LR=0.001] Epoch 12 | TrainLoss=0.5660 | ValLoss=1.4753
    [AE α=35 LR=0.001] Epoch 13 | TrainLoss=0.5615 | ValLoss=1.0151
    [AE α=35 LR=0.001] Epoch 14 | TrainLoss=0.5589 | ValLoss=0.6383
    [AE α=35 LR=0.001] Epoch 15 | TrainLoss=0.5369 | ValLoss=1.5474
    [AE α=35 LR=0.001] Epoch 16 | TrainLoss=0.5134 | ValLoss=1.4899
    [AE α=35 LR=0.001] Epoch 17 | TrainLoss=0.5043 | ValLoss=1.3124
    [AE α=35 LR=0.001] Epoch 18 | TrainLoss=0.4934 | ValLoss=1.0004
    [AE α=35 LR=0.001] Epoch 19 | TrainLoss=0.4857 | ValLoss=1.4021
    [AE α=35 LR=0.001] Epoch 20 | TrainLoss=0.4717 | ValLoss=1.1513
    [AE α=35 LR=0.001] Epoch 21 | TrainLoss=0.4528 | ValLoss=1.4200
    [AE α=35 LR=0.001] Epoch 22 | TrainLoss=0.4515 | ValLoss=0.9900
    [AE α=35 LR=0.001] Epoch 23 | TrainLoss=0.4424 | ValLoss=0.6982
    [AE α=35 LR=0.001] Epoch 24 | TrainLoss=0.4170 | ValLoss=1.9861
    [AE α=35 LR=0.001] Epoch 25 | TrainLoss=0.4168 | ValLoss=2.2236
    [AE α=35 LR=0.001] Epoch 26 | TrainLoss=0.4107 | ValLoss=1.6287
    [AE α=35 LR=0.001] Epoch 27 | TrainLoss=0.3975 | ValLoss=1.1417
    [AE α=35 LR=0.001] Epoch 28 | TrainLoss=0.3981 | ValLoss=1.0233
    [AE α=35 LR=0.001] Epoch 29 | TrainLoss=0.3869 | ValLoss=1.3378
    Early stopping triggered.
    
    =====================================
    Training AE for α=35, LR=0.002
    =====================================
    [AE α=35 LR=0.002] Epoch 1 | TrainLoss=1.8179 | ValLoss=2.5706
    [AE α=35 LR=0.002] Epoch 2 | TrainLoss=1.1753 | ValLoss=1.3360
    [AE α=35 LR=0.002] Epoch 3 | TrainLoss=1.0480 | ValLoss=1.2463
    [AE α=35 LR=0.002] Epoch 4 | TrainLoss=0.9467 | ValLoss=2.9487
    [AE α=35 LR=0.002] Epoch 5 | TrainLoss=0.9160 | ValLoss=2.5421
    [AE α=35 LR=0.002] Epoch 6 | TrainLoss=0.8561 | ValLoss=1.6439
    [AE α=35 LR=0.002] Epoch 7 | TrainLoss=0.8156 | ValLoss=0.9313
    [AE α=35 LR=0.002] Epoch 8 | TrainLoss=0.7784 | ValLoss=1.9499
    [AE α=35 LR=0.002] Epoch 9 | TrainLoss=0.7315 | ValLoss=1.8354
    [AE α=35 LR=0.002] Epoch 10 | TrainLoss=0.7080 | ValLoss=0.7644
    [AE α=35 LR=0.002] Epoch 11 | TrainLoss=0.6783 | ValLoss=1.2822
    [AE α=35 LR=0.002] Epoch 12 | TrainLoss=0.6403 | ValLoss=1.0170
    [AE α=35 LR=0.002] Epoch 13 | TrainLoss=0.6205 | ValLoss=0.9932
    [AE α=35 LR=0.002] Epoch 14 | TrainLoss=0.6016 | ValLoss=0.7021
    [AE α=35 LR=0.002] Epoch 15 | TrainLoss=0.5754 | ValLoss=1.1694
    [AE α=35 LR=0.002] Epoch 16 | TrainLoss=0.5652 | ValLoss=1.9505
    [AE α=35 LR=0.002] Epoch 17 | TrainLoss=0.5436 | ValLoss=1.5973
    [AE α=35 LR=0.002] Epoch 18 | TrainLoss=0.5419 | ValLoss=0.8892
    [AE α=35 LR=0.002] Epoch 19 | TrainLoss=0.5069 | ValLoss=2.3503
    [AE α=35 LR=0.002] Epoch 20 | TrainLoss=0.5048 | ValLoss=1.8080
    [AE α=35 LR=0.002] Epoch 21 | TrainLoss=0.4833 | ValLoss=1.1169
    [AE α=35 LR=0.002] Epoch 22 | TrainLoss=0.4750 | ValLoss=0.7644
    [AE α=35 LR=0.002] Epoch 23 | TrainLoss=0.4668 | ValLoss=1.2223
    [AE α=35 LR=0.002] Epoch 24 | TrainLoss=0.4662 | ValLoss=1.1196
    [AE α=35 LR=0.002] Epoch 25 | TrainLoss=0.4509 | ValLoss=1.2252
    [AE α=35 LR=0.002] Epoch 26 | TrainLoss=0.4404 | ValLoss=2.5244
    [AE α=35 LR=0.002] Epoch 27 | TrainLoss=0.4390 | ValLoss=1.7540
    [AE α=35 LR=0.002] Epoch 28 | TrainLoss=0.4359 | ValLoss=1.5487
    [AE α=35 LR=0.002] Epoch 29 | TrainLoss=0.4099 | ValLoss=1.8194
    Early stopping triggered.
    
    =====================================
    Training AE for α=35, LR=0.005
    =====================================
    [AE α=35 LR=0.005] Epoch 1 | TrainLoss=2.0272 | ValLoss=3.6428
    [AE α=35 LR=0.005] Epoch 2 | TrainLoss=1.3670 | ValLoss=1.8702
    [AE α=35 LR=0.005] Epoch 3 | TrainLoss=1.1485 | ValLoss=1.4912
    [AE α=35 LR=0.005] Epoch 4 | TrainLoss=1.0674 | ValLoss=1.0865
    [AE α=35 LR=0.005] Epoch 5 | TrainLoss=0.9765 | ValLoss=1.3374
    [AE α=35 LR=0.005] Epoch 6 | TrainLoss=0.9262 | ValLoss=1.3357
    [AE α=35 LR=0.005] Epoch 7 | TrainLoss=0.8707 | ValLoss=1.6883
    [AE α=35 LR=0.005] Epoch 8 | TrainLoss=0.8293 | ValLoss=3.2089
    [AE α=35 LR=0.005] Epoch 9 | TrainLoss=0.7951 | ValLoss=2.8814
    [AE α=35 LR=0.005] Epoch 10 | TrainLoss=0.7562 | ValLoss=0.9401
    [AE α=35 LR=0.005] Epoch 11 | TrainLoss=0.7111 | ValLoss=1.5226
    [AE α=35 LR=0.005] Epoch 12 | TrainLoss=0.6946 | ValLoss=0.8949
    [AE α=35 LR=0.005] Epoch 13 | TrainLoss=0.6834 | ValLoss=1.3444
    [AE α=35 LR=0.005] Epoch 14 | TrainLoss=0.6395 | ValLoss=2.1399
    [AE α=35 LR=0.005] Epoch 15 | TrainLoss=0.6227 | ValLoss=0.8645
    [AE α=35 LR=0.005] Epoch 16 | TrainLoss=0.6033 | ValLoss=0.9114
    [AE α=35 LR=0.005] Epoch 17 | TrainLoss=0.5928 | ValLoss=1.3517
    [AE α=35 LR=0.005] Epoch 18 | TrainLoss=0.5942 | ValLoss=2.4593
    [AE α=35 LR=0.005] Epoch 19 | TrainLoss=0.5643 | ValLoss=1.9158
    [AE α=35 LR=0.005] Epoch 20 | TrainLoss=0.5689 | ValLoss=1.0751
    [AE α=35 LR=0.005] Epoch 21 | TrainLoss=0.5325 | ValLoss=1.1096
    [AE α=35 LR=0.005] Epoch 22 | TrainLoss=0.5307 | ValLoss=0.9572
    [AE α=35 LR=0.005] Epoch 23 | TrainLoss=0.5083 | ValLoss=1.3843
    [AE α=35 LR=0.005] Epoch 24 | TrainLoss=0.5098 | ValLoss=2.2001
    [AE α=35 LR=0.005] Epoch 25 | TrainLoss=0.5167 | ValLoss=0.5397
    [AE α=35 LR=0.005] Epoch 26 | TrainLoss=0.4856 | ValLoss=2.7300
    [AE α=35 LR=0.005] Epoch 27 | TrainLoss=0.4889 | ValLoss=2.1040
    [AE α=35 LR=0.005] Epoch 28 | TrainLoss=0.4726 | ValLoss=1.8738
    [AE α=35 LR=0.005] Epoch 29 | TrainLoss=0.4709 | ValLoss=5.0243
    [AE α=35 LR=0.005] Epoch 30 | TrainLoss=0.4724 | ValLoss=1.2940
    [AE α=35 LR=0.005] Epoch 31 | TrainLoss=0.4815 | ValLoss=1.9601
    [AE α=35 LR=0.005] Epoch 32 | TrainLoss=0.4518 | ValLoss=1.7415
    [AE α=35 LR=0.005] Epoch 33 | TrainLoss=0.4628 | ValLoss=1.6118
    [AE α=35 LR=0.005] Epoch 34 | TrainLoss=0.4494 | ValLoss=0.9737
    [AE α=35 LR=0.005] Epoch 35 | TrainLoss=0.4372 | ValLoss=0.9971
    [AE α=35 LR=0.005] Epoch 36 | TrainLoss=0.4375 | ValLoss=1.4718
    [AE α=35 LR=0.005] Epoch 37 | TrainLoss=0.4344 | ValLoss=2.6445
    [AE α=35 LR=0.005] Epoch 38 | TrainLoss=0.4354 | ValLoss=1.1815
    [AE α=35 LR=0.005] Epoch 39 | TrainLoss=0.4199 | ValLoss=2.8371
    [AE α=35 LR=0.005] Epoch 40 | TrainLoss=0.4218 | ValLoss=1.4884
    Early stopping triggered.
    
    New best AE
       α=35, LR=0.005, ValLoss=0.5397
    
    =====================================
    Training AE for α=35, LR=0.01
    =====================================
    [AE α=35 LR=0.01] Epoch 1 | TrainLoss=2.3912 | ValLoss=4.0870
    [AE α=35 LR=0.01] Epoch 2 | TrainLoss=1.6826 | ValLoss=1.5756
    [AE α=35 LR=0.01] Epoch 3 | TrainLoss=1.3575 | ValLoss=1.3879
    [AE α=35 LR=0.01] Epoch 4 | TrainLoss=1.2076 | ValLoss=1.4973
    [AE α=35 LR=0.01] Epoch 5 | TrainLoss=1.1389 | ValLoss=1.4745
    [AE α=35 LR=0.01] Epoch 6 | TrainLoss=1.0782 | ValLoss=1.2413
    [AE α=35 LR=0.01] Epoch 7 | TrainLoss=1.0422 | ValLoss=1.1067
    [AE α=35 LR=0.01] Epoch 8 | TrainLoss=0.9916 | ValLoss=1.2789
    [AE α=35 LR=0.01] Epoch 9 | TrainLoss=0.9158 | ValLoss=1.4803
    [AE α=35 LR=0.01] Epoch 10 | TrainLoss=0.9212 | ValLoss=1.0152
    [AE α=35 LR=0.01] Epoch 11 | TrainLoss=0.8804 | ValLoss=1.3447
    [AE α=35 LR=0.01] Epoch 12 | TrainLoss=0.8395 | ValLoss=1.7058
    [AE α=35 LR=0.01] Epoch 13 | TrainLoss=0.8262 | ValLoss=0.9582
    [AE α=35 LR=0.01] Epoch 14 | TrainLoss=0.8164 | ValLoss=1.0538
    [AE α=35 LR=0.01] Epoch 15 | TrainLoss=0.7778 | ValLoss=1.0236
    [AE α=35 LR=0.01] Epoch 16 | TrainLoss=0.7635 | ValLoss=2.0717
    [AE α=35 LR=0.01] Epoch 17 | TrainLoss=0.7358 | ValLoss=1.2344
    [AE α=35 LR=0.01] Epoch 18 | TrainLoss=0.7045 | ValLoss=0.7837
    [AE α=35 LR=0.01] Epoch 19 | TrainLoss=0.7024 | ValLoss=1.1519
    [AE α=35 LR=0.01] Epoch 20 | TrainLoss=0.6776 | ValLoss=1.4114
    [AE α=35 LR=0.01] Epoch 21 | TrainLoss=0.6587 | ValLoss=1.8797
    [AE α=35 LR=0.01] Epoch 22 | TrainLoss=0.6695 | ValLoss=2.4222
    [AE α=35 LR=0.01] Epoch 23 | TrainLoss=0.6514 | ValLoss=1.5389
    [AE α=35 LR=0.01] Epoch 24 | TrainLoss=0.6327 | ValLoss=1.5643
    [AE α=35 LR=0.01] Epoch 25 | TrainLoss=0.6280 | ValLoss=0.9201
    [AE α=35 LR=0.01] Epoch 26 | TrainLoss=0.6104 | ValLoss=1.4910
    [AE α=35 LR=0.01] Epoch 27 | TrainLoss=0.6022 | ValLoss=1.4671
    [AE α=35 LR=0.01] Epoch 28 | TrainLoss=0.5757 | ValLoss=2.3966
    [AE α=35 LR=0.01] Epoch 29 | TrainLoss=0.6004 | ValLoss=1.1156
    [AE α=35 LR=0.01] Epoch 30 | TrainLoss=0.5927 | ValLoss=0.9395
    [AE α=35 LR=0.01] Epoch 31 | TrainLoss=0.5762 | ValLoss=1.0676
    [AE α=35 LR=0.01] Epoch 32 | TrainLoss=0.5602 | ValLoss=1.0390
    [AE α=35 LR=0.01] Epoch 33 | TrainLoss=0.5697 | ValLoss=2.1369
    Early stopping triggered.
    
    =====================================
    Training AE for α=35, LR=0.05
    =====================================
    [AE α=35 LR=0.05] Epoch 1 | TrainLoss=5.8372 | ValLoss=2.7508
    [AE α=35 LR=0.05] Epoch 2 | TrainLoss=2.7924 | ValLoss=2.6965
    [AE α=35 LR=0.05] Epoch 3 | TrainLoss=2.6813 | ValLoss=2.6411
    [AE α=35 LR=0.05] Epoch 4 | TrainLoss=2.6344 | ValLoss=2.6050
    [AE α=35 LR=0.05] Epoch 5 | TrainLoss=2.6124 | ValLoss=2.5551
    [AE α=35 LR=0.05] Epoch 6 | TrainLoss=2.5926 | ValLoss=2.5477
    [AE α=35 LR=0.05] Epoch 7 | TrainLoss=2.5859 | ValLoss=2.5774
    [AE α=35 LR=0.05] Epoch 8 | TrainLoss=2.5832 | ValLoss=2.5374
    [AE α=35 LR=0.05] Epoch 9 | TrainLoss=2.8170 | ValLoss=2.6871
    [AE α=35 LR=0.05] Epoch 10 | TrainLoss=2.6588 | ValLoss=2.5903
    [AE α=35 LR=0.05] Epoch 11 | TrainLoss=2.6196 | ValLoss=2.5877
    [AE α=35 LR=0.05] Epoch 12 | TrainLoss=2.6000 | ValLoss=2.6155
    [AE α=35 LR=0.05] Epoch 13 | TrainLoss=2.5882 | ValLoss=2.5585
    [AE α=35 LR=0.05] Epoch 14 | TrainLoss=2.5794 | ValLoss=2.5518
    [AE α=35 LR=0.05] Epoch 15 | TrainLoss=2.5677 | ValLoss=2.5379
    [AE α=35 LR=0.05] Epoch 16 | TrainLoss=2.5697 | ValLoss=2.5473
    [AE α=35 LR=0.05] Epoch 17 | TrainLoss=2.5579 | ValLoss=2.5413
    [AE α=35 LR=0.05] Epoch 18 | TrainLoss=2.5631 | ValLoss=2.5213
    [AE α=35 LR=0.05] Epoch 19 | TrainLoss=2.5538 | ValLoss=2.5194
    [AE α=35 LR=0.05] Epoch 20 | TrainLoss=2.5560 | ValLoss=2.5461
    [AE α=35 LR=0.05] Epoch 21 | TrainLoss=2.5908 | ValLoss=2.5294
    [AE α=35 LR=0.05] Epoch 22 | TrainLoss=2.5616 | ValLoss=2.5199
    [AE α=35 LR=0.05] Epoch 23 | TrainLoss=2.5576 | ValLoss=2.5407
    [AE α=35 LR=0.05] Epoch 24 | TrainLoss=2.5496 | ValLoss=2.5106
    [AE α=35 LR=0.05] Epoch 25 | TrainLoss=2.5477 | ValLoss=2.5349
    [AE α=35 LR=0.05] Epoch 26 | TrainLoss=2.5430 | ValLoss=2.5112
    [AE α=35 LR=0.05] Epoch 27 | TrainLoss=9.7443 | ValLoss=2.6308
    [AE α=35 LR=0.05] Epoch 28 | TrainLoss=2.6680 | ValLoss=2.5691
    [AE α=35 LR=0.05] Epoch 29 | TrainLoss=2.6065 | ValLoss=2.5757
    [AE α=35 LR=0.05] Epoch 30 | TrainLoss=2.5890 | ValLoss=2.5217
    [AE α=35 LR=0.05] Epoch 31 | TrainLoss=2.5732 | ValLoss=2.5200
    [AE α=35 LR=0.05] Epoch 32 | TrainLoss=2.5647 | ValLoss=2.5067
    [AE α=35 LR=0.05] Epoch 33 | TrainLoss=2.5587 | ValLoss=2.5198
    [AE α=35 LR=0.05] Epoch 34 | TrainLoss=2.5973 | ValLoss=2.6655
    [AE α=35 LR=0.05] Epoch 35 | TrainLoss=2.6127 | ValLoss=2.5357
    [AE α=35 LR=0.05] Epoch 36 | TrainLoss=2.5632 | ValLoss=2.5122
    [AE α=35 LR=0.05] Epoch 37 | TrainLoss=2.5861 | ValLoss=2.5254
    [AE α=35 LR=0.05] Epoch 38 | TrainLoss=2.5704 | ValLoss=2.5179
    [AE α=35 LR=0.05] Epoch 39 | TrainLoss=2.5594 | ValLoss=2.5165
    [AE α=35 LR=0.05] Epoch 40 | TrainLoss=2.5639 | ValLoss=2.5109
    [AE α=35 LR=0.05] Epoch 41 | TrainLoss=2.5545 | ValLoss=2.4956
    [AE α=35 LR=0.05] Epoch 42 | TrainLoss=2.5492 | ValLoss=2.5142
    [AE α=35 LR=0.05] Epoch 43 | TrainLoss=2.5447 | ValLoss=2.5124
    [AE α=35 LR=0.05] Epoch 44 | TrainLoss=2.5528 | ValLoss=2.5152
    [AE α=35 LR=0.05] Epoch 45 | TrainLoss=2.5609 | ValLoss=2.5079
    [AE α=35 LR=0.05] Epoch 46 | TrainLoss=2.5478 | ValLoss=2.4993
    [AE α=35 LR=0.05] Epoch 47 | TrainLoss=2.5466 | ValLoss=2.4924
    [AE α=35 LR=0.05] Epoch 48 | TrainLoss=2.5388 | ValLoss=2.5000
    [AE α=35 LR=0.05] Epoch 49 | TrainLoss=2.5404 | ValLoss=2.4920
    [AE α=35 LR=0.05] Epoch 50 | TrainLoss=2.5377 | ValLoss=2.4947
    [AE α=35 LR=0.05] Epoch 51 | TrainLoss=2.5303 | ValLoss=4.1169
    [AE α=35 LR=0.05] Epoch 52 | TrainLoss=2.5579 | ValLoss=2.5751
    [AE α=35 LR=0.05] Epoch 53 | TrainLoss=2.5729 | ValLoss=2.5377
    [AE α=35 LR=0.05] Epoch 54 | TrainLoss=2.5513 | ValLoss=2.4961
    [AE α=35 LR=0.05] Epoch 55 | TrainLoss=2.5385 | ValLoss=2.4971
    [AE α=35 LR=0.05] Epoch 56 | TrainLoss=2.5314 | ValLoss=2.5068
    [AE α=35 LR=0.05] Epoch 57 | TrainLoss=2.5342 | ValLoss=2.4897
    [AE α=35 LR=0.05] Epoch 58 | TrainLoss=2.5286 | ValLoss=2.5024
    [AE α=35 LR=0.05] Epoch 59 | TrainLoss=2.5246 | ValLoss=2.4861
    [AE α=35 LR=0.05] Epoch 60 | TrainLoss=2.5244 | ValLoss=2.4803
    [AE α=35 LR=0.05] Epoch 61 | TrainLoss=2.8855 | ValLoss=2.6922
    [AE α=35 LR=0.05] Epoch 62 | TrainLoss=2.6678 | ValLoss=2.5199
    [AE α=35 LR=0.05] Epoch 63 | TrainLoss=2.5942 | ValLoss=2.5094
    [AE α=35 LR=0.05] Epoch 64 | TrainLoss=2.5663 | ValLoss=2.4930
    [AE α=35 LR=0.05] Epoch 65 | TrainLoss=2.5618 | ValLoss=2.4833
    [AE α=35 LR=0.05] Epoch 66 | TrainLoss=2.5634 | ValLoss=2.4793
    [AE α=35 LR=0.05] Epoch 67 | TrainLoss=2.5558 | ValLoss=2.4996
    [AE α=35 LR=0.05] Epoch 68 | TrainLoss=3.8690 | ValLoss=2.6182
    [AE α=35 LR=0.05] Epoch 69 | TrainLoss=2.5928 | ValLoss=2.5019
    [AE α=35 LR=0.05] Epoch 70 | TrainLoss=2.5707 | ValLoss=2.5063
    [AE α=35 LR=0.05] Epoch 71 | TrainLoss=2.5681 | ValLoss=2.5021
    [AE α=35 LR=0.05] Epoch 72 | TrainLoss=2.5560 | ValLoss=2.5267
    [AE α=35 LR=0.05] Epoch 73 | TrainLoss=2.5565 | ValLoss=2.4960
    [AE α=35 LR=0.05] Epoch 74 | TrainLoss=2.5533 | ValLoss=2.4936
    [AE α=35 LR=0.05] Epoch 75 | TrainLoss=2.5467 | ValLoss=2.4785
    [AE α=35 LR=0.05] Epoch 76 | TrainLoss=2.5503 | ValLoss=2.4917
    [AE α=35 LR=0.05] Epoch 77 | TrainLoss=2.5475 | ValLoss=2.4745
    [AE α=35 LR=0.05] Epoch 78 | TrainLoss=2.5440 | ValLoss=2.4837
    [AE α=35 LR=0.05] Epoch 79 | TrainLoss=2.5474 | ValLoss=2.4819
    [AE α=35 LR=0.05] Epoch 80 | TrainLoss=5.0783 | ValLoss=2.5611
    
    =====================================
    Training AE for α=35, LR=0.1
    =====================================
    [AE α=35 LR=0.1] Epoch 1 | TrainLoss=27.5139 | ValLoss=2.5606
    [AE α=35 LR=0.1] Epoch 2 | TrainLoss=2.6633 | ValLoss=2.3789
    [AE α=35 LR=0.1] Epoch 3 | TrainLoss=2.5929 | ValLoss=2.9284
    [AE α=35 LR=0.1] Epoch 4 | TrainLoss=2.4799 | ValLoss=2.4329
    [AE α=35 LR=0.1] Epoch 5 | TrainLoss=2.4428 | ValLoss=2.5728
    [AE α=35 LR=0.1] Epoch 6 | TrainLoss=2.3464 | ValLoss=2.3890
    [AE α=35 LR=0.1] Epoch 7 | TrainLoss=2.3276 | ValLoss=2.3582
    [AE α=35 LR=0.1] Epoch 8 | TrainLoss=2.8945 | ValLoss=2.7442
    [AE α=35 LR=0.1] Epoch 9 | TrainLoss=2.8791 | ValLoss=2.7089
    [AE α=35 LR=0.1] Epoch 10 | TrainLoss=2.7423 | ValLoss=2.6443
    [AE α=35 LR=0.1] Epoch 11 | TrainLoss=2.6844 | ValLoss=2.5982
    [AE α=35 LR=0.1] Epoch 12 | TrainLoss=2.6827 | ValLoss=2.5939
    [AE α=35 LR=0.1] Epoch 13 | TrainLoss=2.6714 | ValLoss=2.5827
    [AE α=35 LR=0.1] Epoch 14 | TrainLoss=2.6956 | ValLoss=2.5897
    [AE α=35 LR=0.1] Epoch 15 | TrainLoss=2.6589 | ValLoss=2.5955
    [AE α=35 LR=0.1] Epoch 16 | TrainLoss=2.6513 | ValLoss=2.5851
    [AE α=35 LR=0.1] Epoch 17 | TrainLoss=2.6462 | ValLoss=2.6139
    [AE α=35 LR=0.1] Epoch 18 | TrainLoss=2.6477 | ValLoss=2.5798
    [AE α=35 LR=0.1] Epoch 19 | TrainLoss=2.6429 | ValLoss=2.5739
    [AE α=35 LR=0.1] Epoch 20 | TrainLoss=2.6462 | ValLoss=2.6167
    [AE α=35 LR=0.1] Epoch 21 | TrainLoss=2.6304 | ValLoss=2.5883
    [AE α=35 LR=0.1] Epoch 22 | TrainLoss=2.6292 | ValLoss=2.5764
    Early stopping triggered.
    
    =====================================
    Training AE for α=40, LR=0.0001
    =====================================
    [AE α=40 LR=0.0001] Epoch 1 | TrainLoss=2.8638 | ValLoss=1.8167
    [AE α=40 LR=0.0001] Epoch 2 | TrainLoss=1.6789 | ValLoss=1.8466
    [AE α=40 LR=0.0001] Epoch 3 | TrainLoss=1.3043 | ValLoss=1.9890
    [AE α=40 LR=0.0001] Epoch 4 | TrainLoss=1.1545 | ValLoss=1.7696
    [AE α=40 LR=0.0001] Epoch 5 | TrainLoss=1.0646 | ValLoss=2.0285
    [AE α=40 LR=0.0001] Epoch 6 | TrainLoss=0.9928 | ValLoss=2.0694
    [AE α=40 LR=0.0001] Epoch 7 | TrainLoss=0.9393 | ValLoss=2.2489
    [AE α=40 LR=0.0001] Epoch 8 | TrainLoss=0.8982 | ValLoss=2.2042
    [AE α=40 LR=0.0001] Epoch 9 | TrainLoss=0.8702 | ValLoss=2.3435
    [AE α=40 LR=0.0001] Epoch 10 | TrainLoss=0.8395 | ValLoss=1.8093
    [AE α=40 LR=0.0001] Epoch 11 | TrainLoss=0.7957 | ValLoss=1.9540
    [AE α=40 LR=0.0001] Epoch 12 | TrainLoss=0.7719 | ValLoss=1.8495
    [AE α=40 LR=0.0001] Epoch 13 | TrainLoss=0.7638 | ValLoss=2.3706
    [AE α=40 LR=0.0001] Epoch 14 | TrainLoss=0.7409 | ValLoss=1.9987
    [AE α=40 LR=0.0001] Epoch 15 | TrainLoss=0.7334 | ValLoss=2.5799
    [AE α=40 LR=0.0001] Epoch 16 | TrainLoss=0.6985 | ValLoss=2.7995
    [AE α=40 LR=0.0001] Epoch 17 | TrainLoss=0.6781 | ValLoss=2.6832
    [AE α=40 LR=0.0001] Epoch 18 | TrainLoss=0.6760 | ValLoss=2.3312
    [AE α=40 LR=0.0001] Epoch 19 | TrainLoss=0.6700 | ValLoss=2.4723
    Early stopping triggered.
    
    =====================================
    Training AE for α=40, LR=0.0002
    =====================================
    [AE α=40 LR=0.0002] Epoch 1 | TrainLoss=2.3238 | ValLoss=1.7906
    [AE α=40 LR=0.0002] Epoch 2 | TrainLoss=1.3582 | ValLoss=2.5152
    [AE α=40 LR=0.0002] Epoch 3 | TrainLoss=1.1158 | ValLoss=3.4252
    [AE α=40 LR=0.0002] Epoch 4 | TrainLoss=1.0069 | ValLoss=2.5855
    [AE α=40 LR=0.0002] Epoch 5 | TrainLoss=0.9523 | ValLoss=2.7216
    [AE α=40 LR=0.0002] Epoch 6 | TrainLoss=0.8882 | ValLoss=2.4813
    [AE α=40 LR=0.0002] Epoch 7 | TrainLoss=0.8557 | ValLoss=2.2205
    [AE α=40 LR=0.0002] Epoch 8 | TrainLoss=0.8139 | ValLoss=2.7580
    [AE α=40 LR=0.0002] Epoch 9 | TrainLoss=0.7936 | ValLoss=2.1487
    [AE α=40 LR=0.0002] Epoch 10 | TrainLoss=0.7553 | ValLoss=1.5260
    [AE α=40 LR=0.0002] Epoch 11 | TrainLoss=0.7274 | ValLoss=2.7591
    [AE α=40 LR=0.0002] Epoch 12 | TrainLoss=0.7001 | ValLoss=2.0746
    [AE α=40 LR=0.0002] Epoch 13 | TrainLoss=0.6841 | ValLoss=2.1515
    [AE α=40 LR=0.0002] Epoch 14 | TrainLoss=0.6638 | ValLoss=1.7481
    [AE α=40 LR=0.0002] Epoch 15 | TrainLoss=0.6495 | ValLoss=2.3568
    [AE α=40 LR=0.0002] Epoch 16 | TrainLoss=0.6250 | ValLoss=2.4039
    [AE α=40 LR=0.0002] Epoch 17 | TrainLoss=0.6116 | ValLoss=3.0394
    [AE α=40 LR=0.0002] Epoch 18 | TrainLoss=0.5978 | ValLoss=2.7712
    [AE α=40 LR=0.0002] Epoch 19 | TrainLoss=0.5907 | ValLoss=2.3341
    [AE α=40 LR=0.0002] Epoch 20 | TrainLoss=0.5726 | ValLoss=2.2552
    [AE α=40 LR=0.0002] Epoch 21 | TrainLoss=0.5647 | ValLoss=2.0015
    [AE α=40 LR=0.0002] Epoch 22 | TrainLoss=0.5601 | ValLoss=2.6302
    [AE α=40 LR=0.0002] Epoch 23 | TrainLoss=0.5569 | ValLoss=2.3076
    [AE α=40 LR=0.0002] Epoch 24 | TrainLoss=0.5252 | ValLoss=2.6671
    [AE α=40 LR=0.0002] Epoch 25 | TrainLoss=0.5276 | ValLoss=2.0231
    Early stopping triggered.
    
    =====================================
    Training AE for α=40, LR=0.0005
    =====================================
    [AE α=40 LR=0.0005] Epoch 1 | TrainLoss=2.1524 | ValLoss=1.9457
    [AE α=40 LR=0.0005] Epoch 2 | TrainLoss=1.2091 | ValLoss=1.2290
    [AE α=40 LR=0.0005] Epoch 3 | TrainLoss=1.0369 | ValLoss=3.1502
    [AE α=40 LR=0.0005] Epoch 4 | TrainLoss=0.9510 | ValLoss=1.2843
    [AE α=40 LR=0.0005] Epoch 5 | TrainLoss=0.9017 | ValLoss=1.8072
    [AE α=40 LR=0.0005] Epoch 6 | TrainLoss=0.8291 | ValLoss=1.7387
    [AE α=40 LR=0.0005] Epoch 7 | TrainLoss=0.7962 | ValLoss=1.5985
    [AE α=40 LR=0.0005] Epoch 8 | TrainLoss=0.7558 | ValLoss=2.1412
    [AE α=40 LR=0.0005] Epoch 9 | TrainLoss=0.7307 | ValLoss=1.8863
    [AE α=40 LR=0.0005] Epoch 10 | TrainLoss=0.6780 | ValLoss=1.3713
    [AE α=40 LR=0.0005] Epoch 11 | TrainLoss=0.6698 | ValLoss=1.5297
    [AE α=40 LR=0.0005] Epoch 12 | TrainLoss=0.6419 | ValLoss=1.6259
    [AE α=40 LR=0.0005] Epoch 13 | TrainLoss=0.6152 | ValLoss=1.7947
    [AE α=40 LR=0.0005] Epoch 14 | TrainLoss=0.6027 | ValLoss=1.7034
    [AE α=40 LR=0.0005] Epoch 15 | TrainLoss=0.5871 | ValLoss=0.9726
    [AE α=40 LR=0.0005] Epoch 16 | TrainLoss=0.5863 | ValLoss=1.1436
    [AE α=40 LR=0.0005] Epoch 17 | TrainLoss=0.5471 | ValLoss=1.2466
    [AE α=40 LR=0.0005] Epoch 18 | TrainLoss=0.5319 | ValLoss=0.9867
    [AE α=40 LR=0.0005] Epoch 19 | TrainLoss=0.5212 | ValLoss=1.0815
    [AE α=40 LR=0.0005] Epoch 20 | TrainLoss=0.5267 | ValLoss=2.8300
    [AE α=40 LR=0.0005] Epoch 21 | TrainLoss=0.5005 | ValLoss=2.2175
    [AE α=40 LR=0.0005] Epoch 22 | TrainLoss=0.4926 | ValLoss=2.7066
    [AE α=40 LR=0.0005] Epoch 23 | TrainLoss=0.4891 | ValLoss=1.1067
    [AE α=40 LR=0.0005] Epoch 24 | TrainLoss=0.4720 | ValLoss=1.8988
    [AE α=40 LR=0.0005] Epoch 25 | TrainLoss=0.4561 | ValLoss=2.3855
    [AE α=40 LR=0.0005] Epoch 26 | TrainLoss=0.4711 | ValLoss=1.9357
    [AE α=40 LR=0.0005] Epoch 27 | TrainLoss=0.4544 | ValLoss=1.9066
    [AE α=40 LR=0.0005] Epoch 28 | TrainLoss=0.4534 | ValLoss=1.0229
    [AE α=40 LR=0.0005] Epoch 29 | TrainLoss=0.4326 | ValLoss=1.8750
    [AE α=40 LR=0.0005] Epoch 30 | TrainLoss=0.4278 | ValLoss=1.2172
    Early stopping triggered.
    
    =====================================
    Training AE for α=40, LR=0.001
    =====================================
    [AE α=40 LR=0.001] Epoch 1 | TrainLoss=1.9184 | ValLoss=1.4152
    [AE α=40 LR=0.001] Epoch 2 | TrainLoss=1.1938 | ValLoss=2.0562
    [AE α=40 LR=0.001] Epoch 3 | TrainLoss=1.0389 | ValLoss=1.8972
    [AE α=40 LR=0.001] Epoch 4 | TrainLoss=0.9455 | ValLoss=1.2691
    [AE α=40 LR=0.001] Epoch 5 | TrainLoss=0.9078 | ValLoss=1.4083
    [AE α=40 LR=0.001] Epoch 6 | TrainLoss=0.8307 | ValLoss=1.2757
    [AE α=40 LR=0.001] Epoch 7 | TrainLoss=0.7858 | ValLoss=1.7036
    [AE α=40 LR=0.001] Epoch 8 | TrainLoss=0.7591 | ValLoss=2.9300
    [AE α=40 LR=0.001] Epoch 9 | TrainLoss=0.7107 | ValLoss=1.6623
    [AE α=40 LR=0.001] Epoch 10 | TrainLoss=0.6731 | ValLoss=1.0074
    [AE α=40 LR=0.001] Epoch 11 | TrainLoss=0.6432 | ValLoss=1.5156
    [AE α=40 LR=0.001] Epoch 12 | TrainLoss=0.6232 | ValLoss=1.8130
    [AE α=40 LR=0.001] Epoch 13 | TrainLoss=0.5996 | ValLoss=0.9159
    [AE α=40 LR=0.001] Epoch 14 | TrainLoss=0.5953 | ValLoss=0.7257
    [AE α=40 LR=0.001] Epoch 15 | TrainLoss=0.5733 | ValLoss=1.7363
    [AE α=40 LR=0.001] Epoch 16 | TrainLoss=0.5603 | ValLoss=1.0561
    [AE α=40 LR=0.001] Epoch 17 | TrainLoss=0.5392 | ValLoss=2.3286
    [AE α=40 LR=0.001] Epoch 18 | TrainLoss=0.5208 | ValLoss=1.5740
    [AE α=40 LR=0.001] Epoch 19 | TrainLoss=0.5224 | ValLoss=1.2620
    [AE α=40 LR=0.001] Epoch 20 | TrainLoss=0.5043 | ValLoss=0.8468
    [AE α=40 LR=0.001] Epoch 21 | TrainLoss=0.4886 | ValLoss=0.6072
    [AE α=40 LR=0.001] Epoch 22 | TrainLoss=0.4747 | ValLoss=2.0162
    [AE α=40 LR=0.001] Epoch 23 | TrainLoss=0.4778 | ValLoss=1.5154
    [AE α=40 LR=0.001] Epoch 24 | TrainLoss=0.4670 | ValLoss=1.1693
    [AE α=40 LR=0.001] Epoch 25 | TrainLoss=0.4595 | ValLoss=1.9263
    [AE α=40 LR=0.001] Epoch 26 | TrainLoss=0.4449 | ValLoss=2.5598
    [AE α=40 LR=0.001] Epoch 27 | TrainLoss=0.4527 | ValLoss=1.2372
    [AE α=40 LR=0.001] Epoch 28 | TrainLoss=0.4364 | ValLoss=0.8854
    [AE α=40 LR=0.001] Epoch 29 | TrainLoss=0.4373 | ValLoss=1.5867
    [AE α=40 LR=0.001] Epoch 30 | TrainLoss=0.4139 | ValLoss=1.1718
    [AE α=40 LR=0.001] Epoch 31 | TrainLoss=0.4212 | ValLoss=1.9204
    [AE α=40 LR=0.001] Epoch 32 | TrainLoss=0.3976 | ValLoss=1.8672
    [AE α=40 LR=0.001] Epoch 33 | TrainLoss=0.4012 | ValLoss=1.2673
    [AE α=40 LR=0.001] Epoch 34 | TrainLoss=0.4028 | ValLoss=1.4274
    [AE α=40 LR=0.001] Epoch 35 | TrainLoss=0.3846 | ValLoss=2.7786
    [AE α=40 LR=0.001] Epoch 36 | TrainLoss=0.3867 | ValLoss=2.0150
    Early stopping triggered.
    
    =====================================
    Training AE for α=40, LR=0.002
    =====================================
    [AE α=40 LR=0.002] Epoch 1 | TrainLoss=1.9160 | ValLoss=3.3251
    [AE α=40 LR=0.002] Epoch 2 | TrainLoss=1.2480 | ValLoss=1.6261
    [AE α=40 LR=0.002] Epoch 3 | TrainLoss=1.1028 | ValLoss=1.2688
    [AE α=40 LR=0.002] Epoch 4 | TrainLoss=1.0038 | ValLoss=1.7122
    [AE α=40 LR=0.002] Epoch 5 | TrainLoss=0.9347 | ValLoss=6.4106
    [AE α=40 LR=0.002] Epoch 6 | TrainLoss=0.8741 | ValLoss=1.5272
    [AE α=40 LR=0.002] Epoch 7 | TrainLoss=0.8276 | ValLoss=1.8597
    [AE α=40 LR=0.002] Epoch 8 | TrainLoss=0.8074 | ValLoss=1.1137
    [AE α=40 LR=0.002] Epoch 9 | TrainLoss=0.7777 | ValLoss=1.4696
    [AE α=40 LR=0.002] Epoch 10 | TrainLoss=0.7221 | ValLoss=1.1566
    [AE α=40 LR=0.002] Epoch 11 | TrainLoss=0.7073 | ValLoss=1.2876
    [AE α=40 LR=0.002] Epoch 12 | TrainLoss=0.6737 | ValLoss=1.4701
    [AE α=40 LR=0.002] Epoch 13 | TrainLoss=0.6466 | ValLoss=1.0115
    [AE α=40 LR=0.002] Epoch 14 | TrainLoss=0.6445 | ValLoss=0.9981
    [AE α=40 LR=0.002] Epoch 15 | TrainLoss=0.6138 | ValLoss=1.6789
    [AE α=40 LR=0.002] Epoch 16 | TrainLoss=0.6035 | ValLoss=1.0332
    [AE α=40 LR=0.002] Epoch 17 | TrainLoss=0.5764 | ValLoss=1.1579
    [AE α=40 LR=0.002] Epoch 18 | TrainLoss=0.5626 | ValLoss=1.1570
    [AE α=40 LR=0.002] Epoch 19 | TrainLoss=0.5531 | ValLoss=0.7301
    [AE α=40 LR=0.002] Epoch 20 | TrainLoss=0.5542 | ValLoss=2.8472
    [AE α=40 LR=0.002] Epoch 21 | TrainLoss=0.5221 | ValLoss=1.1922
    [AE α=40 LR=0.002] Epoch 22 | TrainLoss=0.5172 | ValLoss=0.7582
    [AE α=40 LR=0.002] Epoch 23 | TrainLoss=0.4926 | ValLoss=1.1556
    [AE α=40 LR=0.002] Epoch 24 | TrainLoss=0.4850 | ValLoss=0.9735
    [AE α=40 LR=0.002] Epoch 25 | TrainLoss=0.4865 | ValLoss=2.7352
    [AE α=40 LR=0.002] Epoch 26 | TrainLoss=0.4711 | ValLoss=1.0482
    [AE α=40 LR=0.002] Epoch 27 | TrainLoss=0.4655 | ValLoss=0.9404
    [AE α=40 LR=0.002] Epoch 28 | TrainLoss=0.4667 | ValLoss=1.7004
    [AE α=40 LR=0.002] Epoch 29 | TrainLoss=0.4486 | ValLoss=2.3189
    [AE α=40 LR=0.002] Epoch 30 | TrainLoss=0.4522 | ValLoss=0.8649
    [AE α=40 LR=0.002] Epoch 31 | TrainLoss=0.4337 | ValLoss=2.0691
    [AE α=40 LR=0.002] Epoch 32 | TrainLoss=0.4174 | ValLoss=1.8899
    [AE α=40 LR=0.002] Epoch 33 | TrainLoss=0.4247 | ValLoss=2.3538
    [AE α=40 LR=0.002] Epoch 34 | TrainLoss=0.4161 | ValLoss=1.9523
    Early stopping triggered.
    
    =====================================
    Training AE for α=40, LR=0.005
    =====================================
    [AE α=40 LR=0.005] Epoch 1 | TrainLoss=2.1893 | ValLoss=1.5158
    [AE α=40 LR=0.005] Epoch 2 | TrainLoss=1.4168 | ValLoss=1.9756
    [AE α=40 LR=0.005] Epoch 3 | TrainLoss=1.1670 | ValLoss=1.2189
    [AE α=40 LR=0.005] Epoch 4 | TrainLoss=1.0745 | ValLoss=2.0893
    [AE α=40 LR=0.005] Epoch 5 | TrainLoss=1.0088 | ValLoss=1.0017
    [AE α=40 LR=0.005] Epoch 6 | TrainLoss=0.9449 | ValLoss=1.0278
    [AE α=40 LR=0.005] Epoch 7 | TrainLoss=0.9094 | ValLoss=1.1585
    [AE α=40 LR=0.005] Epoch 8 | TrainLoss=0.8837 | ValLoss=1.2978
    [AE α=40 LR=0.005] Epoch 9 | TrainLoss=0.8460 | ValLoss=1.5229
    [AE α=40 LR=0.005] Epoch 10 | TrainLoss=0.8306 | ValLoss=0.9893
    [AE α=40 LR=0.005] Epoch 11 | TrainLoss=0.7836 | ValLoss=0.9216
    [AE α=40 LR=0.005] Epoch 12 | TrainLoss=0.7536 | ValLoss=1.4181
    [AE α=40 LR=0.005] Epoch 13 | TrainLoss=0.7369 | ValLoss=2.0557
    [AE α=40 LR=0.005] Epoch 14 | TrainLoss=0.7157 | ValLoss=2.0726
    [AE α=40 LR=0.005] Epoch 15 | TrainLoss=0.6989 | ValLoss=6.3120
    [AE α=40 LR=0.005] Epoch 16 | TrainLoss=0.6792 | ValLoss=2.1239
    [AE α=40 LR=0.005] Epoch 17 | TrainLoss=0.6437 | ValLoss=0.9317
    [AE α=40 LR=0.005] Epoch 18 | TrainLoss=0.6224 | ValLoss=1.9386
    [AE α=40 LR=0.005] Epoch 19 | TrainLoss=0.6144 | ValLoss=1.1983
    [AE α=40 LR=0.005] Epoch 20 | TrainLoss=0.6231 | ValLoss=0.9523
    [AE α=40 LR=0.005] Epoch 21 | TrainLoss=0.5887 | ValLoss=0.9250
    [AE α=40 LR=0.005] Epoch 22 | TrainLoss=0.5827 | ValLoss=0.8284
    [AE α=40 LR=0.005] Epoch 23 | TrainLoss=0.5775 | ValLoss=1.5252
    [AE α=40 LR=0.005] Epoch 24 | TrainLoss=0.5548 | ValLoss=1.5756
    [AE α=40 LR=0.005] Epoch 25 | TrainLoss=0.5498 | ValLoss=0.8434
    [AE α=40 LR=0.005] Epoch 26 | TrainLoss=0.5411 | ValLoss=1.6357
    [AE α=40 LR=0.005] Epoch 27 | TrainLoss=0.5240 | ValLoss=1.1369
    [AE α=40 LR=0.005] Epoch 28 | TrainLoss=0.5225 | ValLoss=1.0346
    [AE α=40 LR=0.005] Epoch 29 | TrainLoss=0.5208 | ValLoss=1.3435
    [AE α=40 LR=0.005] Epoch 30 | TrainLoss=0.5053 | ValLoss=1.3288
    [AE α=40 LR=0.005] Epoch 31 | TrainLoss=0.4978 | ValLoss=0.7653
    [AE α=40 LR=0.005] Epoch 32 | TrainLoss=0.4916 | ValLoss=1.3698
    [AE α=40 LR=0.005] Epoch 33 | TrainLoss=0.4761 | ValLoss=1.7094
    [AE α=40 LR=0.005] Epoch 34 | TrainLoss=0.4839 | ValLoss=1.1762
    [AE α=40 LR=0.005] Epoch 35 | TrainLoss=0.4715 | ValLoss=0.7374
    [AE α=40 LR=0.005] Epoch 36 | TrainLoss=0.4591 | ValLoss=1.4323
    [AE α=40 LR=0.005] Epoch 37 | TrainLoss=0.4623 | ValLoss=1.8279
    [AE α=40 LR=0.005] Epoch 38 | TrainLoss=0.4728 | ValLoss=0.8745
    [AE α=40 LR=0.005] Epoch 39 | TrainLoss=0.4543 | ValLoss=2.3033
    [AE α=40 LR=0.005] Epoch 40 | TrainLoss=0.4546 | ValLoss=2.0832
    [AE α=40 LR=0.005] Epoch 41 | TrainLoss=0.4406 | ValLoss=0.7127
    [AE α=40 LR=0.005] Epoch 42 | TrainLoss=0.4404 | ValLoss=4.2961
    [AE α=40 LR=0.005] Epoch 43 | TrainLoss=0.4395 | ValLoss=2.3552
    [AE α=40 LR=0.005] Epoch 44 | TrainLoss=0.4504 | ValLoss=1.4532
    [AE α=40 LR=0.005] Epoch 45 | TrainLoss=0.4347 | ValLoss=0.8316
    [AE α=40 LR=0.005] Epoch 46 | TrainLoss=0.4334 | ValLoss=1.3206
    [AE α=40 LR=0.005] Epoch 47 | TrainLoss=0.4171 | ValLoss=1.9214
    [AE α=40 LR=0.005] Epoch 48 | TrainLoss=0.4222 | ValLoss=1.6379
    [AE α=40 LR=0.005] Epoch 49 | TrainLoss=0.4200 | ValLoss=1.6041
    [AE α=40 LR=0.005] Epoch 50 | TrainLoss=0.4041 | ValLoss=1.2296
    [AE α=40 LR=0.005] Epoch 51 | TrainLoss=0.4029 | ValLoss=1.2835
    [AE α=40 LR=0.005] Epoch 52 | TrainLoss=0.3982 | ValLoss=3.2034
    [AE α=40 LR=0.005] Epoch 53 | TrainLoss=0.3917 | ValLoss=0.9248
    [AE α=40 LR=0.005] Epoch 54 | TrainLoss=0.4075 | ValLoss=1.8996
    [AE α=40 LR=0.005] Epoch 55 | TrainLoss=0.4015 | ValLoss=1.0004
    [AE α=40 LR=0.005] Epoch 56 | TrainLoss=0.3972 | ValLoss=1.4870
    Early stopping triggered.
    
    =====================================
    Training AE for α=40, LR=0.01
    =====================================
    [AE α=40 LR=0.01] Epoch 1 | TrainLoss=2.4939 | ValLoss=2.3636
    [AE α=40 LR=0.01] Epoch 2 | TrainLoss=1.7394 | ValLoss=1.6678
    [AE α=40 LR=0.01] Epoch 3 | TrainLoss=1.4216 | ValLoss=3.1000
    [AE α=40 LR=0.01] Epoch 4 | TrainLoss=1.2647 | ValLoss=1.2401
    [AE α=40 LR=0.01] Epoch 5 | TrainLoss=1.1357 | ValLoss=2.2367
    [AE α=40 LR=0.01] Epoch 6 | TrainLoss=1.0805 | ValLoss=1.4819
    [AE α=40 LR=0.01] Epoch 7 | TrainLoss=1.0416 | ValLoss=1.2690
    [AE α=40 LR=0.01] Epoch 8 | TrainLoss=1.0242 | ValLoss=1.7552
    [AE α=40 LR=0.01] Epoch 9 | TrainLoss=0.9779 | ValLoss=1.8638
    [AE α=40 LR=0.01] Epoch 10 | TrainLoss=0.9382 | ValLoss=2.0018
    [AE α=40 LR=0.01] Epoch 11 | TrainLoss=0.9080 | ValLoss=0.9458
    [AE α=40 LR=0.01] Epoch 12 | TrainLoss=0.8723 | ValLoss=1.4794
    [AE α=40 LR=0.01] Epoch 13 | TrainLoss=0.8243 | ValLoss=0.8182
    [AE α=40 LR=0.01] Epoch 14 | TrainLoss=0.7958 | ValLoss=1.7101
    [AE α=40 LR=0.01] Epoch 15 | TrainLoss=0.7908 | ValLoss=0.8509
    [AE α=40 LR=0.01] Epoch 16 | TrainLoss=0.7775 | ValLoss=1.0541
    [AE α=40 LR=0.01] Epoch 17 | TrainLoss=0.7618 | ValLoss=0.9301
    [AE α=40 LR=0.01] Epoch 18 | TrainLoss=0.7347 | ValLoss=1.3692
    [AE α=40 LR=0.01] Epoch 19 | TrainLoss=0.7138 | ValLoss=1.1700
    [AE α=40 LR=0.01] Epoch 20 | TrainLoss=0.6929 | ValLoss=2.0165
    [AE α=40 LR=0.01] Epoch 21 | TrainLoss=0.7008 | ValLoss=1.1253
    [AE α=40 LR=0.01] Epoch 22 | TrainLoss=0.6686 | ValLoss=1.8421
    [AE α=40 LR=0.01] Epoch 23 | TrainLoss=0.6520 | ValLoss=1.2292
    [AE α=40 LR=0.01] Epoch 24 | TrainLoss=0.6609 | ValLoss=1.0560
    [AE α=40 LR=0.01] Epoch 25 | TrainLoss=0.6522 | ValLoss=1.7139
    [AE α=40 LR=0.01] Epoch 26 | TrainLoss=0.6489 | ValLoss=1.0865
    [AE α=40 LR=0.01] Epoch 27 | TrainLoss=0.6115 | ValLoss=0.8916
    [AE α=40 LR=0.01] Epoch 28 | TrainLoss=0.6379 | ValLoss=1.9587
    Early stopping triggered.
    
    =====================================
    Training AE for α=40, LR=0.05
    =====================================
    [AE α=40 LR=0.05] Epoch 1 | TrainLoss=6.0782 | ValLoss=2.2019
    [AE α=40 LR=0.05] Epoch 2 | TrainLoss=2.3059 | ValLoss=2.3350
    [AE α=40 LR=0.05] Epoch 3 | TrainLoss=2.0180 | ValLoss=1.9623
    [AE α=40 LR=0.05] Epoch 4 | TrainLoss=1.8899 | ValLoss=1.9143
    [AE α=40 LR=0.05] Epoch 5 | TrainLoss=1.8255 | ValLoss=4.1288
    [AE α=40 LR=0.05] Epoch 6 | TrainLoss=1.8466 | ValLoss=1.7173
    [AE α=40 LR=0.05] Epoch 7 | TrainLoss=1.6811 | ValLoss=1.6377
    [AE α=40 LR=0.05] Epoch 8 | TrainLoss=1.6980 | ValLoss=1.7605
    [AE α=40 LR=0.05] Epoch 9 | TrainLoss=1.6054 | ValLoss=1.8992
    [AE α=40 LR=0.05] Epoch 10 | TrainLoss=1.6174 | ValLoss=1.5876
    [AE α=40 LR=0.05] Epoch 11 | TrainLoss=1.6276 | ValLoss=3.1671
    [AE α=40 LR=0.05] Epoch 12 | TrainLoss=1.5439 | ValLoss=2.0853
    [AE α=40 LR=0.05] Epoch 13 | TrainLoss=1.6393 | ValLoss=2.0388
    [AE α=40 LR=0.05] Epoch 14 | TrainLoss=1.5201 | ValLoss=1.8187
    [AE α=40 LR=0.05] Epoch 15 | TrainLoss=1.6510 | ValLoss=1.4876
    [AE α=40 LR=0.05] Epoch 16 | TrainLoss=1.4727 | ValLoss=1.2086
    [AE α=40 LR=0.05] Epoch 17 | TrainLoss=1.3920 | ValLoss=1.6867
    [AE α=40 LR=0.05] Epoch 18 | TrainLoss=1.3309 | ValLoss=1.5184
    [AE α=40 LR=0.05] Epoch 19 | TrainLoss=1.2926 | ValLoss=1.6285
    [AE α=40 LR=0.05] Epoch 20 | TrainLoss=1.3202 | ValLoss=2.2112
    [AE α=40 LR=0.05] Epoch 21 | TrainLoss=1.7501 | ValLoss=1.6995
    [AE α=40 LR=0.05] Epoch 22 | TrainLoss=1.6707 | ValLoss=2.0127
    [AE α=40 LR=0.05] Epoch 23 | TrainLoss=1.5269 | ValLoss=1.5363
    [AE α=40 LR=0.05] Epoch 24 | TrainLoss=1.4588 | ValLoss=1.3698
    [AE α=40 LR=0.05] Epoch 25 | TrainLoss=1.3295 | ValLoss=1.4893
    [AE α=40 LR=0.05] Epoch 26 | TrainLoss=1.3026 | ValLoss=1.1807
    [AE α=40 LR=0.05] Epoch 27 | TrainLoss=1.2906 | ValLoss=1.3970
    [AE α=40 LR=0.05] Epoch 28 | TrainLoss=1.2505 | ValLoss=1.3735
    [AE α=40 LR=0.05] Epoch 29 | TrainLoss=1.2986 | ValLoss=1.2832
    [AE α=40 LR=0.05] Epoch 30 | TrainLoss=1.2576 | ValLoss=1.1745
    [AE α=40 LR=0.05] Epoch 31 | TrainLoss=1.2554 | ValLoss=1.3167
    [AE α=40 LR=0.05] Epoch 32 | TrainLoss=1.1920 | ValLoss=1.5105
    [AE α=40 LR=0.05] Epoch 33 | TrainLoss=1.4993 | ValLoss=1.5010
    [AE α=40 LR=0.05] Epoch 34 | TrainLoss=1.3108 | ValLoss=1.4733
    [AE α=40 LR=0.05] Epoch 35 | TrainLoss=1.2745 | ValLoss=1.3607
    [AE α=40 LR=0.05] Epoch 36 | TrainLoss=2.9976 | ValLoss=3.0559
    [AE α=40 LR=0.05] Epoch 37 | TrainLoss=3.1371 | ValLoss=2.7434
    [AE α=40 LR=0.05] Epoch 38 | TrainLoss=2.7403 | ValLoss=2.6536
    [AE α=40 LR=0.05] Epoch 39 | TrainLoss=2.6716 | ValLoss=2.5809
    [AE α=40 LR=0.05] Epoch 40 | TrainLoss=2.6232 | ValLoss=2.5725
    [AE α=40 LR=0.05] Epoch 41 | TrainLoss=2.6513 | ValLoss=2.8376
    [AE α=40 LR=0.05] Epoch 42 | TrainLoss=2.7053 | ValLoss=2.5811
    [AE α=40 LR=0.05] Epoch 43 | TrainLoss=2.6336 | ValLoss=2.5742
    [AE α=40 LR=0.05] Epoch 44 | TrainLoss=2.6141 | ValLoss=2.5550
    [AE α=40 LR=0.05] Epoch 45 | TrainLoss=2.6051 | ValLoss=2.5585
    Early stopping triggered.
    
    =====================================
    Training AE for α=40, LR=0.1
    =====================================
    [AE α=40 LR=0.1] Epoch 1 | TrainLoss=21.0335 | ValLoss=3.2469
    [AE α=40 LR=0.1] Epoch 2 | TrainLoss=3.6596 | ValLoss=3.4568
    [AE α=40 LR=0.1] Epoch 3 | TrainLoss=3.7818 | ValLoss=5.0892
    [AE α=40 LR=0.1] Epoch 4 | TrainLoss=3.6203 | ValLoss=3.3269
    [AE α=40 LR=0.1] Epoch 5 | TrainLoss=3.6194 | ValLoss=3.3489
    [AE α=40 LR=0.1] Epoch 6 | TrainLoss=3.6124 | ValLoss=3.3041
    [AE α=40 LR=0.1] Epoch 7 | TrainLoss=3.6142 | ValLoss=3.2873
    [AE α=40 LR=0.1] Epoch 8 | TrainLoss=3.6130 | ValLoss=3.3895
    [AE α=40 LR=0.1] Epoch 9 | TrainLoss=3.6213 | ValLoss=3.3035
    [AE α=40 LR=0.1] Epoch 10 | TrainLoss=3.6205 | ValLoss=3.3171
    [AE α=40 LR=0.1] Epoch 11 | TrainLoss=3.6170 | ValLoss=3.3311
    [AE α=40 LR=0.1] Epoch 12 | TrainLoss=3.6126 | ValLoss=3.3526
    [AE α=40 LR=0.1] Epoch 13 | TrainLoss=3.5568 | ValLoss=3.4510
    [AE α=40 LR=0.1] Epoch 14 | TrainLoss=3.5313 | ValLoss=3.3884
    [AE α=40 LR=0.1] Epoch 15 | TrainLoss=3.5313 | ValLoss=3.3758
    [AE α=40 LR=0.1] Epoch 16 | TrainLoss=3.5308 | ValLoss=3.4296
    Early stopping triggered.
       α=35, LR=0.005
       BEST ValLoss=0.5397
       Saved in: models_best/AE_GLOBAL_BEST.pt
    
    

## Training Performance Overview


```python
# Load JSON
with open("models_best/validation_losses.json", "r") as f:
    results_json = json.load(f)

heatmap = np.zeros((len(alpha_values), len(lr_values)))

for i, a in enumerate(alpha_values):
    for j, lr in enumerate(lr_values):
        key = f"alpha={a}, lr={lr}"
        heatmap[i, j] = results_json[key]

# Plot heatmap
plt.figure(figsize=(10, 6))
plt.imshow(heatmap, cmap="viridis", aspect="auto")
plt.colorbar(label="Validation Loss")
plt.xticks(range(len(lr_values)), lr_values, rotation=45)
plt.yticks(range(len(alpha_values)), alpha_values)
plt.xlabel("Learning Rate")
plt.ylabel("Alpha")
plt.title("Validation Loss Heatmap")
plt.tight_layout()
plt.show()
```


    
![png](output_54_0.png)
    


The supervised autoencoder performs best with α = 35 and a learning rate of 0.005, achieving a minimum validation loss of 0.5397.

This result shows that all α values in the tested range (20–40) lead to comparable performance, as long as the learning rate lies in the stable region. The heatmap indicates that the learning rate has a much stronger influence on validation loss than the exact choice of α.

The validation-loss heatmap further highlights how the model behaves across different hyperparameter combinations:

* Low LR: Very small learning rates ($10^{-4}, 2 \cdot 10^{-4}$) slow training significantly.
* High LR: Larger learning rates (0.05, 0.1) cause divergence and instability.
* Effective Training: Effective training occurs only within a narrow interval, roughly between $10^{-3}$ and $5 \cdot 10^{-3}$.

Within this stable region, models with higher α values consistently perform better, confirming that giving more weight to the reconstruction term leads to a more well-balanced and effective supervised autoencoder.


```python
plt.figure(figsize=(10, 5))
plt.plot(best_train_losses, label="Train Loss")
plt.plot(best_val_losses_curve, label="Validation Loss")
plt.title(f"Loss Curves (Best AE: α={best_alpha}, lr={best_lr})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```


    
![png](output_56_0.png)
    


The training and validation loss curves for the best model (α = 35, LR = 0.005) reveal a mixed but interpretable behavior. On the one hand, the training loss decreases smoothly and consistently, indicating that the chosen learning rate allows the optimizer to make steady progress without instability. On the other hand, the validation loss exhibits considerable variability across epochs, which is not unexpected in a supervised autoencoder but does suggest that the model’s generalization behavior is less stable than its training dynamics.

This oscillatory pattern reflects the dual nature of the objective: the reconstruction and classification terms respond differently to changes in the latent representation and improvements in one component do not always guarantee improvements in the other. Despite this variability, the validation loss repeatedly returns to relatively low values and eventually achieves the lowest minimum among all tested hyperparameter combinations. However, the fluctuations indicate that the model remains sensitive to initialization and stochasticity during training.

Overall, the combination α = 35 and LR = 0.005 appears to offer the best compromise found in the grid search. It balances reconstruction and classification sufficiently well to outperform other configurations, but the instability in validation loss suggests that the supervised autoencoder does not fully converge to a uniformly smooth solution. The resulting model is functional and reasonably well-balanced, though not perfectly stable, consistent with the inherent difficulty of jointly optimizing reconstruction and classification within a single latent space.

## Extraction of Latent Features

Before training the classifier, it is necessary to convert each image into its corresponding latent representation. The extracted latent vectors form a new dataset on which the classifier is trained. This allows evaluating the discriminative quality of the latent space learned by the autoencoder, independently of the raw input images.

To accomplish this, I define the function `extract_features`, which will be applied after the encoder of the trained autoencoder has been frozen.

The function takes a dataloader and an encoder as input and forwards each batch of images through the encoder to obtain the latent vectors. These vectors are progressively collected into a feature matrix $X$, while the associated labels are stored in a vector $y$. Once all batches have been processed, the function concatenates the results into two complete tensors.

During this phase, the encoder will operate in evaluation mode with gradient computation disabled, ensuring that the extraction is efficient and that its parameters remain unchanged.

The resulting dataset $(X, y)$ will then be used to train the external MLP classifier, which allows evaluating the discriminative quality of the latent space learned for each $(\alpha, \beta)$ configuration.


```python
def extract_features(loader, encoder):
    X_list, y_list = [], []
    encoder.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            z = encoder(imgs)         
            X_list.append(z.cpu())
            y_list.append(labels.cpu())

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    return X, y
```

## MLP Classifier

To assess the discriminative quality of the latent representations produced by the autoencoder, I use a Multilayer Perceptron (MLP) as an external classifier. Once the encoder has been frozen and all latent vectors have been extracted, this classifier is trained exclusively on those embeddings. This setup isolates the effect of the latent space itself: the MLP’s performance directly reflects how well each autoencoder configuration organizes the data into class-separable structures, independent of the decoder or reconstruction objective.

The MLP takes as input a latent vector of size input_dim, generated by the encoder, and outputs num_classes logits, which are used to compute the cross-entropy loss during training.

#### 1. Fully Connected Layers (Linear Layers)
The network is intentionally small and composed of three fully connected layers:

$$
\text{Input} \to 128 \text{ units} \to 64 \text{ units} \to \text{num\_classes}
$$

Each linear layer performs an affine transformation and allows the classifier to combine latent features in increasingly abstract ways. Because the autoencoder already compresses the input into a highly informative latent representation, a compact architecture is sufficient; the classifier does not need to be large to achieve meaningful performance.

#### 2. ReLU Activations

#### 3. Batch Normalization

#### 4. Dropout

A dropout layer with probability 0.3 follows the first hidden layer. By randomly deactivating a portion of neurons during training, dropout discourages co-adaptation of features and strengthens generalization. This is particularly valuable in this context, as the classifier operates on pre-extracted features and must avoid overfitting to the latent dataset.

#### 5. Final Linear Layer
The final linear layer maps the 64-dimensional hidden representation to the number of classes, producing a vector of logits:

$$
z = Wh + b
$$

These logits are then passed through the cross-entropy loss (which implicitly applies softmax) to compute class probabilities during training.

Overall, the MLP is deliberately lightweight and regularized: its purpose is not to outperform the autoencoder, but to reveal the discriminative power of the latent space learned under each hyperparameter configuration.


```python
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3), 

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
```

## Latent-Space MLP Classifier Training and Learning Rate Grid Search

Once the best-performing autoencoder configuration had been identified, the next step was to evaluate how well its latent space supports classification. To do this, the encoder is isolated from the rest of the network, frozen, and used solely as a feature extractor. This ensures that the quality of the latent representations is assessed independently of both the decoder and the reconstruction objective.

The frozen encoder is applied to the training, validation and test sets, producing three matrices of latent vectors alongside their corresponding label vectors. These embeddings replace the original images and serve as input to an external MLP classifier. Because the encoder is fixed, all learning now occurs exclusively within the classifier; this setup makes the learning rate the primary hyperparameter governing optimization dynamics.

To make training tractable and efficient, the extracted latent features are wrapped into new DataLoaders. At this point, the classification task becomes analogous to training on tabular data: each sample is a single latent vector, and the classifier must learn to map this representation into the correct category.

A grid search over multiple learning rates is then conducted to evaluate how sensitive the classifier is to this hyperparameter when operating on the autoencoder’s latent space. Training is performed using the Adam optimizer with weight decay, which provides both adaptive step sizes and mild regularization. Weight decay is particularly useful in this context because the classifier is small and trained on fixed embeddings, making it prone to overfitting unless some form of regularization is applied.

* If the learning rate is too small, the optimizer updates the classifier weights very slowly, preventing the model from fully adapting to the geometry of the latent space. Training may converge, but only toward a weak decision boundary.
* A learning rate that is too large leads to unstable behavior, especially in combination with Batch Normalization layers which amplify sensitivity to overly aggressive updates. At such learning rates, the classifier often becomes unstable and may oscillate or fail to converge.

Between these extremes lies a narrow interval in which the learning rate is appropriately matched to the structure of the latent space. Within this range, the classifier converges efficiently, producing high validation accuracy and demonstrating that the autoencoder has learned a sufficiently discriminative latent representation.

For each learning rate tested, the model checkpoint achieving the highest validation accuracy is stored. After completing the grid search, the best classifier across all configurations is identified and saved, together with its validation and test accuracies. This final model represents the optimal way to leverage the latent space learned by the autoencoder.


```python
os.makedirs("mlp_best", exist_ok=True)

# Load best AE
best_path = "models_best/AE_GLOBAL_BEST.pt"
latent_dim = 64

best_ae = SupervisedAutoencoder(latent_dim=latent_dim, num_classes=10).to(device)
best_ae.load_state_dict(torch.load(best_path, map_location=device))

# Freeze encoder
for p in best_ae.enc.parameters():
    p.requires_grad = False
best_ae.enc.eval()

# Feature extraction
X_train, y_train = extract_features(train_loader, best_ae.enc)
X_val,   y_val   = extract_features(val_loader,   best_ae.enc)
X_test,  y_test  = extract_features(test_loader,  best_ae.enc)

train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_dl   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=64, shuffle=False)
test_dl  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=64, shuffle=False)

lr_values = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
num_epochs = 30

global_best_val = 0
global_best_test = 0
global_best_lr = None
global_best_state = None

for lr in lr_values:
    print(f"\n=====================================")
    print(f"   Training MLP with LR = {lr}")
    print("=====================================")

    clf = MLP(input_dim=latent_dim, num_classes=10).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)

    criterion = torch.nn.CrossEntropyLoss()

    train_acc_curve, val_acc_curve = [], []
    train_loss_curve, val_loss_curve = [], []
    best_val_acc = 0

    #Train
    for e in range(num_epochs):
        clf.train()
        correct_train, total_train = 0, 0
        train_loss_sum = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            logits = clf(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            correct_train += (logits.argmax(1) == yb).sum().item()
            total_train += xb.size(0)

        train_acc = correct_train / total_train
        train_loss = train_loss_sum / total_train

        # Validation
        clf.eval()
        correct_val, total_val = 0, 0
        val_loss_sum = 0

        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = clf(xb)
                loss = criterion(logits, yb)

                val_loss_sum += loss.item() * xb.size(0)
                correct_val += (logits.argmax(1) == yb).sum().item()
                total_val += xb.size(0)

        val_acc = correct_val / total_val
        val_loss = val_loss_sum / total_val

        train_acc_curve.append(train_acc)
        val_acc_curve.append(val_acc)
        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)

        print(f"Epoch {e+1}/{num_epochs} | TrainAcc={train_acc:.3f} ValAcc={val_acc:.3f}")

        # Best val
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_lr = clf.state_dict().copy()

    # Test
    clf.load_state_dict(best_state_lr)
    clf.eval()

    correct_test, total_test = 0, 0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = clf(xb).argmax(1)
            correct_test += (preds == yb).sum().item()
            total_test += xb.size(0)

    test_acc = correct_test / total_test

    # Save plots
    if best_val_acc > global_best_val:
        global_best_val = best_val_acc
        global_best_test = test_acc
        global_best_lr = lr
        global_best_state = best_state_lr.copy()

        plt.figure(figsize=(8,5))
        plt.plot(train_acc_curve, label="Train")
        plt.plot(val_acc_curve, label="Validation")
        plt.title(f"Accuracy Curve – Best LR = {lr}")
        plt.legend()
        plt.grid()
        plt.savefig("mlp_best/best_lr_accuracy_curve.png")
        plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(train_loss_curve, label="Train")
        plt.plot(val_loss_curve, label="Validation")
        plt.title(f"Loss Curve – Best LR = {lr}")
        plt.legend()
        plt.grid()
        plt.savefig("mlp_best/best_lr_loss_curve.png")
        plt.close()

# Save best MLP
torch.save(global_best_state, "mlp_best/MLP_GLOBAL_BEST.pt")

print("\n--------------------------------------")
print("Best MLP")
print("--------------------------------------")
print(f"Best LR             = {global_best_lr}")
print(f"Best Validation Acc = {global_best_val:.4f}")
print(f"Test Acc (finale)   = {global_best_test:.4f}")
```

    C:\Users\Matti\AppData\Local\Temp\ipykernel_7444\1534089791.py:8: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      best_ae.load_state_dict(torch.load(best_path, map_location=device))
    

    
    =====================================
       Training MLP with LR = 1e-06
    =====================================
    Epoch 1/30 | TrainAcc=0.115 ValAcc=0.178
    Epoch 2/30 | TrainAcc=0.125 ValAcc=0.190
    Epoch 3/30 | TrainAcc=0.130 ValAcc=0.197
    Epoch 4/30 | TrainAcc=0.141 ValAcc=0.208
    Epoch 5/30 | TrainAcc=0.146 ValAcc=0.213
    Epoch 6/30 | TrainAcc=0.164 ValAcc=0.225
    Epoch 7/30 | TrainAcc=0.166 ValAcc=0.232
    Epoch 8/30 | TrainAcc=0.177 ValAcc=0.247
    Epoch 9/30 | TrainAcc=0.194 ValAcc=0.253
    Epoch 10/30 | TrainAcc=0.198 ValAcc=0.257
    Epoch 11/30 | TrainAcc=0.215 ValAcc=0.269
    Epoch 12/30 | TrainAcc=0.221 ValAcc=0.284
    Epoch 13/30 | TrainAcc=0.237 ValAcc=0.301
    Epoch 14/30 | TrainAcc=0.249 ValAcc=0.313
    Epoch 15/30 | TrainAcc=0.262 ValAcc=0.325
    Epoch 16/30 | TrainAcc=0.280 ValAcc=0.335
    Epoch 17/30 | TrainAcc=0.292 ValAcc=0.358
    Epoch 18/30 | TrainAcc=0.307 ValAcc=0.382
    Epoch 19/30 | TrainAcc=0.320 ValAcc=0.396
    Epoch 20/30 | TrainAcc=0.334 ValAcc=0.432
    Epoch 21/30 | TrainAcc=0.358 ValAcc=0.456
    Epoch 22/30 | TrainAcc=0.366 ValAcc=0.471
    Epoch 23/30 | TrainAcc=0.379 ValAcc=0.486
    Epoch 24/30 | TrainAcc=0.395 ValAcc=0.517
    Epoch 25/30 | TrainAcc=0.419 ValAcc=0.527
    Epoch 26/30 | TrainAcc=0.430 ValAcc=0.537
    Epoch 27/30 | TrainAcc=0.442 ValAcc=0.578
    Epoch 28/30 | TrainAcc=0.457 ValAcc=0.583
    Epoch 29/30 | TrainAcc=0.475 ValAcc=0.573
    Epoch 30/30 | TrainAcc=0.482 ValAcc=0.596
    
    =====================================
       Training MLP with LR = 5e-06
    =====================================
    Epoch 1/30 | TrainAcc=0.194 ValAcc=0.311
    Epoch 2/30 | TrainAcc=0.248 ValAcc=0.364
    Epoch 3/30 | TrainAcc=0.305 ValAcc=0.428
    Epoch 4/30 | TrainAcc=0.363 ValAcc=0.496
    Epoch 5/30 | TrainAcc=0.424 ValAcc=0.535
    Epoch 6/30 | TrainAcc=0.459 ValAcc=0.579
    Epoch 7/30 | TrainAcc=0.499 ValAcc=0.604
    Epoch 8/30 | TrainAcc=0.542 ValAcc=0.624
    Epoch 9/30 | TrainAcc=0.568 ValAcc=0.643
    Epoch 10/30 | TrainAcc=0.590 ValAcc=0.658
    Epoch 11/30 | TrainAcc=0.611 ValAcc=0.669
    Epoch 12/30 | TrainAcc=0.634 ValAcc=0.680
    Epoch 13/30 | TrainAcc=0.654 ValAcc=0.691
    Epoch 14/30 | TrainAcc=0.665 ValAcc=0.698
    Epoch 15/30 | TrainAcc=0.676 ValAcc=0.705
    Epoch 16/30 | TrainAcc=0.689 ValAcc=0.713
    Epoch 17/30 | TrainAcc=0.705 ValAcc=0.716
    Epoch 18/30 | TrainAcc=0.712 ValAcc=0.724
    Epoch 19/30 | TrainAcc=0.724 ValAcc=0.728
    Epoch 20/30 | TrainAcc=0.730 ValAcc=0.730
    Epoch 21/30 | TrainAcc=0.736 ValAcc=0.733
    Epoch 22/30 | TrainAcc=0.743 ValAcc=0.733
    Epoch 23/30 | TrainAcc=0.750 ValAcc=0.738
    Epoch 24/30 | TrainAcc=0.751 ValAcc=0.743
    Epoch 25/30 | TrainAcc=0.762 ValAcc=0.738
    Epoch 26/30 | TrainAcc=0.766 ValAcc=0.742
    Epoch 27/30 | TrainAcc=0.772 ValAcc=0.740
    Epoch 28/30 | TrainAcc=0.774 ValAcc=0.743
    Epoch 29/30 | TrainAcc=0.777 ValAcc=0.749
    Epoch 30/30 | TrainAcc=0.781 ValAcc=0.739
    
    =====================================
       Training MLP with LR = 1e-05
    =====================================
    Epoch 1/30 | TrainAcc=0.119 ValAcc=0.238
    Epoch 2/30 | TrainAcc=0.228 ValAcc=0.374
    Epoch 3/30 | TrainAcc=0.339 ValAcc=0.453
    Epoch 4/30 | TrainAcc=0.444 ValAcc=0.501
    Epoch 5/30 | TrainAcc=0.518 ValAcc=0.577
    Epoch 6/30 | TrainAcc=0.568 ValAcc=0.634
    Epoch 7/30 | TrainAcc=0.607 ValAcc=0.646
    Epoch 8/30 | TrainAcc=0.644 ValAcc=0.661
    Epoch 9/30 | TrainAcc=0.665 ValAcc=0.681
    Epoch 10/30 | TrainAcc=0.681 ValAcc=0.697
    Epoch 11/30 | TrainAcc=0.704 ValAcc=0.702
    Epoch 12/30 | TrainAcc=0.719 ValAcc=0.709
    Epoch 13/30 | TrainAcc=0.730 ValAcc=0.714
    Epoch 14/30 | TrainAcc=0.740 ValAcc=0.713
    Epoch 15/30 | TrainAcc=0.746 ValAcc=0.733
    Epoch 16/30 | TrainAcc=0.752 ValAcc=0.730
    Epoch 17/30 | TrainAcc=0.765 ValAcc=0.722
    Epoch 18/30 | TrainAcc=0.772 ValAcc=0.729
    Epoch 19/30 | TrainAcc=0.779 ValAcc=0.733
    Epoch 20/30 | TrainAcc=0.785 ValAcc=0.736
    Epoch 21/30 | TrainAcc=0.789 ValAcc=0.736
    Epoch 22/30 | TrainAcc=0.793 ValAcc=0.739
    Epoch 23/30 | TrainAcc=0.798 ValAcc=0.735
    Epoch 24/30 | TrainAcc=0.804 ValAcc=0.740
    Epoch 25/30 | TrainAcc=0.807 ValAcc=0.743
    Epoch 26/30 | TrainAcc=0.812 ValAcc=0.748
    Epoch 27/30 | TrainAcc=0.814 ValAcc=0.748
    Epoch 28/30 | TrainAcc=0.819 ValAcc=0.748
    Epoch 29/30 | TrainAcc=0.817 ValAcc=0.745
    Epoch 30/30 | TrainAcc=0.825 ValAcc=0.756
    
    =====================================
       Training MLP with LR = 5e-05
    =====================================
    Epoch 1/30 | TrainAcc=0.354 ValAcc=0.585
    Epoch 2/30 | TrainAcc=0.672 ValAcc=0.726
    Epoch 3/30 | TrainAcc=0.755 ValAcc=0.748
    Epoch 4/30 | TrainAcc=0.799 ValAcc=0.753
    Epoch 5/30 | TrainAcc=0.815 ValAcc=0.747
    Epoch 6/30 | TrainAcc=0.825 ValAcc=0.747
    Epoch 7/30 | TrainAcc=0.838 ValAcc=0.748
    Epoch 8/30 | TrainAcc=0.842 ValAcc=0.747
    Epoch 9/30 | TrainAcc=0.853 ValAcc=0.747
    Epoch 10/30 | TrainAcc=0.854 ValAcc=0.753
    Epoch 11/30 | TrainAcc=0.859 ValAcc=0.745
    Epoch 12/30 | TrainAcc=0.863 ValAcc=0.747
    Epoch 13/30 | TrainAcc=0.868 ValAcc=0.742
    Epoch 14/30 | TrainAcc=0.869 ValAcc=0.745
    Epoch 15/30 | TrainAcc=0.872 ValAcc=0.744
    Epoch 16/30 | TrainAcc=0.874 ValAcc=0.738
    Epoch 17/30 | TrainAcc=0.874 ValAcc=0.741
    Epoch 18/30 | TrainAcc=0.876 ValAcc=0.739
    Epoch 19/30 | TrainAcc=0.880 ValAcc=0.734
    Epoch 20/30 | TrainAcc=0.881 ValAcc=0.737
    Epoch 21/30 | TrainAcc=0.878 ValAcc=0.742
    Epoch 22/30 | TrainAcc=0.879 ValAcc=0.741
    Epoch 23/30 | TrainAcc=0.881 ValAcc=0.733
    Epoch 24/30 | TrainAcc=0.887 ValAcc=0.730
    Epoch 25/30 | TrainAcc=0.883 ValAcc=0.732
    Epoch 26/30 | TrainAcc=0.886 ValAcc=0.726
    Epoch 27/30 | TrainAcc=0.885 ValAcc=0.732
    Epoch 28/30 | TrainAcc=0.885 ValAcc=0.733
    Epoch 29/30 | TrainAcc=0.885 ValAcc=0.725
    Epoch 30/30 | TrainAcc=0.888 ValAcc=0.731
    
    =====================================
       Training MLP with LR = 0.0001
    =====================================
    Epoch 1/30 | TrainAcc=0.539 ValAcc=0.694
    Epoch 2/30 | TrainAcc=0.762 ValAcc=0.747
    Epoch 3/30 | TrainAcc=0.813 ValAcc=0.768
    Epoch 4/30 | TrainAcc=0.840 ValAcc=0.765
    Epoch 5/30 | TrainAcc=0.847 ValAcc=0.770
    Epoch 6/30 | TrainAcc=0.862 ValAcc=0.762
    Epoch 7/30 | TrainAcc=0.864 ValAcc=0.748
    Epoch 8/30 | TrainAcc=0.867 ValAcc=0.743
    Epoch 9/30 | TrainAcc=0.875 ValAcc=0.752
    Epoch 10/30 | TrainAcc=0.876 ValAcc=0.745
    Epoch 11/30 | TrainAcc=0.878 ValAcc=0.741
    Epoch 12/30 | TrainAcc=0.880 ValAcc=0.733
    Epoch 13/30 | TrainAcc=0.885 ValAcc=0.746
    Epoch 14/30 | TrainAcc=0.885 ValAcc=0.741
    Epoch 15/30 | TrainAcc=0.887 ValAcc=0.742
    Epoch 16/30 | TrainAcc=0.889 ValAcc=0.738
    Epoch 17/30 | TrainAcc=0.890 ValAcc=0.735
    Epoch 18/30 | TrainAcc=0.892 ValAcc=0.738
    Epoch 19/30 | TrainAcc=0.894 ValAcc=0.732
    Epoch 20/30 | TrainAcc=0.892 ValAcc=0.732
    Epoch 21/30 | TrainAcc=0.897 ValAcc=0.738
    Epoch 22/30 | TrainAcc=0.894 ValAcc=0.731
    Epoch 23/30 | TrainAcc=0.898 ValAcc=0.737
    Epoch 24/30 | TrainAcc=0.900 ValAcc=0.726
    Epoch 25/30 | TrainAcc=0.899 ValAcc=0.740
    Epoch 26/30 | TrainAcc=0.898 ValAcc=0.726
    Epoch 27/30 | TrainAcc=0.900 ValAcc=0.726
    Epoch 28/30 | TrainAcc=0.899 ValAcc=0.732
    Epoch 29/30 | TrainAcc=0.903 ValAcc=0.727
    Epoch 30/30 | TrainAcc=0.902 ValAcc=0.731
    
    =====================================
       Training MLP with LR = 0.0005
    =====================================
    Epoch 1/30 | TrainAcc=0.754 ValAcc=0.751
    Epoch 2/30 | TrainAcc=0.862 ValAcc=0.732
    Epoch 3/30 | TrainAcc=0.880 ValAcc=0.732
    Epoch 4/30 | TrainAcc=0.885 ValAcc=0.722
    Epoch 5/30 | TrainAcc=0.893 ValAcc=0.728
    Epoch 6/30 | TrainAcc=0.893 ValAcc=0.726
    Epoch 7/30 | TrainAcc=0.894 ValAcc=0.727
    Epoch 8/30 | TrainAcc=0.902 ValAcc=0.722
    Epoch 9/30 | TrainAcc=0.901 ValAcc=0.718
    Epoch 10/30 | TrainAcc=0.900 ValAcc=0.725
    Epoch 11/30 | TrainAcc=0.907 ValAcc=0.726
    Epoch 12/30 | TrainAcc=0.909 ValAcc=0.725
    Epoch 13/30 | TrainAcc=0.911 ValAcc=0.716
    Epoch 14/30 | TrainAcc=0.909 ValAcc=0.721
    Epoch 15/30 | TrainAcc=0.910 ValAcc=0.718
    Epoch 16/30 | TrainAcc=0.913 ValAcc=0.734
    Epoch 17/30 | TrainAcc=0.911 ValAcc=0.711
    Epoch 18/30 | TrainAcc=0.913 ValAcc=0.725
    Epoch 19/30 | TrainAcc=0.911 ValAcc=0.718
    Epoch 20/30 | TrainAcc=0.914 ValAcc=0.722
    Epoch 21/30 | TrainAcc=0.914 ValAcc=0.720
    Epoch 22/30 | TrainAcc=0.913 ValAcc=0.730
    Epoch 23/30 | TrainAcc=0.915 ValAcc=0.724
    Epoch 24/30 | TrainAcc=0.918 ValAcc=0.726
    Epoch 25/30 | TrainAcc=0.919 ValAcc=0.716
    Epoch 26/30 | TrainAcc=0.918 ValAcc=0.730
    Epoch 27/30 | TrainAcc=0.922 ValAcc=0.725
    Epoch 28/30 | TrainAcc=0.920 ValAcc=0.731
    Epoch 29/30 | TrainAcc=0.922 ValAcc=0.714
    Epoch 30/30 | TrainAcc=0.920 ValAcc=0.726
    
    =====================================
       Training MLP with LR = 0.001
    =====================================
    Epoch 1/30 | TrainAcc=0.798 ValAcc=0.735
    Epoch 2/30 | TrainAcc=0.875 ValAcc=0.730
    Epoch 3/30 | TrainAcc=0.891 ValAcc=0.728
    Epoch 4/30 | TrainAcc=0.889 ValAcc=0.726
    Epoch 5/30 | TrainAcc=0.896 ValAcc=0.723
    Epoch 6/30 | TrainAcc=0.900 ValAcc=0.717
    Epoch 7/30 | TrainAcc=0.900 ValAcc=0.729
    Epoch 8/30 | TrainAcc=0.903 ValAcc=0.730
    Epoch 9/30 | TrainAcc=0.906 ValAcc=0.724
    Epoch 10/30 | TrainAcc=0.907 ValAcc=0.724
    Epoch 11/30 | TrainAcc=0.910 ValAcc=0.727
    Epoch 12/30 | TrainAcc=0.908 ValAcc=0.733
    Epoch 13/30 | TrainAcc=0.909 ValAcc=0.735
    Epoch 14/30 | TrainAcc=0.914 ValAcc=0.735
    Epoch 15/30 | TrainAcc=0.914 ValAcc=0.727
    Epoch 16/30 | TrainAcc=0.915 ValAcc=0.726
    Epoch 17/30 | TrainAcc=0.913 ValAcc=0.722
    Epoch 18/30 | TrainAcc=0.916 ValAcc=0.720
    Epoch 19/30 | TrainAcc=0.919 ValAcc=0.721
    Epoch 20/30 | TrainAcc=0.916 ValAcc=0.718
    Epoch 21/30 | TrainAcc=0.918 ValAcc=0.708
    Epoch 22/30 | TrainAcc=0.922 ValAcc=0.719
    Epoch 23/30 | TrainAcc=0.916 ValAcc=0.728
    Epoch 24/30 | TrainAcc=0.915 ValAcc=0.710
    Epoch 25/30 | TrainAcc=0.919 ValAcc=0.728
    Epoch 26/30 | TrainAcc=0.919 ValAcc=0.725
    Epoch 27/30 | TrainAcc=0.925 ValAcc=0.734
    Epoch 28/30 | TrainAcc=0.922 ValAcc=0.725
    Epoch 29/30 | TrainAcc=0.923 ValAcc=0.722
    Epoch 30/30 | TrainAcc=0.922 ValAcc=0.722
    
    =====================================
       Training MLP with LR = 0.005
    =====================================
    Epoch 1/30 | TrainAcc=0.837 ValAcc=0.729
    Epoch 2/30 | TrainAcc=0.878 ValAcc=0.722
    Epoch 3/30 | TrainAcc=0.888 ValAcc=0.737
    Epoch 4/30 | TrainAcc=0.890 ValAcc=0.728
    Epoch 5/30 | TrainAcc=0.896 ValAcc=0.728
    Epoch 6/30 | TrainAcc=0.898 ValAcc=0.717
    Epoch 7/30 | TrainAcc=0.901 ValAcc=0.718
    Epoch 8/30 | TrainAcc=0.895 ValAcc=0.737
    Epoch 9/30 | TrainAcc=0.903 ValAcc=0.737
    Epoch 10/30 | TrainAcc=0.898 ValAcc=0.731
    Epoch 11/30 | TrainAcc=0.905 ValAcc=0.718
    Epoch 12/30 | TrainAcc=0.905 ValAcc=0.719
    Epoch 13/30 | TrainAcc=0.904 ValAcc=0.720
    Epoch 14/30 | TrainAcc=0.907 ValAcc=0.725
    Epoch 15/30 | TrainAcc=0.911 ValAcc=0.715
    Epoch 16/30 | TrainAcc=0.910 ValAcc=0.733
    Epoch 17/30 | TrainAcc=0.910 ValAcc=0.732
    Epoch 18/30 | TrainAcc=0.908 ValAcc=0.722
    Epoch 19/30 | TrainAcc=0.911 ValAcc=0.720
    Epoch 20/30 | TrainAcc=0.908 ValAcc=0.724
    Epoch 21/30 | TrainAcc=0.911 ValAcc=0.712
    Epoch 22/30 | TrainAcc=0.911 ValAcc=0.722
    Epoch 23/30 | TrainAcc=0.913 ValAcc=0.727
    Epoch 24/30 | TrainAcc=0.910 ValAcc=0.714
    Epoch 25/30 | TrainAcc=0.909 ValAcc=0.733
    Epoch 26/30 | TrainAcc=0.910 ValAcc=0.728
    Epoch 27/30 | TrainAcc=0.911 ValAcc=0.725
    Epoch 28/30 | TrainAcc=0.915 ValAcc=0.712
    Epoch 29/30 | TrainAcc=0.912 ValAcc=0.732
    Epoch 30/30 | TrainAcc=0.914 ValAcc=0.724
    
    =====================================
       Training MLP with LR = 0.01
    =====================================
    Epoch 1/30 | TrainAcc=0.842 ValAcc=0.738
    Epoch 2/30 | TrainAcc=0.874 ValAcc=0.729
    Epoch 3/30 | TrainAcc=0.877 ValAcc=0.740
    Epoch 4/30 | TrainAcc=0.886 ValAcc=0.722
    Epoch 5/30 | TrainAcc=0.888 ValAcc=0.734
    Epoch 6/30 | TrainAcc=0.887 ValAcc=0.744
    Epoch 7/30 | TrainAcc=0.893 ValAcc=0.715
    Epoch 8/30 | TrainAcc=0.892 ValAcc=0.712
    Epoch 9/30 | TrainAcc=0.894 ValAcc=0.715
    Epoch 10/30 | TrainAcc=0.896 ValAcc=0.736
    Epoch 11/30 | TrainAcc=0.900 ValAcc=0.724
    Epoch 12/30 | TrainAcc=0.895 ValAcc=0.719
    Epoch 13/30 | TrainAcc=0.897 ValAcc=0.741
    Epoch 14/30 | TrainAcc=0.899 ValAcc=0.714
    Epoch 15/30 | TrainAcc=0.897 ValAcc=0.725
    Epoch 16/30 | TrainAcc=0.899 ValAcc=0.739
    Epoch 17/30 | TrainAcc=0.899 ValAcc=0.725
    Epoch 18/30 | TrainAcc=0.901 ValAcc=0.725
    Epoch 19/30 | TrainAcc=0.900 ValAcc=0.712
    Epoch 20/30 | TrainAcc=0.897 ValAcc=0.716
    Epoch 21/30 | TrainAcc=0.900 ValAcc=0.721
    Epoch 22/30 | TrainAcc=0.898 ValAcc=0.721
    Epoch 23/30 | TrainAcc=0.901 ValAcc=0.727
    Epoch 24/30 | TrainAcc=0.899 ValAcc=0.715
    Epoch 25/30 | TrainAcc=0.903 ValAcc=0.708
    Epoch 26/30 | TrainAcc=0.899 ValAcc=0.713
    Epoch 27/30 | TrainAcc=0.897 ValAcc=0.711
    Epoch 28/30 | TrainAcc=0.899 ValAcc=0.736
    Epoch 29/30 | TrainAcc=0.895 ValAcc=0.751
    Epoch 30/30 | TrainAcc=0.906 ValAcc=0.720
    
    =====================================
       Training MLP with LR = 0.05
    =====================================
    Epoch 1/30 | TrainAcc=0.823 ValAcc=0.743
    Epoch 2/30 | TrainAcc=0.849 ValAcc=0.652
    Epoch 3/30 | TrainAcc=0.855 ValAcc=0.699
    Epoch 4/30 | TrainAcc=0.855 ValAcc=0.719
    Epoch 5/30 | TrainAcc=0.852 ValAcc=0.718
    Epoch 6/30 | TrainAcc=0.855 ValAcc=0.715
    Epoch 7/30 | TrainAcc=0.856 ValAcc=0.726
    Epoch 8/30 | TrainAcc=0.856 ValAcc=0.732
    Epoch 9/30 | TrainAcc=0.853 ValAcc=0.733
    Epoch 10/30 | TrainAcc=0.855 ValAcc=0.722
    Epoch 11/30 | TrainAcc=0.857 ValAcc=0.736
    Epoch 12/30 | TrainAcc=0.850 ValAcc=0.703
    Epoch 13/30 | TrainAcc=0.857 ValAcc=0.716
    Epoch 14/30 | TrainAcc=0.861 ValAcc=0.742
    Epoch 15/30 | TrainAcc=0.858 ValAcc=0.751
    Epoch 16/30 | TrainAcc=0.855 ValAcc=0.724
    Epoch 17/30 | TrainAcc=0.852 ValAcc=0.723
    Epoch 18/30 | TrainAcc=0.855 ValAcc=0.707
    Epoch 19/30 | TrainAcc=0.857 ValAcc=0.719
    Epoch 20/30 | TrainAcc=0.855 ValAcc=0.698
    Epoch 21/30 | TrainAcc=0.854 ValAcc=0.699
    Epoch 22/30 | TrainAcc=0.853 ValAcc=0.712
    Epoch 23/30 | TrainAcc=0.856 ValAcc=0.712
    Epoch 24/30 | TrainAcc=0.855 ValAcc=0.702
    Epoch 25/30 | TrainAcc=0.853 ValAcc=0.702
    Epoch 26/30 | TrainAcc=0.856 ValAcc=0.738
    Epoch 27/30 | TrainAcc=0.859 ValAcc=0.686
    Epoch 28/30 | TrainAcc=0.859 ValAcc=0.688
    Epoch 29/30 | TrainAcc=0.855 ValAcc=0.732
    Epoch 30/30 | TrainAcc=0.853 ValAcc=0.717
    
    =====================================
       Training MLP with LR = 0.1
    =====================================
    Epoch 1/30 | TrainAcc=0.798 ValAcc=0.715
    Epoch 2/30 | TrainAcc=0.825 ValAcc=0.729
    Epoch 3/30 | TrainAcc=0.823 ValAcc=0.761
    Epoch 4/30 | TrainAcc=0.830 ValAcc=0.694
    Epoch 5/30 | TrainAcc=0.829 ValAcc=0.652
    Epoch 6/30 | TrainAcc=0.822 ValAcc=0.718
    Epoch 7/30 | TrainAcc=0.827 ValAcc=0.679
    Epoch 8/30 | TrainAcc=0.823 ValAcc=0.732
    Epoch 9/30 | TrainAcc=0.831 ValAcc=0.715
    Epoch 10/30 | TrainAcc=0.828 ValAcc=0.710
    Epoch 11/30 | TrainAcc=0.823 ValAcc=0.723
    Epoch 12/30 | TrainAcc=0.824 ValAcc=0.727
    Epoch 13/30 | TrainAcc=0.825 ValAcc=0.736
    Epoch 14/30 | TrainAcc=0.833 ValAcc=0.704
    Epoch 15/30 | TrainAcc=0.820 ValAcc=0.733
    Epoch 16/30 | TrainAcc=0.827 ValAcc=0.639
    Epoch 17/30 | TrainAcc=0.822 ValAcc=0.717
    Epoch 18/30 | TrainAcc=0.830 ValAcc=0.727
    Epoch 19/30 | TrainAcc=0.822 ValAcc=0.697
    Epoch 20/30 | TrainAcc=0.829 ValAcc=0.712
    Epoch 21/30 | TrainAcc=0.833 ValAcc=0.719
    Epoch 22/30 | TrainAcc=0.829 ValAcc=0.710
    Epoch 23/30 | TrainAcc=0.820 ValAcc=0.692
    Epoch 24/30 | TrainAcc=0.825 ValAcc=0.707
    Epoch 25/30 | TrainAcc=0.831 ValAcc=0.720
    Epoch 26/30 | TrainAcc=0.823 ValAcc=0.730
    Epoch 27/30 | TrainAcc=0.825 ValAcc=0.713
    Epoch 28/30 | TrainAcc=0.828 ValAcc=0.708
    Epoch 29/30 | TrainAcc=0.829 ValAcc=0.729
    Epoch 30/30 | TrainAcc=0.829 ValAcc=0.708
    
    --------------------------------------
    Best MLP
    --------------------------------------
    Best LR             = 0.0001
    Best Validation Acc = 0.7697
    Test Acc (finale)   = 0.7473
    

## Results


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("mlp_best/best_lr_accuracy_curve.png")
plt.figure(figsize=(8, 5))
plt.imshow(img)
plt.axis('off')
plt.title("Best LR – Accuracy Curve")
plt.show()
```


    
![png](output_68_0.png)
    


The accuracy curve for the best learning rate (0.0001) exhibits a characteristic but somewhat irregular training dynamic. The training accuracy increases smoothly throughout the entire training process and eventually approaches 90%, which indicates that the MLP classifier is able to fit the latent representations extracted from the autoencoder without difficulty.

The validation accuracy increases rapidly during the first few epochs, reaching a peak of around 0.76. This indicates that the classifier is able to immediately exploit the portions of the latent space that are already well-structured and separable. After this initial phase, however, the validation accuracy gradually declines and stabilizes at a lower plateau (≈0.72–0.75). This behaviour suggests that the classifier quickly extracts all the generalizable structure available in the latent vectors, and further training only leads to overfitting on the training embeddings without improving true generalization.


```python
img = mpimg.imread("mlp_best/best_lr_loss_curve.png")
plt.figure(figsize=(8, 5))
plt.imshow(img)
plt.axis('off')
plt.title("Best LR – Loss Curve")
plt.show()
```


    
![png](output_70_0.png)
    


The loss curve highlights even more clearly the overfitting behaviour already suggested by the accuracy curve. The training loss decreases smoothly and monotonically over the entire training process, approaching values below 0.3. This indicates that the MLP classifier is able to fit the latent representations extremely well: optimization is stable and the learning rate of 0.0001 is sufficiently small to avoid oscillations or divergence.

In contrast, the validation loss displays a markedly different trend. It initially decreases during the first few epochs, reflecting the model’s early ability to exploit the structure present in the latent space, but shortly afterwards it begins to rise steadily. From epoch 5 onward, the validation loss increases almost monotonically, eventually reaching values above 1.5. The classifier continues to adapt to the latent vectors in the training set, but does so in a way that harms generalization to unseen data.

The shape of this curve is especially informative. Unlike cases where validation loss fluctuates due to noise, here the increase is smooth and consistent, indicating that the classifier is systematically departing from the regions of latent space that generalize well. This behaviour suggests that the latent representations lack strong class-separating margins. Because the encoder was not explicitly trained to produce discriminative embeddings, the classifier can lower its loss on the training set but cannot find similarly effective boundaries in the validation set.


```python
#Load best model
clf = MLP(input_dim=latent_dim, num_classes=10).to(device)
clf.load_state_dict(torch.load("mlp_best/MLP_GLOBAL_BEST.pt"))
clf.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in test_dl:
        xb = xb.to(device)
        preds = clf(xb).argmax(1).cpu()
        all_preds.append(preds)
        all_labels.append(yb)

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix – Best MLP")
plt.show()

```

    C:\Users\Matti\AppData\Local\Temp\ipykernel_7444\1250368632.py:3: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      clf.load_state_dict(torch.load("mlp_best/MLP_GLOBAL_BEST.pt"))
    


    <Figure size 800x800 with 0 Axes>



    
![png](output_72_2.png)
    



```python
print(classification_report(all_labels, all_preds, digits=4))
```

                  precision    recall  f1-score   support
    
               0     0.6173    0.9346    0.7435       321
               1     0.9167    0.0375    0.0721       293
               2     0.8945    0.6610    0.7602       295
               3     0.9414    0.9040    0.9223       302
               4     0.9525    0.9525    0.9525       295
               5     0.8800    0.4731    0.6154       279
               6     0.8533    0.7111    0.7758       270
               7     0.9833    0.9365    0.9593       315
               8     0.8433    0.8057    0.8241       314
               9     0.4282    0.9810    0.5962       316
    
        accuracy                         0.7473      3000
       macro avg     0.8311    0.7397    0.7221      3000
    weighted avg     0.8272    0.7473    0.7247      3000
    
    

The confusion matrix shows that, despite an overall test accuracy of 74.73%, the classifier’s performance varies substantially across classes, reflecting limitations in how the autoencoder organizes the latent space.

Several categories form clean, compact clusters in the latent space and are classified reliably. Industrial (4), Residential (7), Highway (3), SeaLake (9) all achieve high recall (≥0.90). These classes have strong visual signatures (regular structures, built environments or uniform water textures) which the encoder can represent clearly.

Vegetation related categories show noticeable confusion. HerbaceousVegetation (2), Pasture (5), and PermanentCrop (6) often overlap. Their visual similarities (texture, color, vegetation density) are compressed into nearby latent regions, making them difficult to distinguish.

Two classes exhibit severe imbalance:
* Forest (1) has very low recall (0.04) and is mostly misclassified as SeaLake (9), indicating a collapse of its embeddings into the wrong latent region.
* SeaLake (9) obtains very high recall (0.98) but low precision (0.43), acting as an attractor for samples from other categories.

AnnualCrop is not problematic but shows asymmetry. It has a high recall (0.93) but lower precision (0.62). Many vegetation-related samples drift into this cluster, again reflecting overlap in how the encoder represents plant-dominated landscapes.

## Conclusion

This project examined whether a supervised convolutional autoencoder can learn a latent space that is not only compact enough for image reconstruction but also sufficiently structured to support downstream land-use classification. The model was extensively tuned through a grid search over both the reconstruction weight α and the learning rate. The most stable and performant configuration (α = 35, LR = 0.005) produced a well-behaved training loss and consistent convergence during reconstruction.

However, the validation loss displayed significant variability, indicating that the autoencoder struggled to maintain a coherent balance between reconstruction and classification across different minibatches. This instability already suggested that the latent space might not be uniformly structured.

Once the encoder was frozen and used purely as a feature extractor, these limitations became clearer. Although the external MLP classifier could fit the training embeddings easily, reaching above 90% accuracy, the validation and test performance remained significantly lower (76.97% and 74.73%, respectively) and the loss curves showed a widening gap between training and validation behavior. This mismatch indicates that the latent representations, rather than the classifier itself, impose a strong bottleneck on generalization.

The confusion matrix confirms that the latent space is only partially discriminative. Visually distinctive classes such as Industrial, Residential, Highway, and SeaLake form well-separated clusters, consistently achieving high recall. In contrast, vegetation-related categories (Herbaceous Vegetation, Pasture, Permanent Crop) frequently overlap and the Forest class collapses almost entirely into the SeaLake region. These systematic errors reveal that the autoencoder compresses semantically different classes into overlapping or ambiguous latent regions, limiting the classifier’s ability to distinguish them even with sufficient capacity.

Performance is constrained by how the encoder organizes the data, rather than by the classifier’s capacity. The latent space retains enough structure to separate visually distinctive categories but fails to preserve finer distinctions, revealing fundamental representational limits in the autoencoder architecture for this task.

## Future work and possible improvements

Although the supervised autoencoder was able to learn meaningful latent representations, several targeted adjustments could help mitigate the limitations observed in the experiments.

#### Refine the loss balance and training stability.
  
The experiments revealed that the interaction between the reconstruction loss (MSE) and the classification loss (CrossEntropy) can lead to oscillations in validation performance. Further exploration of the reconstruction weight α, potentially combined with techniques such as learning-rate scheduling or gradient clipping, could help stabilize training and yield a more consistent latent space. Improving stability may also mitigate some of the extreme class collapses observed in the confusion matrix.

#### Strengthen the MLP classifier.
Since the encoder is frozen during classification, the burden of separating classes falls entirely on the MLP. Adding an extra hidden layer, increasing the size of existing layers or tuning dropout could enable the classifier to better exploit the structure present in the embeddings.

#### Apply stronger data augmentation.
Rotations, flips, color jitter and small geometric transformations during autoencoder training could help the encoder generalize better and produce more robust latent representations. This is particularly relevant for classes with high intra-class variability.

#### Emphasize hard vegetation classes.
Vegetation related categories, particularly Forest, systematically exhibit high misclassification rates. Introducing class-dependent weighting in the classification loss (e.g., weights proportional to validation error) or applying targeted class-specific data augmentation could bias the encoder toward learning more discriminative latent features for these challenging classes.

#### Increase the capacity of the encoder–decoder architecture.
A more substantial modification would be to increase the overall capacity of the autoencoder by adding additional convolutional blocks, widening existing layers or introducing skip connections. The current encoder compresses the input very aggressively in only four steps, which may cause visually similar classes, especially vegetation types, to collapse into overlapping regions of the latent space. A deeper or wider encoder could extract richer high-level features before compression, while a slightly more expressive decoder would help maintain reconstruction stability. Although more computationally demanding, increasing model capacity would directly address the structural bottlenecks observed in the latent representations.

#### Stabilize classifier training on latent embeddings.
The classification curves showed that the MLP can overfit the latent vectors even when regularized, leading to oscillatory or uneven validation accuracy. Exploring additional regularization strategies such as stronger weight decay, alternative dropout rates or adjusting the batch size could help smooth the optimization process and improve generalization. Since the encoder is frozen, even small improvements in classifier stability may lead to more reliable performance without altering the autoencoder itself.
