import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


games = ['Magic: The Gathering', 'Pokemon', 'Yu-Gi-Oh!']
card_rarity = ['Common', 'Uncommon', 'Rare', 'Mythic Rare', 'Legendary']
num_cards = 300

data = []
for _ in range(num_cards):
    game = random.choice(games)
    rarity = random.choice(card_rarity)
    card_value = random.uniform(1, 100)  # Arbitrary card value
    card_power = random.uniform(0, 10)  # Arbitrary power level
    data.append([game, rarity, card_value, card_power])

df = pd.DataFrame(data, columns=['Game', 'Rarity', 'Card Value', 'Card Power'])

# Encoding categorical features
game_encoding = {'Magic: The Gathering': 0, 'Pokemon': 1, 'Yu-Gi-Oh!': 2}
rarity_encoding = {'Common': 0, 'Uncommon': 1, 'Rare': 2, 'Mythic Rare': 3, 'Legendary': 4}
df['Game'] = df['Game'].map(game_encoding)
df['Rarity'] = df['Rarity'].map(rarity_encoding)

# Feature matrix
X = df[['Game', 'Rarity', 'Card Value', 'Card Power']].values

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plotting the t-SNE result
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Game'], cmap='viridis', label=df['Game'])
plt.colorbar()
plt.title('t-SNE Visualization of Rare Cards from Magic: The Gathering, Pokemon, and Yu-Gi-Oh!')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


games = ['Magic: The Gathering', 'Pokemon', 'Yu-Gi-Oh!']
card_rarity = ['Common', 'Uncommon', 'Rare', 'Mythic Rare', 'Legendary']
num_cards = 300

data = []
for _ in range(num_cards):
    game = random.choice(games)
    rarity = random.choice(card_rarity)
    card_value = random.uniform(1, 100)  # Arbitrary card value
    card_power = random.uniform(0, 10)  # Arbitrary power level
    data.append([game, rarity, card_value, card_power])

df = pd.DataFrame(data, columns=['Game', 'Rarity', 'Card Value', 'Card Power'])

# Encoding categorical features
game_encoding = {'Magic: The Gathering': 0, 'Pokemon': 1, 'Yu-Gi-Oh!': 2}
rarity_encoding = {'Common': 0, 'Uncommon': 1, 'Rare': 2, 'Mythic Rare': 3, 'Legendary': 4}
df['Game'] = df['Game'].map(game_encoding)
df['Rarity'] = df['Rarity'].map(rarity_encoding)

# Feature matrix
X = df[['Game', 'Rarity', 'Card Value', 'Card Power']].values

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plotting the t-SNE result
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Game'], cmap='viridis', label=df['Game'])
plt.colorbar()
plt.title('t-SNE Visualization of Rare Cards from Magic: The Gathering, Pokemon, and Yu-Gi-Oh!')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


games = ['Magic: The Gathering', 'Pokemon', 'Yu-Gi-Oh!']
card_rarity = ['Common', 'Uncommon', 'Rare', 'Mythic Rare', 'Legendary']
num_cards = 300

data = []
for _ in range(num_cards):
    game = random.choice(games)
    rarity = random.choice(card_rarity)
    card_value = random.uniform(1, 100)  # Arbitrary card value
    card_power = random.uniform(0, 10)  # Arbitrary power level
    data.append([game, rarity, card_value, card_power])

df = pd.DataFrame(data, columns=['Game', 'Rarity', 'Card Value', 'Card Power'])

# Encoding categorical features
game_encoding = {'Magic: The Gathering': 0, 'Pokemon': 1, 'Yu-Gi-Oh!': 2}
rarity_encoding = {'Common': 0, 'Uncommon': 1, 'Rare': 2, 'Mythic Rare': 3, 'Legendary': 4}
df['Game'] = df['Game'].map(game_encoding)
df['Rarity'] = df['Rarity'].map(rarity_encoding)

# Feature matrix
X = df[['Game', 'Rarity', 'Card Value', 'Card Power']].values

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plotting the t-SNE result
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Game'], cmap='viridis', label=df['Game'])
plt.colorbar()
plt.title('t-SNE Visualization of Rare Cards from Magic: The Gathering, Pokemon, and Yu-Gi-Oh!')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


games = ['Magic: The Gathering', 'Pokemon', 'Yu-Gi-Oh!']
card_rarity = ['Common', 'Uncommon', 'Rare', 'Mythic Rare', 'Legendary']
num_cards = 300

data = []
for _ in range(num_cards):
    game = random.choice(games)
    rarity = random.choice(card_rarity)
    card_value = random.uniform(1, 100)  # Arbitrary card value
    card_power = random.uniform(0, 10)  # Arbitrary power level
    data.append([game, rarity, card_value, card_power])

df = pd.DataFrame(data, columns=['Game', 'Rarity', 'Card Value', 'Card Power'])

# Encoding categorical features
game_encoding = {'Magic: The Gathering': 0, 'Pokemon': 1, 'Yu-Gi-Oh!': 2}
rarity_encoding = {'Common': 0, 'Uncommon': 1, 'Rare': 2, 'Mythic Rare': 3, 'Legendary': 4}
df['Game'] = df['Game'].map(game_encoding)
df['Rarity'] = df['Rarity'].map(rarity_encoding)

# Feature matrix
X = df[['Game', 'Rarity', 'Card Value', 'Card Power']].values

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plotting the t-SNE result
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Game'], cmap='viridis', label=df['Game'])
plt.colorbar()
plt.title('t-SNE Visualization of Rare Cards from Magic: The Gathering, Pokemon, and Yu-Gi-Oh!')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


games = ['Magic: The Gathering', 'Pokemon', 'Yu-Gi-Oh!']
card_rarity = ['Common', 'Uncommon', 'Rare', 'Mythic Rare', 'Legendary']
num_cards = 300

data = []
for _ in range(num_cards):
    game = random.choice(games)
    rarity = random.choice(card_rarity)
    card_value = random.uniform(1, 100)  # Arbitrary card value
    card_power = random.uniform(0, 10)  # Arbitrary power level
    data.append([game, rarity, card_value, card_power])

df = pd.DataFrame(data, columns=['Game', 'Rarity', 'Card Value', 'Card Power'])

# Encoding categorical features
game_encoding = {'Magic: The Gathering': 0, 'Pokemon': 1, 'Yu-Gi-Oh!': 2}
rarity_encoding = {'Common': 0, 'Uncommon': 1, 'Rare': 2, 'Mythic Rare': 3, 'Legendary': 4}
df['Game'] = df['Game'].map(game_encoding)
df['Rarity'] = df['Rarity'].map(rarity_encoding)

# Feature matrix
X = df[['Game', 'Rarity', 'Card Value', 'Card Power']].values

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plotting the t-SNE result
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Game'], cmap='viridis', label=df['Game'])
plt.colorbar()
plt.title('t-SNE Visualization of Rare Cards from Magic: The Gathering, Pokemon, and Yu-Gi-Oh!')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


games = ['Magic: The Gathering', 'Pokemon', 'Yu-Gi-Oh!']
card_rarity = ['Common', 'Uncommon', 'Rare', 'Mythic Rare', 'Legendary']
num_cards = 300

data = []
for _ in range(num_cards):
    game = random.choice(games)
    rarity = random.choice(card_rarity)
    card_value = random.uniform(1, 100)  # Arbitrary card value
    card_power = random.uniform(0, 10)  # Arbitrary power level
    data.append([game, rarity, card_value, card_power])

df = pd.DataFrame(data, columns=['Game', 'Rarity', 'Card Value', 'Card Power'])

# Encoding categorical features
game_encoding = {'Magic: The Gathering': 0, 'Pokemon': 1, 'Yu-Gi-Oh!': 2}
rarity_encoding = {'Common': 0, 'Uncommon': 1, 'Rare': 2, 'Mythic Rare': 3, 'Legendary': 4}
df['Game'] = df['Game'].map(game_encoding)
df['Rarity'] = df['Rarity'].map(rarity_encoding)

# Feature matrix
X = df[['Game', 'Rarity', 'Card Value', 'Card Power']].values

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plotting the t-SNE result
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Game'], cmap='viridis', label=df['Game'])
plt.colorbar()
plt.title('t-SNE Visualization of Rare Cards from Magic: The Gathering, Pokemon, and Yu-Gi-Oh!')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


games = ['Magic: The Gathering', 'Pokemon', 'Yu-Gi-Oh!']
card_rarity = ['Common', 'Uncommon', 'Rare', 'Mythic Rare', 'Legendary']
num_cards = 300

data = []
for _ in range(num_cards):
    game = random.choice(games)
    rarity = random.choice(card_rarity)
    card_value = random.uniform(1, 100)  # Arbitrary card value
    card_power = random.uniform(0, 10)  # Arbitrary power level
    data.append([game, rarity, card_value, card_power])

df = pd.DataFrame(data, columns=['Game', 'Rarity', 'Card Value', 'Card Power'])

# Encoding categorical features
game_encoding = {'Magic: The Gathering': 0, 'Pokemon': 1, 'Yu-Gi-Oh!': 2}
rarity_encoding = {'Common': 0, 'Uncommon': 1, 'Rare': 2, 'Mythic Rare': 3, 'Legendary': 4}
df['Game'] = df['Game'].map(game_encoding)
df['Rarity'] = df['Rarity'].map(rarity_encoding)

# Feature matrix
X = df[['Game', 'Rarity', 'Card Value', 'Card Power']].values

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plotting the t-SNE result
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Game'], cmap='viridis', label=df['Game'])
plt.colorbar()
plt.title('t-SNE Visualization of Rare Cards from Magic: The Gathering, Pokemon, and Yu-Gi-Oh!')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


games = ['Magic: The Gathering', 'Pokemon', 'Yu-Gi-Oh!']
card_rarity = ['Common', 'Uncommon', 'Rare', 'Mythic Rare', 'Legendary']
num_cards = 300

data = []
for _ in range(num_cards):
    game = random.choice(games)
    rarity = random.choice(card_rarity)
    card_value = random.uniform(1, 100)  # Arbitrary card value
    card_power = random.uniform(0, 10)  # Arbitrary power level
    data.append([game, rarity, card_value, card_power])

df = pd.DataFrame(data, columns=['Game', 'Rarity', 'Card Value', 'Card Power'])

# Encoding categorical features
game_encoding = {'Magic: The Gathering': 0, 'Pokemon': 1, 'Yu-Gi-Oh!': 2}
rarity_encoding = {'Common': 0, 'Uncommon': 1, 'Rare': 2, 'Mythic Rare': 3, 'Legendary': 4}
df['Game'] = df['Game'].map(game_encoding)
df['Rarity'] = df['Rarity'].map(rarity_encoding)

# Feature matrix
X = df[['Game', 'Rarity', 'Card Value', 'Card Power']].values

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plotting the t-SNE result
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Game'], cmap='viridis', label=df['Game'])
plt.colorbar()
plt.title('t-SNE Visualization of Rare Cards from Magic: The Gathering, Pokemon, and Yu-Gi-Oh!')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


games = ['Magic: The Gathering', 'Pokemon', 'Yu-Gi-Oh!']
card_rarity = ['Common', 'Uncommon', 'Rare', 'Mythic Rare', 'Legendary']
num_cards = 300

data = []
for _ in range(num_cards):
    game = random.choice(games)
    rarity = random.choice(card_rarity)
    card_value = random.uniform(1, 100)  # Arbitrary card value
    card_power = random.uniform(0, 10)  # Arbitrary power level
    data.append([game, rarity, card_value, card_power])

df = pd.DataFrame(data, columns=['Game', 'Rarity', 'Card Value', 'Card Power'])

# Encoding categorical features
game_encoding = {'Magic: The Gathering': 0, 'Pokemon': 1, 'Yu-Gi-Oh!': 2}
rarity_encoding = {'Common': 0, 'Uncommon': 1, 'Rare': 2, 'Mythic Rare': 3, 'Legendary': 4}
df['Game'] = df['Game'].map(game_encoding)
df['Rarity'] = df['Rarity'].map(rarity_encoding)

# Feature matrix
X = df[['Game', 'Rarity', 'Card Value', 'Card Power']].values

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plotting the t-SNE result
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Game'], cmap='viridis', label=df['Game'])
plt.colorbar()
plt.title('t-SNE Visualization of Rare Cards from Magic: The Gathering, Pokemon, and Yu-Gi-Oh!')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
