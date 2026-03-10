
"""Generates training data for Latin verb conjugations.
Uses amo, amare (to love) as the base verb.
Example output:
    amare present first singular
    amo
    amare imperfect third plural
    amabant"""

import os
import pickle
import numpy as np
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Verb conjugation data
# Using amo, amare (to love) as the base verb
VERB_STEM = "am"  # Remove the -o from amo to get stem

# Tenses and their endings
TENSES = {
    'present': {
        'singular': ['o', 's', 't'],
        'plural': ['mus', 'tis', 'nt']
    },
    'imperfect': {
        'singular': ['bam', 'bas', 'bat'],
        'plural': ['bamus', 'batis', 'bant']
    },
    'future': {
        'singular': ['bo', 'bis', 'bit'],
        'plural': ['bimus', 'bitis', 'bunt']
    }
}

PERSONS = ['first', 'second', 'third']
NUMBERS_GRAMMAR = ['singular', 'plural']

def generate_verb_form(tense, person, number):
    "Generate a verb conjugation"
    person_idx = PERSONS.index(person)
    ending = TENSES[tense][number][person_idx]
    conjugated = VERB_STEM + ending
    return conjugated

def generate_verb_data(num_sequences=10000):
    "Generate verb conjugations"
    data = []
    
    for _ in range(num_sequences):
        tense = random.choice(list(TENSES.keys()))
        person = random.choice(PERSONS)
        number = random.choice(NUMBERS_GRAMMAR)
        
        conjugated = generate_verb_form(tense, person, number)
        
        # Format: "amare [tense] [person] [number]\n[conjugated form]\n"
        line = f"amare {tense} {person} {number}\n{conjugated}\n"
        data.append(line)
    
    return ''.join(data)

# Generate data
print("Generating Latin verb conjugation data...")
train_data = generate_verb_data(num_sequences=8000)
val_data = generate_verb_data(num_sequences=1000)

# Get all unique characters
all_text = train_data + val_data
chars = sorted(list(set(all_text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {chars}")

# Create character to index mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode function
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Encode the data
train_ids = encode(train_data)
val_ids = encode(val_data)

print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# Export to binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Save metadata
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Data preparation complete!")
print(f"Files created: train.bin, val.bin, meta.pkl")

# Show sample of training data
print("\nSample training data:")
print(train_data[:300])
