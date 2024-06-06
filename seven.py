import pandas as pd
import itertools

# Data creation
data = {'tid': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
        'items': [['HotDogs', 'Buns', 'Ketchup'],
                  ['HotDogs', 'Buns'],
                  ['HotDogs', 'Coke', 'Chips'],
                  ['Chips', 'Coke'],
                  ['Chips', 'Ketchup'],
                  ['HotDogs', 'Coke', 'Chips']]}

df = pd.DataFrame(data)

# Creating the basket (one-hot encoded DataFrame)
basket = pd.DataFrame(df['items'].tolist(), index=df['tid']).stack().reset_index(level=1, drop=True).reset_index()
basket.columns = ['tid', 'items']
basket = pd.crosstab(basket['tid'], basket['items'])

# Encoding function
def encode_units(x):
    return 1 if x >= 1 else 0

basket_sets = basket.applymap(encode_units)

# Function to generate frequent itemsets
def apriori_manual(df, min_support=0.3334):
    itemset_support = {}
    num_transactions = len(df)
    items = df.columns

    def get_support(itemset):
        mask = df[list(itemset)].all(axis=1)
        return mask.sum() / num_transactions

    # Generate frequent 1-itemsets
    for item in items:
        support = get_support([item])
        if support >= min_support:
            itemset_support[frozenset([item])] = support

    current_itemsets = list(itemset_support.keys())
    k = 2

    while current_itemsets:
        new_itemsets = list(itertools.combinations(set(itertools.chain.from_iterable(current_itemsets)), k))
        new_itemset_support = {}

        for itemset in new_itemsets:
            support = get_support(itemset)
            if support >= min_support:
                new_itemset_support[frozenset(itemset)] = support

        itemset_support.update(new_itemset_support)
        current_itemsets = new_itemset_support.keys()
        k += 1

    frequent_itemsets = pd.DataFrame(
        [(list(itemset), support) for itemset, support in itemset_support.items()],
        columns=['itemsets', 'support']
    )

    return frequent_itemsets

frequent_itemsets = apriori_manual(basket_sets)

# Function to generate association rules
def generate_association_rules(frequent_itemsets, min_confidence=0.6):
    rules = []
    itemsets = frequent_itemsets['itemsets'].tolist()
    supports = dict(zip(map(frozenset, frequent_itemsets['itemsets']), frequent_itemsets['support']))

    for itemset in itemsets:
        if len(itemset) > 1:
            for subset in itertools.chain(*[itertools.combinations(itemset, r) for r in range(1, len(itemset))]):
                antecedent = frozenset(subset)
                consequent = frozenset(itemset) - antecedent
                if supports[frozenset(itemset)] / supports[antecedent] >= min_confidence:
                    rules.append({
                        'antecedent': list(antecedent),
                        'consequent': list(consequent),
                        'antecedent support': supports[antecedent],
                        'consequent support': supports[consequent],
                        'support': supports[frozenset(itemset)],
                        'confidence': supports[frozenset(itemset)] / supports[antecedent],
                        'lift': supports[frozenset(itemset)] / (supports[antecedent] * supports[consequent])
                    })

    return pd.DataFrame(rules)

rules = generate_association_rules(frequent_itemsets)

print("Frequent itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules.sort_values(by='confidence', ascending=False))