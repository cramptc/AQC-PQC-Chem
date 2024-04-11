import json

import matplotlib.pyplot as plt

fls = ['esu2BravyiKitaevMapper1000results.json', 'esu2JordanWignerMapper1000results.json',
       'uccsdBravyiKitaevMapper1000results.json', 'uccsdJordanWignerMapper1000results.json']
ttls = ['Using ESU2 ansatz with Bravyi-Kitaev Mapper', 'Using ESU2 ansatz with Jordan-Wigner Mapper',
        'Using UCCSD ansatz with Bravyi-Kitaev Mapper', 'Using UCCSD ansatz with Jordan-Wigner Mapper']
bars = []



for filenm, title in list(zip(fls, ttls)):
    # Open the JSON file
    with open(filenm) as file:
        data = json.load(file)

    # Extract the values from the JSON data
    values = list(data)

    # Create a frequency bar graph for the values in the bars array
    plt.hist(bars, edgecolor='black', color='red', linewidth=0.1, fill=False, alpha=0.5)
    
    # Create a frequency bar graph for the values from the JSON data
    plt.bar(values,  edgecolor='black', color='blue', linewidth=0.5, fill=False, alpha=0.5)
    
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(title)

    # Save the graph as an image
    plt.savefig(filenm[:-4] + '.png')