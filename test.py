import pandas

mood_dataset = pandas.read_csv("resources/data_moods.csv", usecols=['id', 'energy', 'valence'], index_col='id')
key = '2H7PHVdQ3mXqEHXcvclTB0'
if key in mood_dataset.index:
    row = mood_dataset.loc[key]['energy']
    print(row)
else:
    print('no key')