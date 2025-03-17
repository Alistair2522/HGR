import pickle

data_dict = pickle.load(open('./data.pickle', 'rb'))

X = data_dict['data']
Y = data_dict['labels']
print(data_dict)
for i in range(len(X)):
    if len(X[i])!=42:
        print("dimension Fault")
        print(Y[i])
        print(i)