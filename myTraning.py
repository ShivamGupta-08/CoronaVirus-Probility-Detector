import pandas as pd
import pickle
import numpy as np

def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size =int(len(data)*ratio)
    test_incides= shuffled[:test_set_size]
    train_incides= shuffled[test_set_size:]
    return data.iloc[train_incides],data.iloc[test_incides]

if __name__ == '__main__':
    df= pd.read_csv('data.csv')
    train,test = data_split(df,0.2)
    x_train = train[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()
    x_test = test[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()
    y_train = train[['infectionProb']].to_numpy().reshape(1752 )
    y_test = test[['infectionProb']].to_numpy().reshape(438 )
    from sklearn.linear_model import LogisticRegression
    clf= LogisticRegression()
    clf.fit(x_train,y_train)

    file = open('model.pkl','wb')
    pickle.dump(clf,file)
    file.close()

    