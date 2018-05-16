import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# fetch data and format it
data = fetch_movielens(min_rating=4.0)

# print traning and testing data

print(repr(data['train']))
print(repr(data['test']))

#create model

model = LightFM(loss='warp')

# train model

model.fit(data['train'] , epochs=30 , num_threads=2)

def sample_recommodention(model , data, user_id) :
    # number of users and movies in training data
    n_user , n_items =  data['train'].shape
    # generate recommedation for each user we input
    for user_id in user_id :
        known_positive = data['item_labels'][data['train'].tocsr()[user_id].indices]
        # movies our model prdicts they will like
        score = model.predict(user_id , np.arange(n_items))
        # rank them in orgder of most liked to least
        top_items = data['item_labels'][np.argsort(-score)]
        # print out the results
        print("User %s" % user_id)
        print("   Known positives :")

        for x in known_positive[:3]:
            print("               %s" % x)
        print("      Recommendation :")

        for x in top_items[:3]:
            print("             %s" % x)

sample_recommodention(model , data , [3,25,420])

