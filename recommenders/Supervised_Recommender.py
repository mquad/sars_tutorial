from util.data_expansion import data_expansion,user_profile_expansion
from recommenders.ISeqRecommender import ISeqRecommender
from util.split import balance_dataset
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

"""Adapted from Zimdars, Andrew, David Maxwell Chickering, and Christopher Meek.
"Using temporal data for making recommendations." In Proceedings of the Seventeenth conference
on Uncertainty in artificial intelligence, pp. 580-588. Morgan Kaufmann Publishers Inc., 2001."""

class SupervisedRecommender(ISeqRecommender):

    def __init__(self,history_length,classifier=DecisionTreeClassifier(), balance=True):
        """
        :param history_length: How many recent items to consider
        :param classifier: anything from sklearn: decision tree, logistic regression
        :param balance : whether to balance or not the training data for each item
        :return:
        """

        super(SupervisedRecommender, self).__init__()
        self.classifier = classifier
        self.history_length = history_length
        self.balance=balance

    def fit(self, sequences):

        data,self.mapping = data_expansion(sequences,self.history_length)
        self.item_classifier={}
        #for each column i.e. item, build a classifier
        with tqdm(total=len(self.mapping)) as pbar:
            for key,value in self.mapping.items():
                train,test =self._split_train_test(data,value,len(self.mapping))
                if self.balance:
                    train,test = balance_dataset(train,test)
                self.item_classifier[key] = self.classifier.fit(train,test.toarray().ravel())
                #reset classifier
                self.classifier=clone(self.classifier)
                pbar.update(1)

    def recommend(self, user_profile):
        #print('recommending')
        data= user_profile_expansion(user_profile,self.history_length,self.mapping)
        recommendations =[]
        for item,c in self.item_classifier.items():
            if c.predict(data)==[1]:
                recommendations.append(item)
        return [([x],1/len(recommendations)) for x in recommendations]



    def _split_train_test(self,data,col_index,n_unique_items):
        test = data[:,col_index]
        train = data[:,[x for x in range(data.shape[1]) if x >= n_unique_items]]
        return train,test

    def set_classifier(self,classifier):
        self.classifier = classifier
