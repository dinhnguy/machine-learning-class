#CODE FOR MACHINE LEARNING HW2

import numpy as np
from numpy import *
import scipy.linalg
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston #LOAD THE DATA FROM SKLEARN
import itertools

#%%
boston = load_boston();
data = boston.data; # feature data
target = boston.target; #target value
test_data = data[range(0,506,7),:]; #splitting the data, test is every 7i th row
train_data = np.delete(data,range(0,506,7),0); #train is the left over after deleting the 7i th rows
test_target = target[range(0,506,7)];#splitting the target values
train_target = np.delete(target,range(0,506,7)); #target values for training


# Histogram of atributes and correlation between each feature and target

for i in range(train_data.shape[1]):
    plt.figure();
    plt.hist(train_data[:,i],bins=10);
    plt.title("%dth Feature Histogram" %(i));
    plt.xlabel("%dth Feature" %(i));
    plt.ylabel("Frequency");
    corr = np.corrcoef(train_data[:,i],train_target)[0,1];
    
    print("The correlation between the %d feature and the target values is %f" %(i,corr));
    print("");

plt.figure();    
plt.hist(train_target, bins=10);
plt.title("Target Histogram");
plt.xlabel("Target Value");
plt.ylabel("Frequency");
plt.show(block=True);


#%%
# define linear regression classifiere
def lin_reg(data,y, x):

    # data = training data
    # x = testing attribute vectors, in form of 2d array, even if it's just one vector i.e np.array([vector])
    # y = target values of data
    row_size = data.shape[0]; # number of training data
    col_size = data.shape[1]; # size of each data
    
    #Standardize training data and the testing data using mean and sd from training set
    column_mean = np.mean(data, axis=0); # feature mean vector
    column_sd = np.std(data, axis =0); # feature sd vector
    norm_data = np.divide(data - column_mean, column_sd);
    norm_x = np.divide(x - column_mean, column_sd);
    
    x_new = np.concatenate((np.array(np.matrix([1]*x.shape[0]).transpose()), norm_x), axis =1); #concatenate 1 to normalized x
    x_0 = np.array(np.matrix([1]*row_size).transpose()); # create the extra 1's column for x
    X = np.concatenate((x_0,norm_data), axis=1); # matrix X after append the 1's column
    
    w = ((linalg.pinv(X.T.dot(X))).dot(X.T)).dot(y); # w=(X^t*X)^-1*X^T*y
    return w.dot(x_new.T); # w times x = y


# In[48]:

# MSE for training set:
print("");
print("Linear Regression")
train_MSE = np.mean((lin_reg(train_data, train_target, train_data) - train_target)**2);
print("The MSE on training set is: %f" %(train_MSE));

# MSE for test set:
test_MSE = np.mean((lin_reg(train_data, train_target, test_data) - test_target)**2);

print("The MSE on test set is: %f" %(test_MSE));


# In[187]:
# define ridge regression classifer;
def ridge_reg(data,lam, y, x):
    import numpy as np
    import scipy.linalg
    # data = training data
    # x = attribute vector
    # lam = lambda for ridge regeression
    # y = target values of data
    row_size = data.shape[0]; # number of training data
    col_size = data.shape[1]; # size of each data
    
    #Standardize training data and the testing data using mean and sd from training set
    column_mean = np.mean(data, axis=0); # feature mean vector
    column_sd = np.std(data, axis =0); # feature sd vector
    norm_data = np.divide(data - column_mean, column_sd);
    norm_x = np.divide(x - column_mean, column_sd);
    
    x_new = np.concatenate((np.array(np.matrix([1]*x.shape[0]).transpose()), norm_x), axis=1); #concatenate 1 to x,
    lam_I = lam * np.identity(col_size + 1);
    x_0 = np.array(np.matrix([1]*row_size).transpose()); # create the extra 1's column for x
    X = np.concatenate((x_0, norm_data), axis=1); # matrix X after append the 1's column
    w = ((linalg.pinv(X.T.dot(X) + lam_I )).dot(X.T)).dot(y); # w=(X^t*X + lam*I)^-1*X^T*y
    return w.dot(x_new.T);


# In[188]:

print("");
print("Ridge Regression")
l =[.01,.1,1.0];
for lam in l:
    print("");
    print("lambda = %f" %(lam));
    # MSE for training set:
    train_MSE = np.mean((ridge_reg(train_data, lam, train_target, train_data) - train_target)**2);
    print("The MSE on training set is: %f" %(train_MSE));

    # MSE for test set:
    test_MSE=np.mean((ridge_reg(train_data, lam, train_target, test_data) - test_target)**2);
    print("The MSE on test set is: %f" %(test_MSE));
    


# In[193]:
print("");
print("Ridge Regression with Cross Validation")
index =np.arange(train_data.shape[0]);
np.random.shuffle(index);
ten_fold = np.array_split(index,10); # ten partitions of indices for train data;
for lam in [.0001, .001, .01, .1, 1, 10]:
    error = 0;
    for fold in ten_fold:
        temp_test = train_data[fold,:]; #temp test set using the fold indicies
        temp_train = np.delete(train_data, fold, 0); # temp train set using the leftover
        temp_test_target = train_target[fold]; #temp test target values
        temp_train_target = np.delete(train_target,fold); # temp train target values
        error += np.mean((ridge_reg(temp_train, lam, temp_train_target, temp_test) - temp_test_target)**2);
    error /=10.0; # performance of lambda
    print("lambda = %f" %(lam));
    print("CV performance: %f" %(error));
    test_MSE=np.mean((ridge_reg(train_data, lam, train_target, test_data) - test_target)**2);
    print("The MSE on test set is: %f" %(test_MSE));
    print("")


print("");
print("Feature Selection");
print("");
print("Selection with Correlation");
corr=[];
for i in range(train_data.shape[1]):
    corr.append(abs(np.corrcoef(train_data[:,i],train_target)[0,1]));
corr = np.array(corr);
four_features = np.argsort(corr)[-4:];
train_MSE = np.mean((lin_reg(train_data[:,four_features], train_target, train_data[:,four_features]) - train_target)**2);
test_MSE = np.mean((lin_reg(train_data[:,four_features], train_target, test_data[:,four_features]) - test_target)**2);
print("The 4 features with the highest correlation with the target: "+str(four_features));
print("The MSE on training set is: %f" %(train_MSE));
print("The MSE on test set is: %f" %(test_MSE));
print("");

feature = np.argsort(corr)[-1:];
for i in range(1,4):
    residue = lin_reg(train_data[:,feature], train_target, train_data[:,feature]) - train_target;
    c=np.array([]);
    for f in np.arange(13):
        c = np.append(c,abs(np.corrcoef(train_data[:,f],residue)[0,1]));
        
    feature = np.append(feature,np.argsort(c)[-1:]);
print("The four features for part b:" + str(feature));
train_MSE = np.mean((lin_reg(train_data[:,feature], train_target, train_data[:, feature]) - train_target)**2);
test_MSE = np.mean((lin_reg(train_data[:,feature], train_target, test_data[:,feature]) - test_target)**2);
print("The MSE on training set is: %f" %(train_MSE));
print("The MSE on test set is: %f" %(test_MSE));

#SELECTION WITH BRUTE FORCE SEARCH
combinations = np.array(list(itertools.combinations(range(13),4)));
best = combinations[0];
error = np.mean((lin_reg(train_data[:,best], train_target, train_data[:,best]) - train_target)**2);

for f in combinations:
    perf = np.mean((lin_reg(train_data[:,f], train_target, train_data[:,f]) - train_target)**2);
    if perf < error:
        best = f;
        error = perf;

print("Selection with Brute-force Search");
print("The best combination of features is: " + str(best));
print("The MSE on training set is: %f" %(error));
test_MSE = np.mean((lin_reg(train_data[:,best], train_target, test_data[:,best]) - test_target)**2);
print("The MSE on testing set is: %f" %(test_MSE));
print("");

#Polynomial feature expansion:
print("Polynomial feature expansion");       
mean = np.mean(train_data, axis = 0);
sd = np.std(train_data, axis =0);
norm_train = np.divide(train_data - mean, sd);
norm_test = np.divide(test_data - mean, sd);
poly_train = train_data;
poly_test = test_data;
for i in range(13):
    poly_train = np.append(poly_train, norm_train[:,i:i+1]**2, axis =1);
    poly_test = np.append(poly_test, norm_test[:,i:i+1]**2, axis = 1);

pairs = np.array(list(itertools.combinations(range(13),2)));
for it in pairs:
    new_train_col = norm_train[:,it[0]:it[0]+1] *norm_train[:, it[1]:it[1]+1];
    new_test_col = norm_test[:,it[0]:it[0]+1] *norm_test[:, it[1]:it[1]+1];
    poly_train = np.append(poly_train, new_train_col, axis =1);
    poly_test = np.append(poly_test, new_test_col, axis =1 );

train_MSE = np.mean((lin_reg(poly_train, train_target, poly_train) - train_target)**2);
test_MSE = np.mean((lin_reg(poly_train, train_target, poly_test) - test_target)**2);
print("");
print("The MSE on training set is: %f" %(train_MSE));
print("The MSE on testing set is: %f" %(test_MSE));


    
    


# In[ ]:



