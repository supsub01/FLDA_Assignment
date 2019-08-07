from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

#function used to draw normal curve
def solve(m1,m2,std1,std2):
    a1 = 1/(2*std1**2)
    a= a1- 1/(2*std2**2)
    b1 = m2/(std2**2)
    b= b1-m1/(std1**2)
    c1 = m1**2 /(2*std1**2)
    c2=c1-m2**2 / (2*std2**2)
    c=c2-np.log(std2/std1)
    return np.roots([a,b,c])

#Reading the data into df, df_class
columns = ["Sno","x_coord","y_coord","class"]
df = pd.read_csv(r'C:\Users\Supriya-PC\Desktop\ML_ASSIGNMENT_1\dataset_3.csv',index_col=False, names = columns)
df_class = pd.read_csv(r'C:\Users\Supriya-PC\Desktop\ML_ASSIGNMENT_1\dataset_3.csv',index_col=False, names = columns)
df_class.drop(['Sno'], axis = 1, inplace = True)
df.drop(['Sno'], axis = 1, inplace = True)

#Plotting the classes
cmap = cm.get_cmap('Spectral')
ax1 = df.plot.scatter(x='x_coord', y='y_coord', c='class', cmap=cmap)
#plt.show()


#Calculating Between Class Scatter Matrix (SB)- per class mean, total mean
mean_vec = []
for i in df["class"].unique():                               #limits i to 0,1
    mean_vec.append( np.array((df[df["class"]==i].mean())))
# print('MEAN_VEC',mean_vec)                                   # mean per class
total_mean_vec=np.array((df.mean()))
# print('TOTAL MEAN VEC',total_mean_vec)                       #total mean

#Calculating Between Class Scatter Matrix (SB)-
for i in range(2):                                            # 2 is number of classes
    SB_mat=mean_vec[0]-mean_vec[1]
    SB_mat=SB_mat[0:2]
print('between-class Scatter Matrix:\n', SB_mat[0:2])

#Calculating Within Class Scatter Matrix (SW)-
SW = np.zeros((2,2))
for i in range(2):                                  #2 is number of classes
    per_class_sc_mat = np.zeros((2,2))
    for j in range(1000):    # 500 iterations for each i
        if(df.loc[j]["class"]==i):
            mv = mean_vec[i][:2].reshape(2,1)           # mv- make mean vector a col vec
            row=df.loc[j][:2]                           # row- jth row, x and y values are made into a col vec
            row=row.values.reshape(2,1)                 #convert Pandas Series to Numpy Array
            temp=row-mv
            per_class_sc_mat += (row-mv).dot((row-mv).T)
        # print('PER CLASS',i,per_class_sc_mat)
    SW += per_class_sc_mat
print('within-class Scatter Matrix:\n', SW)

#Calculating weight inverse(SW)SB
weight = np.dot(np.linalg.pinv(SW),SB_mat.T)
print('weight',weight)

#Transforming to One Dimension by multiplying the given X and Y coordinates with the calculated parameters
X=df
X.drop(['class'], axis = 1, inplace = True)
weight=weight.reshape(2,1)
#print('W \n',weight)
#print('X \n',X)

X_lda=X.dot(weight)
# print('X_lda \n',X_lda)

X_lda.columns=["LDA"]
zero = pd.DataFrame(np.zeros((1000, 1)))
df_col = pd.concat([X_lda,df_class,zero], axis=1)
df_col.columns=["LDA","x_coord","y_coord","class","zero"]

#Visualising transformed points as points on a line
cmap = cm.get_cmap('Spectral')
ax1 = df_col.plot.scatter(x='LDA', y='zero', c='class', cmap=cmap)
# plt.show()


#Seperating the one dimenstion points based on class
df_col_0 = df_col[df_col['class'] == 0]
df_col_1 = df_col[df_col['class'] == 1]

df_col_0=sorted(df_col_0["LDA"])
df_col_1=sorted(df_col_1["LDA"])

# print('df_col_0 \n',df_col_0)
# print('df_col_1 \n',df_col_1)

#Calculating mean and standard deviation for each class
mean1=np.mean(df_col_0)
mean2=np.mean(df_col_1)

std1=np.std(df_col_0)
std2=np.std(df_col_1)

result=solve(mean1,mean2,std1,std2) #Refer to above solve function

#Plotting normal curves for each class and calculating point of intersection of both curves as threshold
fit1=stats.norm.pdf(df_col_0,mean1,std1)
plt.plot(df_col_0,fit1)
fit2=stats.norm.pdf(df_col_1,mean2,std2)
plt.plot(df_col_1,fit2)
plt.plot(result,stats.norm.pdf(result,mean1,std1),'o')
plt.show()

threshold=result[1]
print('Threshold',threshold)




