########## Network science model for political bill passing ############
########## Author: Sanjukta Krishnagopal #########
################## March 2022 #######################

from numpy import *
from matplotlib.pyplot import *
import random
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import pandas as pd
from scipy.spatial.distance import cdist
import community
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import itertools

matplotlib.rc('xtick', labelsize=13) 
matplotlib.rc('ytick', labelsize=13)

df = pd.read_csv('bills.csv', delimiter=',')
data = df.to_numpy()
data = data[data[:, 3].argsort()] 
#data_passed = data[data[:,-1]=='bill passed'] #unique passed bills
data_passed = data[data[:,-2]=='bill advanced'] #unique advanced bills #change here for passed to advanced
billid = data_passed[:,0] 
all_billsid=data[:,0]

topics = len(set(data[:,-5]))


df2 = pd.read_csv('teammember_edgelist.csv', delimiter=',')
data2 = df2.to_numpy()
billidsponsor = data2[data2[:,4]=='sponsor',0]
data_sponsor = data2[data2[:,4]=='sponsor']
billidcosponsor = set(data2[data2[:,4]=='active cosponsor',0])
data_cosponsor = data2[data2[:,4]=='active cosponsor']
total_people = len(np.unique(data2[:,2]))

bills = list(set(billid).intersection(set(billidsponsor)).intersection( billidcosponsor))
all_bills = list(set(all_billsid).intersection(set(billidsponsor)).intersection( billidcosponsor))

sorted_data = [[]]*len(bills)
for i in range(len(bills)):
     loc = np.where(data_passed[:,0]==bills[i])[0]  #change this for passed to advanced
     loc_sponsor = np.where(data_sponsor[:,0]==bills[i])[0]
     loc_cosponsor = np.where(data_cosponsor[:,0]==bills[i])[0]
     people = [data_sponsor[loc_sponsor,2][0]]
     people_affil = [data_sponsor[loc_sponsor,-1][0]] #change this to go from maj/min to Dem/Rep
     for p in range(len(loc_cosponsor)):
          people.append(data_cosponsor[loc_cosponsor[p],2])
          people_affil.append(data_cosponsor[loc_cosponsor[p],3])
     a= [data_passed[loc,0][0], data_passed[loc,3][0], data_sponsor[loc_sponsor,3][0],people, people_affil]
     sorted_data[i]=a

sorted_data = np.array(sorted_data)
#sort by date
sorted_data = sorted_data[sorted_data[:, 1].argsort()]


####build a dictionary of all people and their affiliations
people_aff = dict(zip(list(data2[:,2]),list(data2[:,-1]))) #change from maj/min to Rep/Dem

#influence at every timepoint
unique_dates = np.sort(np.array(list(set(sorted_data[:,1]))))

inf_person = {} #dictionary of within and out of party influence for all members in each year

for p in list(people_aff.keys()):
     inf_person[p]=[]
#influence = np.zeros((len(unique_dates),2))

for date in unique_dates:
     print (date) 
     l_date=np.where(sorted_data[:,1]==date)[0]
     all_cosp = sorted_data[l_date,3]
     all_cosp_aff = sorted_data[l_date,3]
     list_all_cosp = np.unique(np.array(list(itertools.chain.from_iterable(all_cosp)))) 
     no_bills_day = len(all_cosp)

     for co in list_all_cosp:
          friends = []
          for b in range(no_bills_day):
               if co in all_cosp[b]:
                    friends.extend(all_cosp[b])
          friends = np.unique(np.array(friends)) #list of all cosponsors on that day for person co
          friends=friends[friends!=co]
          friend_aff = []
          for f in friends:
               friend_aff.append(people_aff[f]) #list of all consponsor affiliations
          friend_aff = np.array(friend_aff)
          party=people_aff[co]
          friends_dem = friends[friend_aff=='majority']#change here to go from maj/min to Dem/Rep 
          friends_rep = friends[friend_aff=='minority']
          inf_person[co].append([date, list(friends_rep),list(friends_dem)])


#consider a bill
Dem = [] #bill sponsor is a dem
Rep = [] #bill sponsor is a rep
n=180 #number of previous days considered for calculating influence
bill_inf_cosponsors={}

count=0
for bill in np.array(all_bills)[-1000:]: #considering previous 1000 bills
     bill_inf_cosponsors[bill]={}
     count+=1
     print (count)
     loc=np.where(data[:,0]==bill)[0]
     bill_id = data[loc,0][0]
     bill_date = data[loc,3][0]
     pass_ornot = data[loc,-1][0]
     loc_sp = np.where(data_sponsor[:,0]==bill)[0]
     sp_id = list(data_sponsor[loc_sp,2])
     sp_affil  = data_sponsor[loc_sp,-1][0] #change for Dem/Rep to maj/min
     loc_cosp = np.where(data_cosponsor[:,0]==bill)[0]
     cosp_id = list(data_cosponsor[loc_cosp,2])
     all_cosp = sp_id+cosp_id
     rep_inf = [] #within the party proposing the bill
     dem_inf = [] #outisde the party proposing the bill
     
     for j in all_cosp: 
          if not inf_person[j]: 
               continue
          prev_dates = np.array(inf_person[j])[:,0]
          a = datetime.strptime(bill_date, '%Y-%m-%d')
          rel_dates = []
          for i in range(len(prev_dates)):
               b = datetime.strptime(prev_dates[i], '%Y-%m-%d')
               if (a-b).days < n and (a-b).days >0: #
                    rel_dates.append(i)
          rel_ones = np.array(inf_person[j])[rel_dates]
          #to consider influence sum all the unique person that the individual has passed bills with in the past n days
          rep_inf.append(np.unique(np.array(list(itertools.chain.from_iterable(rel_ones[:,1]))))) #influence iwithin the cosponsors party
          dem_inf.append(np.unique(np.array(list(itertools.chain.from_iterable(rel_ones[:,2])))))#influence outside the cosponsors party
          
          bill_inf_cosponsors[bill][j] = [rep_inf,dem_inf]
     rep_inf_ = len(np.unique(np.array(list(itertools.chain.from_iterable(rep_inf)))))
     dem_inf_ = len(np.unique(np.array(list(itertools.chain.from_iterable(dem_inf)))))    
     if sp_affil =='majority':               #change here to go from maj/min to Dem/Rep 
          Dem.append([bill_id, bill_date, rep_inf_, dem_inf_, pass_ornot])
     else:
          Rep.append([bill_id, bill_date, rep_inf_, dem_inf_, pass_ornot])


Dem= np.array(Dem)
Rep= np.array(Rep)

    
np.savetxt("majority_pass.csv", Dem, delimiter=",",fmt='%s')
np.savetxt("minority_pass.csv", Rep, delimiter=",",fmt='%s')



