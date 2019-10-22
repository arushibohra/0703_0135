#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 06:57:41 2019

@author: arushibohra
"""

import numpy as np 
import pandas as pd 
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
df_electrol_data14 = pd.read_csv("LS2014Electors.csv")
df_candidate_data14 = pd.read_csv("LS2014Candidate.csv")
df_candidate_data14.head()
total_electors=df_electrol_data14["Total_Electors"].sum()
print ("There are a total of ",+total_electors ,"electors in India")
total_voters=df_electrol_data14["Total voters"].sum()
print("There are a total of ",+total_voters ,"voters in India")
total_turnout = round(total_voters/total_electors*100,2)
print("Total Turnout in 2014 is ",+total_turnout,"%")
candidate_sex = df_candidate_data14["Candidate Sex"].value_counts()
candidate_sex
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.pie(df_candidate_data14[(df_candidate_data14["Party Abbreviation"]=='INC')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%', startangle=90)

fig = plt.gcf() 
fig.suptitle("Candidates Gender Distribution in 2014 - INC vs BJP", fontsize=14) 
ax = fig.gca() 
label = ax.annotate("INC", xy=(-1.1,-1), fontsize=30, ha="center",va="center")
ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()

plt.subplot(1,2,2)
plt.pie(df_candidate_data14[(df_candidate_data14["Party Abbreviation"]=='BJP')]['Candidate Sex'].value_counts(), labels=['Male','Female'],autopct='%1.1f%%', startangle=90)
fig = plt.gcf() 
ax = fig.gca() 
label = ax.annotate("BJP", xy=(-1.1,-1), fontsize=30, ha="center",va="center")
ax.axis('off')
ax.set_aspect('equal')
ax.autoscale_view()
plt.show();
#party wise winning women candidates
df_womenwinners14 = df_candidate_data14[(df_candidate_data14['Position']==1)&(df_candidate_data14["Candidate Sex"]=="F")]
ax1 = df_womenwinners14["Party Abbreviation"].value_counts().plot(kind="pie",radius=2,autopct='%1.1f%%', startangle=90)
x = df_womenwinners14["Party Abbreviation"].value_counts()
x
#analysing alliances
df_candidate_data14["Alliance"] = df_candidate_data14["Party Abbreviation"]

df_candidate_data14["Alliance"] = df_candidate_data14["Alliance"].replace(to_replace =['INC','NCP', 'RJD', 'DMK', 'IUML', 'JMM','JD(s)','KC(M)','RLD','RSP','CMP(J)','KC(J)','PPI','MD'],value="UPA")
df_candidate_data14["Alliance"] = df_candidate_data14["Alliance"].replace(to_replace =['BJP','SHS', 'LJP', 'SAD', 'RLSP', 'AD','PMK','NPP','AINRC','NPF','RPI(A)','BPF','JD(U)','SDF','NDPP','MNF','RIDALOS','KMDK','IJK','PNK','JSP','GJM','MGP','GFP','GVP','AJSU','IPFT','MPP','KPP','JKPC','KC(T)','BDJS','AGP','JSS','PPA','UDP','HSPDP','PSP','JRS','KVC','PNP','SBSP','KC(N)','PDF','MDPF'],value="NDA")

df_candidate_data14["Alliance"] = df_candidate_data14["Alliance"].replace(to_replace =['YSRCP',"AITC",'AAAP',"BJD","ADMK",'IND', 'AIUDF', 'BLSP', 'JKPDP',"CPM","TRS","TDP","SP", 'JD(S)', 'INLD', 'CPI', 'AIMIM', 'KEC(M)','SWP', 'NPEP', 'JKN', 'AIFB', 'MUL', 'AUDF', 'BOPF', 'BVA', 'HJCBL', 'JVM','MDMK'],value="Others") 

Age14UPA=df_candidate_data14[(df_candidate_data14.Position==1) & (df_candidate_data14.Year==2014)&(df_candidate_data14.Alliance=="UPA")]['Candidate Age'].tolist()
Age14NDA=df_candidate_data14[(df_candidate_data14.Position==1) & (df_candidate_data14.Year==2014)&(df_candidate_data14.Alliance=="NDA")]['Candidate Age'].tolist()
bins = np.linspace(20, 90, 10)
plt.hist([Age14UPA, Age14NDA], bins, label=['UPA', 'NDA'])
plt.legend(loc='upper right')
plt.xlabel('Age Of winners in years')
plt.ylabel('Total Number of winners')
plt.title('Alliance wise Distribution of Age of the winners in 2014')
plt.show()

#party wise seat winners
df_winners14 = df_candidate_data14[df_candidate_data14['Position']==1]
DF14 = df_winners14['Party Abbreviation'].value_counts().head(10)
DF14
df_winners14 = df_candidate_data14[df_candidate_data14['Position']==1]
DF14 = df_winners14['Party Abbreviation'].value_counts().head().to_dict()
S14 = sum(df_winners14['Party Abbreviation'].value_counts().tolist())
DF14['Other Regional Parties'] = S14 - sum(df_winners14['Party Abbreviation'].value_counts().head().tolist())
fig = plt.figure()

ax14 = fig.add_axes([0, 0,.5,.5], aspect=1)
colors = ["#FF5106","#264CE4","#E426A4","#44A122","#F2EC3A","#C96F58"]
ax14.pie(DF14.values(),labels=DF14.keys(),autopct='%1.1f%%',shadow=True,pctdistance=0.8,radius = 2,colors=colors)
ax14.set_title("2014",loc="center",fontdict={'fontsize':20},position=(0.5,1.55))
plt.show()
