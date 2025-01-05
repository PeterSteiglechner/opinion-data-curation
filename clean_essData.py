# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
###########################
####  Load Data
###########################

data = pd.read_csv("data/raw/ess/ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_5-ESS6e02_6-ESS7e02_3-ESS8e02_3-ESS9e03_2-ESS10SC-ESS11-subset.csv")
data.rename(columns={"idno": "ID"}, inplace=True)
data.rename(columns={"essround": "wave"}, inplace=True)
data.rename(columns={"cntry": "country"}, inplace=True)
waves = list(range(1,12))

# %%
###########################
####  Political Identity
###########################

cols = ["prtcl"+x+"de" for x in ["","a","b", "b","c","d","e","e", "e", "f", "g"]]
#prtid_col = {"prtclde": [1], "prtclade":[2], "prtclbde":[3,4],  "prtclcde":[5],"prtcldde":[6], "prtclede":[7,8,9],"prtclfde":[10], "prtclgde":[11]}
prtid_encoding= {
    "prtclde":{1:"SPD", 2:"CDU/CSU", 3:"Grüne", 4:"FDP", 5: "Linke", 6:"other", 7:"other",}, # Die Linke was formerly called PDS
    "prtclade":{1:"SPD", 2:"CDU/CSU", 3:"Grüne", 4:"FDP", 5: "Linke", 6:"other", 7:"other", 8:"other"  }, # Die Linke was formerly called PDS
    "prtclbde":{1:"SPD", 2:"CDU/CSU", 3:"Grüne", 4:"FDP", 5: "Linke", 6:"other", 7:"other", 8:"other",  }, # Die Linke was formerly called PDS
    "prtclcde":{1:"SPD", 2:"CDU/CSU", 3:"Grüne", 4:"FDP", 5: "Linke", 6:"other", 7:"other", 8:"other",   },
    "prtcldde":{1:"SPD", 2:"CDU/CSU", 3:"Grüne", 4:"FDP", 5: "Linke", 6:"other", 7:"other", 8:"other",  },
    "prtclede":{2:"SPD", 1:"CDU/CSU", 4:"Grüne", 5:"FDP", 3: "Linke", 6:"AfD", 7:"other",  8:"other", 9:"other"  },
    "prtclfde":{2:"SPD", 1:"CDU/CSU", 4:"Grüne", 5:"FDP", 3: "Linke", 6:"AfD", 7:"other", },
    "prtclgde":{2:"SPD", 1:"CDU/CSU", 4:"Grüne", 5:"FDP", 3: "Linke", 6:"AfD", 7:"other", 8:"other", 9:"other", 55:"other"},
}

for wave, col in zip(waves, cols):
    data.loc[data.wave==wave, "partyIdent"] = data.loc[data.wave==wave, col].map(prtid_encoding[col])
    data.loc[(data.wave==wave) & (data[col].isin([66,77,88,99])), "partyIdent"] = np.nan

# %%
###########################
####  political interest
###########################

data.loc[data["polintr"].isin([7,8,9]), "polintr"]=np.nan
data["attention_to_politics_general"] = (4-data["polintr"])/(4-1)

# %%
###########################
#### Standard Variables
###########################
 
data["gender"] = data.gndr.map({1:"m", 2:"f", 9:np.nan})
data["birth-year"] = data["yrbrn"].replace([7777, 8888, 9999], np.nan)

# %%
###########################
####  Political Attitudes
###########################

Likert14 = dict(zip([1,2,3,4, 7, 8, 9], [-1, -0.33, 0.33, 1, np.nan, np.nan, np.nan], ))
Likert15 = dict(zip([1,2,3,4,5, 6,7, 8, 9], [-1, -0.5, 0, 0.5, 1, np.nan, np.nan, np.nan, np.nan], ))
Likert15_inv = dict(zip([1,2,3,4,5, 6,7, 8, 9], [1, 0.5, 0, -0.5, -1, np.nan, np.nan, np.nan, np.nan], ))
Likert010 = dict(zip(np.arange(0,11), np.linspace(-1,1,11)))
Likert010_inv = dict(zip(np.arange(0,11), np.linspace(1,-1,11)))


beliefs = {"impcntr":("migr restrict_econMigration (impcntr)", Likert14),  
           "imdfetn":("migr restrict_ethnicMigration (imdfetn)", Likert14), 
           "imueclt": ("migr migration_undermines_culture (imueclt)", Likert010_inv), # inverted
           "gincdif":("econ no_redistribution (gincdif)", Likert15),
           "freehms": ("civi against_gayChoice (freehms)", Likert15), 
           "hmsacld": ("civi against_gayAdoption (hmsacld)", Likert15), 
           "euftf": ("euro eu_tooFar (euftf)", Likert010_inv),
           "wrclmch": ("clim climate_not_worried (wrclmch)", Likert15_inv),
           "lrscale": ("spec right_spectrum (lrscale)", Likert010),
}
atts = [b for b, _ in beliefs.values()]


for b, (new_col, scale) in beliefs.items():
    data[new_col] = data[b].map(scale) 

###########################
####  TIME OF INTERVIEW
###########################

for w in waves:
    if w in [1,2]:
        data.loc[data.wave.isin([1,2]),"yymm"] = data.loc[data.wave.isin([1,2]), "inwyr"] + data.loc[data.wave.isin([1,2]), "inwmm"] /12
    elif w in [10,11]:
        data.loc[data.wave==11,"time"] = pd.to_datetime(data.loc[data.wave==11, "inwds"])
        data.loc[data.wave==10,"time"] = pd.to_datetime(data.loc[data.wave==10, "questcmp"])
        data.loc[data.wave.isin([10,11]),"yymm"] = data.loc[data.wave.isin([10,11]),"time"].dt.year  + data.loc[data.wave.isin([10,11]),"time"].dt.month/12
    else:
        data.loc[data.wave.isin(range(3,10)),"yymm"] = data.loc[data.wave.isin(range(3,10)), "inwyys"] + data.loc[data.wave.isin(range(3,10)), "inwmms"]/12

# %%
data["study"] = "ESS"
data = data[["ID", "wave", "yymm", "partyIdent","gender","birth-year","attention_to_politics_general"]+atts+["country", "study"]]
data.set_index("ID", inplace=True)
data.to_csv("data/clean/ess-germany.csv")

# %%



