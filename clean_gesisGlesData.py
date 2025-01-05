import pandas as pd
import numpy as np

###########################
####  DOWNLOAD 
###########################
# - waves 1- 21: https://search.gesis.org/research_data/ZA6838
# - waves 22-27: https://search.gesis.org/research_data/ZA7728 to ZA7733

filenames = ["ZA6838_v6-0-0.dta/ZA6838_w1to9_sA_v6-0-0.dta"]*9+[f"ZA6838_v6-0-0.dta/ZA6838_w{x}_sA_v6-0-0.dta" for x in range(10,22)]+["ZA7728_v1-0-0.dta", "ZA7729_v1-0-0.dta", "ZA7730_v1-0-0.dta", "ZA7731_sA_v1-0-0.dta", "ZA7732_sA_v1-0-0.dta", "ZA7733_sA_v1-0-0.dta"]

###########################
####  Variables
###########################

# ATTITUDES 
# The attitudes appear in most waves.
# 


polPos_dict={
    "1090": "econ low_tax (1-kpx_1090)",  # inverted
    "1130": "migr restrict_migration (kpx_1130)",
    "1210": "migr assimilate_migrants (1-kpx_1210)", # inverted
    "1290": "clim econ>_climate (kpx_1290)",
    "1411": "secu pro_surveillance (1-kpx_1411)", # inverted
    "1250": "euro eu_tooFar (kpx_1250)",
    "1590": "civi gender_tooFar (kpx_1590)"
}

invertedCols = ["kpx_1090", "kpx_1210", "kpx_1411"] 

polPos = lambda X, k: f"kp{X}_{k}"

# "How important is topic X to you?"

pos2importance = {"1090":"1100", 
"1130":"1140",
"1210":"1220",
"1290":"1300",
"1411":"1421",
"1590":"1160"}


# Party Identififcation
# coded as 
partyMap = {
    1:"CDU/CSU",
    2:"CDU/CSU", # originally this is only CDU
    3:"CDU/CSU", # origianlly this is only CSU
    4:"SPD",
    5:"FDP",
    6:"GRÜNE",
    7:"DIE LINKE",
    322:"AfD", 
    392:"BSW",
    801:"other", 
    808: "none"
}

###########################
####  MAIN PART
###########################

waves = list(range(1,28))
df_arr = []
for wave, name in zip(waves, filenames):
    ###########################
    ####  Load data
    ###########################
    if wave==1 or wave >=10:
        df = pd.read_stata(f"data/raw/"+name, convert_categoricals=False)
        df.set_index("lfdn",inplace=True)
    d = df.loc[:, [c for c in df.columns if (f"kp{wave}" in c) or ("kpx" in c)]]
    ###########################
    ####  general variables
    ###########################
    d["wave"] = wave
    d["gender"] =  d["gender"] =df["kpx_2280"].map({1:"m", 2:"w", 3:"d"}) if wave<22 else np.nan  
    d["birth-year"] = df["kpx_2290s"] if wave<22 else np.nan  
    if f"kp{wave}_2090b" in df.columns:
        d["partyIdent"] = df[f"kp{wave}_2090b"]
    else:
        print(f"no party identification in wave {wave}")
    d['partyIdent'] = d["partyIdent"].map(partyMap)
    d["time"] = df[f"kp{wave}_datetime"]
    
    ###########################
    ####  political interest
    ########################### 
    attention_to_politics_generalerest_map = {
        "very strongly": 1,
        "strongly": 2,
        "average": 3,
        "less strongly": 4,
        "not at all":5,
        'not participating':np.nan, 'interview stopped':np.nan, 'no answer':np.nan   
    }
    d = d.rename(columns={f"kp{wave}_010":"attention_to_politics_general"})
    d["attention_to_politics_general"] = (5-d["attention_to_politics_general"])/(5-1)  # scaled to 0 (no interest) to 1 (full interest)
    
    
    ###########################
    ####  Political attitudes   
    ###########################
    att_cols = [polPos(wave, k) for k in polPos_dict.keys() if polPos(wave,k) in d.columns]
    att_imp_cols = [polPos(wave, pos2importance[k]) for k in polPos_dict.keys() if k in pos2importance.keys() and polPos(wave, pos2importance[k]) in d.columns]
    
    d = d.loc[:, ["wave", "time", "partyIdent", "gender", "birth-year"]+[f"attention_to_politics_general"]+att_cols +att_imp_cols]
   
    ###########################
    ####  select columns and remove NaN
    ###########################
    print(f"wave-{wave}: n_original={len(d)}", end=" --> ", )
    
    for c in [f"attention_to_politics_general"] + att_cols + att_imp_cols:
        d.loc[d[c]<0, c] = np.nan  
    
    d = d.dropna(how="all", subset=att_cols)
    
    ###########################
    ####  column naming
    ###########################    
    d.rename(columns=dict(zip(att_cols, [c[-4:] for c in att_cols])), inplace=True)
    d.rename(columns=dict(zip(att_imp_cols, [c[-4:] for c in att_imp_cols])), inplace=True)

    print(f"n_processed = {len(d)}")
    
    df_arr.append(d)    

###########################
####  Combine DataFrame
###########################
dftot = pd.concat(df_arr)

dftot["time"] = pd.to_datetime(dftot["time"])
dftot['yymm'] = dftot.time.dt.year + dftot.time.dt.month/12
dftot.drop(columns="time", inplace=True)


###########################
####  Invert and recode attitude columns
###########################
invertedCols = ["1090", "1210", "1411"] 
scale7Map = dict(zip(range(1,8), range(-3, 4)))
for c in list(polPos_dict.keys()):
    dftot[polPos_dict[c]] = (-1 if c in invertedCols else 1) * dftot[c].map(scale7Map)
    dftot.drop(columns=c, inplace=True)

dftot = dftot.rename(columns=dict(zip(pos2importance.values(), ["Imp:"+polPos_dict[k] for k,v in pos2importance.items()])))

atts = list(polPos_dict.values())
att_imps = ["Imp:"+c for c in atts if not c=="euro eu_tooFar (kpx_1250)"]

dftot["country"] = "DE"
dftot["study"] = "Gesis-GLES-Panel"
dftot.index.rename("ID", inplace=True)
dftot.to_csv("data/clean/gesis_waves1-27_polAttitude+Importance.csv")




###########################
####  Analysis -- Boxplot
###########################
# import seaborn as sns 
# import matplotlib.pyplot as plt
# sns.boxplot(dftot[atts], showmeans=True, meanprops={"marker": "+","markeredgecolor": "black", "markersize": "20"},palette="tab10")
# plt.gca().set_xticklabels(["\n".join(str(n).split(" ")[1].split("_")) for n in atts], rotation=90)
# plt.show()


###########################
####  Analysis individual 
###########################
# import matplotlib.pyplot as plt
# multiwavethreshold = 18
# multiwave_participants = (dftot.index.value_counts()>multiwavethreshold)
# multiwave_participants = multiwave_participants.loc[multiwave_participants]
# print(f"{len(multiwave_participants)} of {len(dftot.index.unique())} participants completed more than {multiwavethreshold} questionaires")

# dfPanel = dftot.loc[multiwave_participants.index]

# id = dfPanel.sample(1).index
# fig = plt.figure(figsize=(16,7))
# ax = fig.add_subplot(2,1,1)
# dfPanel.loc[id].sort_values("wave").set_index("wave").loc[:, atts].plot(ls="-", marker="o", markersize=5, ax=ax, alpha=0.5)#.plot.line()
# ax.set_ylim(-3.5,3.5)
# ax.set_yticks(np.arange(-3,3.2)), 
# plt.legend(bbox_to_anchor=(1.01, 0.1, 0.6, 0.8))
# ax.set_xticks(waves)
# ax.set_ylabel("position (7-point scale)")
# ax.grid(axis="y")
# ax.set_title(f"individual {id if type(id)==int else id.values[0]}")

# ax = fig.add_subplot(2,1,2)
# dfPanel.loc[id].sort_values("wave").set_index("wave").loc[:, att_imps].plot(ls="-", marker="o", markersize=5, ax=ax, alpha=0.5)#.plot.line()
# ax.set_ylim(0.5,5.5)
# ax.set_yticks(np.arange(0,5.3))
# ax.grid(axis="y")
# ax.set_ylabel("importance (5-point scale)")
# ax.set_xticks(waves)
# plt.legend(bbox_to_anchor=(1.01, 0.1, 0.65, 0.8))
# fig.tight_layout()
# plt.show()


###########################
####  Analysis -- BS
###########################
# import networkx as nx

# a = dftot[atts].corr() - np.diag(np.ones(len(atts)))
# G = nx.from_pandas_adjacency(a)


# def draw_netw(ax, G, mImp, cmap, s=500, s_edge=10):
#     pos = nx.spring_layout(G, k=1, iterations=1000)
#     nodesizes = dict(zip([c[4:] for c in mImp.keys()], [s*mImp[c] for c in mImp.keys()]))
#     nodesizes['eu_tooFar (kpx_1250)'] = s*np.mean(mImp.values)
#     nodesizes = [nodesizes[n] for n in atts]
#     nx.draw_networkx_nodes(G, pos=pos, node_color=[cmap[c] for c in atts], nodelist=atts, alpha=1, node_size=nodesizes, node_shape="o", ax=ax)
#     edge_widths = [s_edge*abs(G[u][v]['weight']) for u, v in G.edges()]
#     edge_styles = ['-' if G[u][v]['weight'] > 0 else '--' for u, v in G.edges()]
#     nx.draw_networkx_edges(G, pos, width=edge_widths, style=edge_styles, alpha=0.7, edge_color='black', ax=ax)
#     edge_labels = dict(zip(G.edges, [f"{G.edges[e]['weight']:.2f}" for e in G.edges()]))
#     nx.draw_networkx_edge_labels(G, pos,  edge_labels=edge_labels, font_size=8, label_pos=0.3, ax=ax)
#     nx.draw_networkx_labels(G, pos, labels=dict(zip(G.nodes(), ["\n".join(n.split(" ")[0].split("_")) for n in G.nodes()])), font_size=8, ax=ax)
#     ax.axis("off")
#     return ax
    
# fig = plt.figure(figsize=(16,9))
# ax = plt.axes()
# mImp = dftot.groupby('yymm')[att_imps].mean().mean(axis=0)
# cmap = dict(zip(atts, [plt.get_cmap("tab10")(n+1) for n in range(len(atts))]))
# ax = draw_netw(ax, G, mImp, cmap)
# plt.show()
