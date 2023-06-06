# This script plots the results from the MAAT toolbox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# *** READ IN DATA FROM MAAT SIMULATION [MAKE SURE FOLDER PATH IS ACCURATE] ***
# df = pd.read_csv("/Users/emy/Desktop/Tests/MAAT/_testing_FATES/results/2023-05-30/out_FATES.csv" , usecols=['leaf.ca_conc','A'])
df = pd.read_csv("/Users/emy/Desktop/Tests/MAAT/_testing_FATES/results/2023-05-30/out_LinInputs_FATES.csv" , usecols=['leaf.ca_conc','leaf.temp','A'], na_values='          NA')
df = df.dropna()
print(df)

# READ IN DATA FROM LIN 2015 OBSERVATIONS
df_2 = pd.read_csv("/Users/emy/Desktop/Tests/MAAT/Lin2015_cleaned_BDTT.csv" , usecols=['CO2S','Tleaf','Photo'], na_values='          NA')
print(df_2)

CO2_conc = df["leaf.ca_conc"]
leaf_temp = df["leaf.temp"]
Anet = df["A"]

CO2_conc_Lin = df_2["CO2S"]
leaf_temp_Lin = df_2["Tleaf"]
Anet_Lin = df_2["Photo"]

# *** Plot Anet vs CO2_conc comparing MAAT-FATES with Lin2015 ***

fig, ax = plt.subplots(figsize=(5, 2.7) ) # , layout='constrained')
ax.plot(CO2_conc, Anet, '.', label='Anet vs CO2_conc (MAAT FATES)')  # Plot some data on the axes.
ax.plot(CO2_conc_Lin, Anet_Lin, '.', label='Anet vs CO2_conc (Lin 2015)')  # Plot some data on the axes.
ax.set_xlabel('CO2_conc')  # Add an x-label to the axes.
ax.set_ylabel('Anet')  # Add a y-label to the axes.
ax.set_title("Photosynthesis vs CO2 Concentration")  # Add a title to the axes.
ax.legend()  # Add a legend.
plt.savefig('/Users/emy/Desktop/Tests/MAAT/_testing_FATES/plotting/Anet vs CO2_con _ FATES.png')
# plt.show()


# *** Plot Anet vs Leaf_temp comparing MAAT-FATES with Lin2015 ***

fig, ax = plt.subplots(figsize=(5, 2.7) ) # , layout='constrained')
ax.plot(leaf_temp, Anet, '.', label='Anet vs Leaf Temp (MAAT FATES)')  # Plot some data on the axes.
ax.plot(leaf_temp_Lin, Anet_Lin, '.', label='Anet vs Leaf Temp (Lin 2015)')  # Plot some data on the axes.
ax.set_xlabel('CO2_conc')  # Add an x-label to the axes.
ax.set_ylabel('Anet')  # Add a y-label to the axes.
ax.set_title("Photosynthesis vs Leaf Temperature")  # Add a title to the axes.
ax.legend()  # Add a legend.
plt.savefig('/Users/emy/Desktop/Tests/MAAT/_testing_FATES/plotting/Anet vs Leaf_temp _ FATES.png')
# plt.show()

# *** Plot Simulated vs Observed Anet comparing MAAT-FATES with Lin2015 ***

fig, ax = plt.subplots(figsize=(5, 2.7) ) # , layout='constrained')
ax.plot(Anet_Lin, Anet, '.', label='Simulated vs Observed (MAAT FATES vs Lin 2015)')  # Plot some data on the axes.
ax.plot( [0,np.max(Anet_Lin) ],[0,np.max(Anet)] )
ax.set_xlabel('Anet_obs')  # Add an x-label to the axes.
ax.set_ylabel('Anet_sim')  # Add a y-label to the axes.
ax.set_title("FATES in MAAT")  # Add a title to the axes.
ax.legend()  # Add a legend.
plt.savefig('/Users/emy/Desktop/Tests/MAAT/_testing_FATES/plotting/Simulated vs Observed Anet _ FATES.png')
# plt.show()

correlation = np.corrcoef(Anet_Lin, Anet)[0,1]
print(correlation)