from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

def main():

    names = ['Grade','Gender','Age_at_diagnosis','Race','IDH1','TP53','ATRX','PTEN','EGFR','CIC','MUC16','PIK3CA','NF1','PIK3R1','FUBP1','RB1','NOTCH1','BCOR','CSMD3','SMARCA4','GRIN2A','IDH2','FAT4','PDGFRA'] 
    features = ['Grade','Gender','Age_at_diagnosis','Race','IDH1','TP53','ATRX','PTEN','EGFR','CIC','MUC16','PIK3CA','NF1','PIK3R1','FUBP1','RB1','NOTCH1','BCOR','CSMD3','SMARCA4','GRIN2A','IDH2','FAT4','PDGFRA']
    input_file = '0-Datasets/TCGA_GBM_LGG_Mutations_all_Clear.csv'
    target = 'Grade'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     names = names,      # Nome das colunas 
                     usecols = features, # Define as colunas que serão  utilizadas
                     na_values='?')      # Define que ? será considerado valores ausentes
    
    df_original = df.copy()
    # Imprime as 15 primeiras linhas do arquivo
    print("PRIMEIRAS 15 LINHAS\n")
    print(df.head(15))
    print("\n")

    x = df.loc[:, features].values
    y = df.loc[:,[target]].values

    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    normalizedDf = pd.DataFrame(data = x, columns = features)
    normalizedDf = pd.concat([normalizedDf, df[[target]]], axis = 1)
    ShowInformationDataFrame(normalizedDf,"Dataframe Normalized")

    # PCA projection
    pca = PCA()    
    principalComponents = pca.fit_transform(x)
    print("Explained variance per component:")
    print(pca.explained_variance_ratio_.tolist())
    print("\n\n")

    principalDf = pd.DataFrame(data = principalComponents[:,0:2], 
                                columns = ['principal component 1', 
                                        'principal component 2'])
    finalDf = pd.concat([principalDf, df[[target]]], axis = 1)    
    ShowInformationDataFrame(finalDf,"Dataframe PCA")

    VisualizePcaProjection(finalDf, target)

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

def VisualizePcaProjection(finalDf, targetColumn):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [0, 1, ]
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf[targetColumn] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                    finalDf.loc[indicesToKeep, 'principal component 2'],
                    c = color, s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()

if __name__ == "__main__":
    main()