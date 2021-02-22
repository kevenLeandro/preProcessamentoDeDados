import  numpy as np # Disponibiliza objetos multidimensionais, tem diversas implementações de funções matematicas
import matplotlib.pyplot as plt # viabiliza a visualização dos dados
import pandas as pd  # fornece a implementação de estutura de dados eficientes
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# com o prefico r, caracteres de escape são ignorados
dataset = pd.read_csv(r'04_dados_exercicio.csv')

print(dataset,end='\n')


#iloc: integer ou index location
#primeiro range: todas as linhas , pega as variaveis dependentes
#segundo range: pega apenas a ultima, a variavel independente/feature
#iloc devolve instâncias associadas a índices
#values pega somente os valores


features = dataset.iloc[:,:-1].values
print( "======= features=======", end='\n')
print(features)

classe = dataset.iloc[:,-1].values
print("======= features=======", end='\n')
print(classe)

# Tratar dados Faltantes
imputer = SimpleImputer(missing_values=np.nan,
strategy="mean")

imputer.fit(features[:, 2:4])
# transform: faz a operação e gera um objeto alterado
features[:, 2:4] = imputer.transform(features[:, 2:4])

print( "======= features=======", end='\n')
print(features)

columnTransformer = ColumnTransformer(
transformers=[('encoder', OneHotEncoder(), [0])],
remainder='passthrough')
#features = np.array(columnTransformer.fit_transform(features[:, 2:4]))

print( "======= features=======", end='\n')
print(features)


print( "======= classe=======", end='\n')
print(classe)

labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)



#treinamento

features_treinamento, features_teste, classe_treinamento, classe_teste = train_test_split(features, classe, test_size = 0.15, random_state=1)

print("======= features_treinamento=======", end='\n')
print(features_treinamento)

print( "======= classe_teste=======", end='\n')
print(classe_teste)


# estudar distribuição de probabilidades, sino
standardScaler = StandardScaler()

features_treinamento[:,4:] = standardScaler.fit_transform(features_treinamento[:,4:])
features_teste[:, 4:] = standardScaler.transform(features_teste[:,4:])

print("====== features treinamento ======")
print(features_treinamento)
print("====== feature teste ======")
print(features_teste)