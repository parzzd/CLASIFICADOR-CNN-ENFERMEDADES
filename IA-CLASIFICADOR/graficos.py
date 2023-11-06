import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargando los datos
data = pd.read_csv("archivo_actualizado.csv")#cambiar al ser utilizado en local

#print(data.columns.tolist())
# Contando las 5 etiquetas más frecuentes en la columna "Finding Labels"
top_5_labels = data["Finding Labels"].value_counts().nlargest(10)

plt.figure(figsize=(18,8))
barplot = sns.barplot(x=top_5_labels.index, y=top_5_labels.values,width=0.1)
plt.title("Mayores 10 datos obtenidos")

# Añadiendo anotaciones con los valores de cada barra
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.2f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'center',
                     xytext = (0, 10),
                     textcoords = 'offset points')

plt.show()