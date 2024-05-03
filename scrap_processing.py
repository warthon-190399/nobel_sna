#%%
import pandas as pd
import requests
from bs4 import BeautifulSoup
import difflib
#%%
papers = pd.read_csv("papers.csv")
df2 = pd.read_csv("nobels.csv")
#%%Scraper
# Obtener el contenido HTML de la página
url = "https://ideas.repec.org/f/pgo601.html"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

#working papers
papers_section = soup.find("a", attrs={"name": "papers"}).find_next("ol", class_="list-group")

papers_list = papers_section.find_all("li", class_="list-group-item")

papers_data = []

for item in papers_list:
    author = item.text.split(",")[0].strip()
    article = item.find("a").text.strip()
    papers_data.append({"Collab": author, "Article": article})

#articulos
articles_section = soup.find("a", attrs={"name": "articles"}).find_next("ol", class_="list-group")

articles_list = articles_section.find_all("li", class_="list-group-item")

for item in articles_list:
    author = item.text.split(",")[0].strip()
    article = item.find("a").text.strip()
    papers_data.append({"Collab": author, "Article": article})

#print
for data in papers_data:
    print("Collab:", data["Collab"])
    print("Article:", data["Article"])
    print("--------")

df = pd.DataFrame(papers_data)
#%%Manejo de datos
df = df[df["Collab"].str.contains("&")]

df = df.assign(Collab = df["Collab"].str.split("&")).explode("Collab")

df = df[~df['Collab'].str.contains("Goldin")] #eliminamos al autor

df.reset_index(drop=True, inplace=True)
#%%Funcion limpiar texto
def limpiar_texto(texto):
    texto_limpio = texto.lower()
    texto_limpio = texto_limpio.replace(".", "")
    texto_limpio = " ".join(texto_limpio.split())
    return texto_limpio.strip()

df["Collab"] = df["Collab"].apply(limpiar_texto)
df["Article"] = df["Article"].apply(limpiar_texto)
#%%Mapeo y homogenización nombres autores
nombres_unicos_autores = df["Collab"].unique()

nombres_similares = {}

for nombre in nombres_unicos_autores:
    similares = difflib.get_close_matches(nombre,
                                          nombres_unicos_autores,
                                          n=5,
                                          cutoff=0.80
                                          )

    if len(similares) > 1:
        nombres_similares[nombre] = similares

for nombre, similares in nombres_similares.items():
    print(f"Nombres similares a '{nombre}': {similares}")
#%%
mapeo_autores = {}
for nombre, similares in nombres_similares.items():
    for similar in similares:
        mapeo_autores[similar] = nombre

df["Collab"] = df["Collab"].replace(mapeo_autores)
#%%Mapeo y homogenizacion nombres articulos
nombres_unicos_article = df['Article'].unique()

nombres_similares_article = {}

for nombre in nombres_unicos_article:
    similares = difflib.get_close_matches(nombre,
                                          nombres_unicos_article,
                                          n=5,
                                          cutoff=0.80
                                          )
    if len(similares) > 1:
        nombres_similares_article[nombre] = similares

for nombre, similares in nombres_similares_article.items():
    print(f"Nombres similares a '{nombre}' en Article: {similares}")
#%%
mapeo_articulos = {}
for nombre, similares in nombres_similares.items():
    for similar in similares:
        mapeo_articulos[similar] = nombre

df["Collab"] = df["Collab"].replace(mapeo_articulos)
#%%
df = df.drop_duplicates(subset=['Collab', 'Article'])
#%%
df["laureate"] = "Claudia Goldin"
#%%
df_concat = pd.concat([papers, df], ignore_index=True)
df_concat.to_csv("papers.csv", index=False)
#%%
# Función para encontrar nombres similares
df = pd.read_csv("papers.csv")
def find_similar_names(name, names_list, threshold=0.8):
    similar_names = difflib.get_close_matches(name, names_list, n=5, cutoff=threshold)
    return similar_names

all_collaborators = df['Collab'].unique()
all_laureates = df['laureate'].unique()

#homogenizar
for laureate in all_laureates:
    similar_names = find_similar_names(laureate, all_collaborators)
    if similar_names:
        print(f"Nobel: {laureate}, autores similares: {similar_names}")
        df.loc[df['Collab'].isin(similar_names), 'Collab'] = laureate
    else:
        print(f"No hay nombres similares al nobel: {laureate}")
# %%
df.to_csv("collabs_homo.csv", index=False)
# %%
