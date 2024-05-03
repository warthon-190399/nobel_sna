import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from collections import Counter
import community

random_seed = 42

st.set_page_config(page_title="Análisis de Redes Sociales",
                   page_icon=":chart_with_upwards_trend:",
                   layout="wide",
                   initial_sidebar_state="expanded"
                  )

def universities_graph(df):
    G = nx.Graph()

    for index, row in df.iterrows():
        laureate = row["laureate"]
        alma_mater = row["alma_mater"]
        teaching = row["teaching"]
        country = row["country"]
        
        G.add_node(laureate, type="laureate")
        
        G.add_node(alma_mater, type="university")
    
        G.add_edge(laureate, alma_mater, relation="studied_at")
        
        G.add_node(teaching, type="university")
        
        G.add_edge(laureate, teaching, relation="taught_at")

    pos = nx.spring_layout(G, k=0.3)

    node_size = [v * 4 for v in dict(G.degree()).values()]

    return G, pos, node_size

def collabs_graph(df):
    G = nx.Graph()

    for index, row in df.iterrows():
        laureate = row["laureate"]
        collab = row["Collab"]

        G.add_node(laureate, type="laureate")

        G.add_node(collab, type="co-autor")

        G.add_edge(laureate, collab, relation="article redaction")

    G = nx.Graph([(u, v) for u, v in G.edges() if G.degree(u) > 1 and G.degree(v) > 1])

    pos = nx.spring_layout(G, k=0.4)

    # Escala logarítmica al tamaño de los nodos
    node_size = [np.log(v + 1) * 10 for v in dict(G.degree()).values()]

    return G, pos, node_size

def nodes_traces_generator(G, pos, node_size):

    #creamos ejes
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    #creamos los trazos
    edge_trace = go.Scatter(x=edge_x,
                            y=edge_y,
                            line=dict(width=0.5, color="#888"),
                            opacity=0.8,
                            hoverinfo="none",
                            mode="lines"
                            )
    #creamos nodos
    node_x = []
    node_y = []

    for x, y in pos.values():
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(x=node_x,
                                y=node_y,
                                mode="markers",
                                hoverinfo="text",
                                marker=dict(
                                        showscale=True,
                                        colorscale="Inferno",
                                        opacity=0.9,
                                        size=node_size, 
                                        colorbar=dict(thickness=15, title="Conexiones Nodos", xanchor="left", titleside="right")
                                    ))

    # Asigna el tamaño de los nodos según el grado con escala logarítmica
    node_trace.marker.color = node_size
    node_trace.text = list(G.nodes())
    
    return edge_x, edge_y, edge_trace, node_trace

df1 = pd.read_csv("universities.csv")
df2 = pd.read_csv("collabs_homo.csv")

logo = "logoPulseNegro.png"
st.sidebar.image(logo)
st.sidebar.title("Tejiendo Conexiones: Análisis de Redes de Premios Nobel de Economía")
select_option = st.sidebar.selectbox("Selecciona el tipo de Análisis de Redes:", ("Colaboraciones Académicas", "Alma Maters de los Nobeles"))

if select_option == "Alma Maters de los Nobeles":
    st.title("Red de Instituciones Educativas de los Laureados")
    G, pos, node_size = universities_graph(df1)
    edge_x, edge_y, edge_trace, node_trace = nodes_traces_generator(G, pos, node_size)

elif select_option == "Colaboraciones Académicas":
    st.title("Sociograma de Publicaciones Académicas")
    G, pos, node_size = collabs_graph(df2)
    edge_x, edge_y, edge_trace, node_trace = nodes_traces_generator(G, pos, node_size)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(showlegend=False,
                                 hovermode="closest",
                                 margin=dict(b=0, l=0, r=0, t=0),
                                 xaxis=dict(showgrid=False,
                                            zeroline=False,
                                            showticklabels=False
                                            ),
                                 yaxis=dict(showgrid=False,
                                            zeroline=False,
                                            showticklabels=False
                                            )
                                 )
                )
st.plotly_chart(fig)

indicadores = ["Grados de la Red", "Centralidad de Intermediación", "Centralidad de Cercanía",
               "Centralidad de Vecindad", "Diámetro de la Red", "Radio de la Red",
               "Coeficiente de Asortatividad"]

indicadores_selec = st.sidebar.multiselect("Indicadores:",indicadores)

roles = ["Bridges", "Hubs", "Outliers"]

roles_selec = st.sidebar.multiselect("Seleccione el Rol:", roles)

detec_graf = ["Comunidades detectadas en la red"]
detec = st.sidebar.multiselect("Ver Red con Detección de Comunidades:", detec_graf)

#Grados de la red
degrees = G.degree()
degrees_counter = Counter(dict(degrees))
top_nodes = degrees_counter.most_common(10)  #Necesitamos solo los 10 mas grandes
nodes, degrees = zip(*top_nodes)
df = pd.DataFrame({
    'Nodo': nodes,
    'Grado': degrees
    })

fig_degree = px.bar(df,
                    x='Nodo',
                    y='Grado',
                    title='Top 10 Nodos por Grado',
                    labels={'Nodo': 'Nodo', 'Grado': 'Grado'},
                    template="plotly_dark",
                    opacity = 0.5
                    )
fig_degree.update_layout(xaxis={'categoryorder': 'total descending'})

#Centralidad de Intermediacion
betweenness_centrality = nx.betweenness_centrality(G)
top_nodes = sorted(betweenness_centrality.items(),
                   key = lambda x:x[1],
                   reverse=True
                   )[:10]

nodes, centralities = zip(*top_nodes)
df = pd.DataFrame({
    'Nodo': nodes,
    'Centralidad de Intermediación': centralities
    })
df = df[::-1]
fig_ci = px.bar(df,
                x='Centralidad de Intermediación',
                y='Nodo',
                orientation='h',
                title='Top 10 Nodos por Centralidad de Intermediación',
                labels={'Nodo': 'Nodo', 'Centralidad de Intermediación': 'Centralidad de Intermediación'},
                template="plotly_dark",
                color='Centralidad de Intermediación',
                color_continuous_scale='Redor',
                opacity = 0.4
                )

#Centralidad de Cercanía
closeness_centrality = nx.closeness_centrality(G)

top_nodes = sorted(closeness_centrality.items(),
                   key=lambda x: x[1],
                   reverse=True)[:10]

nodes, centralities = zip(*top_nodes)
df = pd.DataFrame({
    "Nodo":nodes,
    "Centralidad de Cercanía": centralities})
df = df[::-1]

fig_cc = px.bar(df,
                x="Centralidad de Cercanía",
                y="Nodo",
                orientation="h",
                title="Top 10 Nodos por Centralidad de Cercanía",
                labels={"Nodo": "Nodo", "Centralidad de Cercanía": "Centralidad de Cercanía"},
                color="Centralidad de Cercanía",
                template="plotly_dark",
                color_continuous_scale="Teal",
                opacity = 0.6
                )

#Centralidad de Vecindad:
degree_centrality = nx.degree_centrality(G)


top_nodes_centrality = sorted(degree_centrality.items(),
                              key=lambda x: x[1],
                              reverse=True)[:10]

nodes_closeness, centralities_closeness = zip(*top_nodes_centrality)

df_centrality = pd.DataFrame({
    'Nodo': nodes_closeness,
    'Centralidad de Vecindad': centralities_closeness
    })
df_centrality = df_centrality[::-1]

fig_closeness = px.bar(df_centrality, 
                        y='Nodo', 
                        x='Centralidad de Vecindad', 
                        orientation='h',
                        title='Top 10 Nodos por Centralidad de Vecindad',
                        labels={'Nodo': 'Nodo', 'Centralidad de Vecindad': 'Centralidad de Vecindad'},
                        template="plotly_dark",
                        color='Centralidad de Vecindad',  # Utilizar la variable de centralidad como color
                        color_continuous_scale='Mint',
                        opacity=0.7
                        )
try:
    #Diametro de la red
    diameter = nx.diameter(G)
    #Radio de la red
    radius = nx.radius(G)
    #Coeficiente de asortatividad:
    asort = nx.degree_assortativity_coefficient(G)
except nx.NetworkXError as e:
    diameter = None
    radius = None
    asort = None
    st.warning("El grafo no está completamente conectado. No se podrán calcular algunas métricas.")

#Analisis de roles:
bridges = list(nx.bridges(G))
hubs = [node for node, degree in G.degree() if degree > 2 * (len(G) - 1) / len(G)] #medida que indica nodos más grandes que la media
outliers = [node for node, degree in G.degree() if degree == 2] #fijamos nodos minimos como aquellos con 1 grado

max_length = max(len(bridges), len(hubs), len(outliers)) #fijamos una sola longitud (la mas grande)

bridges += [None] * (max_length - len(bridges))
hubs += [None] * (max_length - len(hubs))
outliers += [None] * (max_length - len(outliers))

#consolidamos el dataframe
df_roles = pd.DataFrame({
    'Bridges':bridges,
    'Hubs':hubs,
    'Outliers':outliers
})

#Deteccion de comunidades

# Obtén las comunidades utilizando el algoritmo de Louvain
partition = community.best_partition(G, random_state=np.random.RandomState(random_seed)) #usamos la semilla para evitar diferentes comunidades

# Asignar colores a nodos según la comunidad a la que pertenecen
node_colors = [partition[node] for node in G.nodes()]

fig_community = go.Figure(data=[
        edge_trace,
        go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode="markers",
            hoverinfo="text",
            marker=dict(
                color=node_colors,
                showscale=True,
                colorscale="rainbow",
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title="Comunidad",
                    xanchor="left",
                    titleside="right"
                )
            ),
            text=list(G.nodes())  # Asegúrate de incluir el texto para el hover
        )
    ],
    layout=go.Layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
)

selected_communities = st.sidebar.selectbox("Selecciona la Comunidad:", sorted(set(partition.values())))
# Aseguramos que sea una lista
if not isinstance(selected_communities, list):
    selected_communities = [selected_communities]

# Filtrar nodos por comunidades seleccionadas
selected_nodes = [node for node, comm_id in partition.items() if comm_id in selected_communities]


if "Grados de la Red" in indicadores_selec:
    st.subheader("Grados de la Red")
    st.write("Los grados de la red en el ámbito académico de la economía indican cuántos enlaces tiene un centro académico o un investigador con otros académicos dentro de su campo. Es una medida de cuántas conexiones tiene un nodo, lo que puede reflejar su influencia, colaboración y participación en la comunidad académica del análisis económico. Los nodos con un alto grado de red suelen tener una amplia gama de conexiones y pueden ser considerados como figuras centrales en su campo.")
    st.plotly_chart(fig_degree)
if "Centralidad de Intermediación" in indicadores_selec:
    st.subheader("Centralidad de Intermediación")
    st.write("La centralidad de intermediación en el ámbito académico de la economía evalúa qué tan crucial es un investigador o universidad como puente entre otros investigadores dentro de su campo. Identifica a aquellos nodos que actúan como enlaces importantes en la red de colaboración, facilitando el flujo de información y la difusión de conocimientos entre distintos grupos de investigación.")
    st.plotly_chart(fig_ci)
if "Centralidad de Cercanía" in indicadores_selec:
    st.subheader("Centralidad de Cercanía")
    st.write("La centralidad de cercanía evalúa qué tan cerca está un nodo de todos los demás en una red. Cuanto más corto sea el camino promedio desde un nodo hacia los demás, mayor será su centralidad de cercanía. Esta medida destaca los nodos que pueden influir rápidamente en toda la red, siendo útil para identificar puntos estratégicos.")
    st.plotly_chart(fig_cc)
if "Centralidad de Vecindad" in indicadores_selec:
    st.subheader("Centralidad de Vecindad")
    st.write("La centralidad de vecindad en el análisis del mundo académico de la economía evalúa la importancia de un investigador basada en su conexión con otros investigadores destacados en su campo. Identifica a los investigadores clave cuyo trabajo y colaboraciones tienen un impacto significativo en el desarrollo del conocimiento económico.")
    st.plotly_chart(fig_closeness)
if "Diámetro de la Red" in indicadores_selec:
    st.subheader("Diámetro de la Red")
    st.write("El diámetro de la red en economía es la máxima distancia entre dos investigadores dentro de la red. Mide la extensión máxima de conexión entre investigadores, lo que refleja la cohesión y accesibilidad de la red académica. Un diámetro corto sugiere una red compacta y fácilmente conectada, mientras que uno largo puede indicar dispersión o fragmentación.")
    if diameter is not None:
        st.metric(label="Diámetro de la Red",
                  value=f"{diameter:.2f}"
                  )
        st.write("El diámetro de 9 indica que la máxima distancia entre cualquier par de nodos en la red es de 9 pasos. Esta medida sugiere que la red tiene una extensión considerable y que la difusión de información puede tardar más en llegar desde un extremo de la red hasta el otro.")
    else:
        st.warning("El grafo no está completamente conectado. No se puede calcular el diámetro de la red.")
if "Radio de la Red" in indicadores_selec:
    st.subheader("Radio de la Red")
    st.write("El radio de la red en economía es la distancia más corta desde un nodo central hasta el nodo más alejado en la red. Mide la accesibilidad y la eficiencia de la difusión de información dentro de la comunidad académica. Un radio corto indica una red densa y conectada, mientras que uno largo puede señalar dispersión o una estructura menos cohesiva.")
    if radius is not None:    
        st.metric(label="Radio de la Red",
            value=f"{radius:.2f}"
                )
        st.write("El radio de 5 indica que la distancia más corta desde cualquier nodo central hasta el nodo más lejano es de 5 pasos. Esto sugiere que la red tiene una buena accesibilidad y que la difusión de información puede ocurrir eficientemente desde cualquier nodo central hacia otros nodos.")
    else:
        st.warning("El grafo no está completamente conectado. No se puede calcular el radio de la red.")
if "Coeficiente de Asortatividad" in indicadores_selec:
    st.subheader("Coeficiente de Asortatividad")
    st.write("El coeficiente de asortatividad en economía evalúa si los investigadores tienden a conectarse con otros de un nivel similar de influencia académica. Un valor positivo indica que los nodos de alto grado se conectan entre sí, mientras que un valor negativo sugiere conexiones entre nodos de diferentes niveles de influencia.")
    if asort is not None:
        st.metric(label="Coeficiente de Asortatividad",
            value=f"{asort:.2f}"
                )
        st.write("Un coeficiente de asortatividad de -0.45 en las redes de Nobel y colaboradores sugiere una tendencia hacia la heterogeneidad en las conexiones. En este contexto, indica que los nodos con diferentes niveles de reconocimiento (representados por los premios Nobel y colaboradores) tienden a conectarse entre sí. Esto puede reflejar una diversidad de colaboraciones entre individuos con diferentes niveles de logros o especialidades en el campo académico.")
    else:
        st.warning("El grafo no está completamente conectado. No se puede calcular el coeficiente de asortatividad.")
if "Outliers" in roles_selec:
    st.subheader("Nodos Outliers")
    st.write("Nodos con conexiones mínimas (Outliers):")
    st.write(df_roles["Outliers"])
if "Hubs" in roles_selec:
    st.subheader("Nodos Centrales (Hubs)")
    st.write("Nodos que actúan como líderes de comunidad:")
    st.write(df_roles["Hubs"])
    st.divider()
if "Bridges" in roles_selec:
    st.subheader("Nodos Puente (Bridges)")
    st.write("Nodos que actúan como puentes entre partes diferentes de la red:")
    st.write(df_roles["Bridges"])
    st.divider()

if "Comunidades detectadas en la red" in detec:
    st.subheader("Comunidades detectadas en la Red")
    st.write("La detección de comunidades consiste en identificar grupos de nodos en una red que están más estrechamente conectados entre sí que con el resto de la red. Estos grupos de nodos forman comunidades o subgrupos dentro de la red más amplia. El objetivo principal de este análisis es revelar la estructura interna y la organización de la red, destacando las relaciones más fuertes y significativas entre sus miembros.")
    st.plotly_chart(fig_community)

#Mostrar tabla con nodos ordenados por tamaño
    if selected_nodes:
        node_sizes = {node: G.degree[node] for node in selected_nodes}
        sorted_nodes = sorted(node_sizes.items(), key=lambda x: x[1], reverse=True)
        
        st.subheader(f"Comunidad {selected_communities}")
        st.table(sorted_nodes)
    else:
        st.sidebar.info("Selecciona al menos una comunidad en el multiselect.")
with st.expander("Acerca de:",
                     expanded=False
                     ):
        st.write('''
            - :orange[**Realizado por:**] [Tato Warthon](https://github.com/warthon-190399).
            - :orange[**Fuente de datos:**] La información relacionadas a las universidades se realizó extrayendo los datos de [NobelPrize.org](https://www.nobelprize.org/prizes/lists/all-prizes-in-economic-sciences/). La extracción de artículos, papers y colaboradores se realizó utilizando métodos de extracción web de la página [RePEc: Research Papers in Economics](https://ideas.repec.org/).
            - :orange[**Metodología:**] Se recopiló información detallada sobre premios Nobel y sus colaboradores académicos, así como las instituciones educativas asociadas. Se emplearon técnicas de análisis de redes para visualizar las complejas relaciones entre estos individuos y organizaciones. Se utilizaron indicadores como grados, medidas de centralidad y detección de comunidades para comprender la estructura y dinámicas de la red.
                '''
                 )

