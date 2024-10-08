import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import math
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import scipy.stats as stats
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

# file read path 
file_read_path = r'C:\Users\jymbc\Desktop\python files for SMAD\SMAD_online\data'

# read raw protein files 
df_lfq = pd.read_csv(f'{file_read_path}/all_proteins_after_SSnormalization.csv',index_col=0)


def get_co_index(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1 & set2
    return list(intersection)

def knn_imputer(df, neighbors=6):
    '''apply KNN imputation to a dataset'''
    from sklearn.impute import KNNImputer

    # Initialize the KNNImputer
    imputer = KNNImputer(n_neighbors=neighbors)

    imputed_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    return imputed_df


def iterative_imputer(df, maxiteration=10, randomstates=0):
    from sklearn.impute import IterativeImputer

    # Initialize the IterativeImputer
    imputer = IterativeImputer(max_iter=maxiteration, random_state=randomstates)

    # Create a new DataFrame with the imputed values
    imputed_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    return imputed_df


def standardscaler(df):
    # # Initialize the StandardScaler
    scaler = StandardScaler()
    # Fit and transform the data according to rows
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    return scaled_df


def calculate_mean(df):
    # Calculate mean values for each group of columns
    mean_T1 = df.iloc[:, 0:6].mean(axis=1)
    mean_T2 = df.iloc[:, 6:12].mean(axis=1)
    mean_T3 = df.iloc[:, 12:18].mean(axis=1)
    mean_T4 = df.iloc[:, 18:24].mean(axis=1)
    # Create a new DataFrame with the calculated means
    new_df = pd.DataFrame({
        'T1': mean_T1,
        'T2': mean_T2,
        'T3': mean_T3,
        'T4': mean_T4
    })
    return new_df

def filter_row_missings(df, number):
    '''Filter protein features and keep proteins with less than "number" missing values, protein features as index.'''
    # Using isna() to count missing values and keep rows with missing values less than "number"
    df_filtered = df[df.isna().sum(axis=1) <= number]
    return df_filtered

def filter_column_missings(df, value):
    '''Filter samples and keep samples with more than "value" protein identifications, samples as columns.'''
    # Using notna() to count non-missing values and keep columns with non-missing values more than "value"
    df_filtered = df.loc[:, df.notna().sum(axis=0) >= value]
    return df_filtered

# function for oneway_ANOVA analysis
def oneway_ANOVA_Ttest(result):
    pval = []
    for i in range(len(result)):
        aa = result.iloc[i, :6].tolist()
        bb = result.iloc[i, 6:12].tolist()
        cc = result.iloc[i, 12:18].tolist()
        dd = result.iloc[i, 18:24].tolist()
        p_value_anova = f_oneway(aa, bb, cc, dd)[1]
        p_value_ttest = stats.ttest_ind(aa, dd)[1]  # ttest between irradiation and control
        pval.append([result.index[i], p_value_anova, p_value_ttest])
        fl = pd.DataFrame(pval)
        fl.columns = ['protein', 'p_value_anova', 'p_value_ttest']
    return (fl)

# 按行进行标准归一化
def standardscaler_row(df):
    scaler = StandardScaler()
    proname = df.T.columns
    k = df.T
    dfdf_pro = scaler.fit_transform(k)
    # 按行进行标准化后的 dataframe
    dfpro_stdbyrow = pd.DataFrame(dfdf_pro.T, index=df.index, columns=df.columns)
    return dfpro_stdbyrow

# set four treatments 
treatment=['Con']*6 + ['LPS']*6 + ['IL_4']*6 + ['IRD']*6

def getdf(protein_name = '1/sp|B2RXS4|PLXB2_MOUSE',df = df_lfq):
    df_protein_pro = df.loc[protein_name].to_frame()
    df_protein_pro['treatment'] = treatment
    df_protein_pro
    return(df_protein_pro)

import plotly.express as px

def plot_interactive_scatter_box_protein(protein_name='1/sp|B2RXS4|PLXB2_MOUSE', dfdf=df_lfq):
    df = getdf(protein_name, df=dfdf)
    # Ensure 'treatment' is treated as a categorical variable
    # df['treatment'] = df['treatment'].astype('category')

    fig = px.box(df, x='treatment', y=protein_name, points="all",
                 title='Boxplot of protein dysregulation in macrophages',
                 hover_data=['treatment'])

    fig.update_layout(
        yaxis_title="Score",
        width=600,
        height=550,
        title={'font': {'size': 16, 'family': 'Arial'}},
        xaxis_title={'font': {'size': 16, 'family': 'Arial'}},
        # yaxis_title={'font': {'size': 14, 'family': 'Arial'}},
        font={'size': 12, 'family': 'Arial'},
        legend_title={'font': {'size': 14}},
        legend={'font': {'size': 14}},
        # xaxis={'categoryorder': 'category ascending'}
    )
    return fig

#  read significantly dysregulated metabolites

df_meta_ori = pd.read_csv(f'{file_read_path}/metabolome_dysregu_sig_macro_withnames.csv',index_col=0)
df_meta = df_meta_ori.iloc[:,:-1]

def get_medf(metabolite_name = 'His-Pro ',df = df_meta):
    df_metabolite_pro = df.loc[metabolite_name].to_frame()
    df_metabolite_pro['treatment'] = treatment
    df_metabolite_pro
    return(df_metabolite_pro)

def plot_interactive_scatter_box_metabolite(metabolite_name='His-Pro ', dfdf=df_meta):
    df = get_medf(metabolite_name, df=dfdf)
    fig = px.box(df, x='treatment', y=metabolite_name, points="all",
                 title='Boxplot of metabolite dysregulation in macrophages', hover_data=['treatment'])

    fig.update_layout(
        yaxis_title="Score",
        width=600,
        height=550,
        title={'font': {'size': 16, 'family': 'Arial'}},
        xaxis_title={'font': {'size': 16, 'family': 'Arial'}},
        # yaxis_title={'font': {'size': 14, 'family': 'Arial'}},
        font={'size': 14, 'family': 'Arial'},
        legend_title={'font': {'size': 14}},
        legend={'font': {'size': 14}}
    )
    return fig
    
# read significantly dysregulated multiome dataset
df_mean_multi_clustered = pd.read_csv(f'{file_read_path}/df_multi_macrophages_mean_with6clusters.csv',index_col=0)

def get_Kmeans(df, molecular_name):
    kmeans_value = int(df.loc[molecular_name, 'kmeans'])
    return kmeans_value

# get co-regulated molecules dataframe
def select_df_of_same_cluster(df, molecular_name):
    """
    Filters the DataFrame based on the kmeans value of the selected molecular name.
    Parameters:
    - df: The DataFrame to filter.
    - molecular_name: The name of the molecule to select.
    Returns:
    - filtered_df: The filtered DataFrame.
    """
    try:
        # Check if the molecule exists in the DataFrame
        if molecular_name in df.index:
            kmeans_value = df.loc[molecular_name, 'kmeans']
            filtered_df = df[df['kmeans'] == kmeans_value].iloc[:, :4]  # Select first 4 columns
            return filtered_df
        else:
            return pd.DataFrame()  # Return an empty DataFrame if molecule not found
    except Exception as e:
        # General exception handling
        print(f"An error occurred: {e}")
        return pd.DataFrame()

# select_df_of_same_cluster(df_mean_multi_clustered, '1/sp|B2RXS4|PLXB2_MOUSE')

def plot_corr_heatmap_selected_molecules(molecular_name):
    if molecular_name is not None:
        tar_df = select_df_of_same_cluster(df_mean_multi_clustered, molecular_name).T
        if not tar_df.empty:
            # Assuming that the DataFrame has numerical values suitable for a heatmap
            heatmap_fig = px.imshow(
                    tar_df,
                    text_auto=True,                 # Automatically display correlation values on the heatmap
                    aspect="auto",
                    color_continuous_scale='RdBu_r',  # Red-Blue color scale for better visualization of positive and negative correlations
                    zmin=-1.5,                        # Minimum value for the color scale
                    zmax=1.5,                         # Maximum value for the color scale
                    labels=dict(x="Molecule", y="Treatment", color="Z-score"),
                    x=tar_df.columns,
                    y=tar_df.index,
                    title=""
                )
                # Update layout for better aesthetics
            heatmap_fig.update_layout(
                    width=1400,
                    height=400,
                    margin=dict(l=100, r=100, t=100, b=100),
                    xaxis_title=" ",
                    yaxis_title=" ",
                    title_x=0.5,  # Center the title
                    font={'size': 16, 'family': 'Arial'},
                )
            heatmap_fig.update_xaxes(showticklabels=False)
            return heatmap_fig
    # Return an empty figure if no data is available
    return {}

def process_data_for_table(molecular_name):
    """
    Helper function to process the DataFrame for the DataTable.
    """
    filtered_df = select_df_of_same_cluster(df_mean_multi_clustered, molecular_name)
    if not filtered_df.empty:
        # Reset the index to convert it into a column
        reset_df = filtered_df.reset_index()
        # Optionally, rename the index column for clarity
        reset_df.rename(columns={'index': 'Index'}, inplace=True)
        # Convert DataFrame to dictionary format for DataTable
        data = reset_df.to_dict('records')
        # Define table columns including the new index column
        columns = [{"name": col, "id": col} for col in reset_df.columns]
        return columns, data, ""
    else:
        return [], [], f"Selected {molecular_name} is not significantly dysregulated in all treatments!!!"


def extract_unique_gene_names(gene_list):
    """
    Extract unique gene names from a list of input strings, ensuring the output list is the same length as the input list.

    Parameters:
    gene_list (list): List of strings containing multiple gene entries.

    Returns:
    list: List of gene names extracted from the input strings, maintaining the same length as the input list.
    """
    gene_names = set()
    result_genes = []
    pattern = re.compile(r'\|([A-Za-z0-9]+)_[A-Za-z]+\b')

    for entry in gene_list:
        # Find all matches of the pattern in the current string
        matches = pattern.findall(entry)
        selected_gene = None
        if matches:
            for gene in matches:
                if gene not in gene_names:
                    gene_names.add(gene)
                    selected_gene = gene
                    break
            # If no new gene is found, select the first gene in the list
            if not selected_gene:
                selected_gene = matches[0]
        else:
            # If no matches are found, keep the original input string
            selected_gene = entry
        result_genes.append(selected_gene)

    return result_genes

# get gene to pathway reflection
dfdf = df_mean_multi_clustered.copy()
dfdf.index = extract_unique_gene_names(df_mean_multi_clustered.index)
gene_value_dict = dfdf.iloc[:,:4].to_dict(orient='index')
meta_gene = dfdf.iloc[-73:,:]
# list(meta_gene[meta_gene['kmeans']==0].index)

import networkx as nx
import plotly.graph_objs as go
import ast

def plot_network_figure(df, gene_value_dict):
    # creat a nx graph
    G = nx.Graph()
    # add nodes
    pathways = df['Term'].tolist()
    G.add_nodes_from(pathways, type='pathway')
    # add nodes and edge 
    for _, row in df.iterrows():
        pathway = row['Term']
        genes = row['Genes']
        for gene in genes:
            values = gene_value_dict.get(gene, {'Con': 0, 'LPS': 0, 'IL_4': 0, 'IRD': 0})
            G.add_node(gene, type='gene')  # 添加基因节点
            G.add_edge(gene, pathway)  # 添加基因与通路之间的边

    # 定位节点（使用 spring 布局）
    pos = nx.spring_layout(G, k=0.7, seed=42)  # 增大 k 值以增大节点间距
    # 提取边的坐标
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    # 提取节点的坐标、颜色和标签
    node_x = []
    node_y = []
    node_color = []
    node_labels = []
    node_size = []
    node_hovertext = []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if data['type'] == 'gene':
            # 获取基因的值
            values = gene_value_dict.get(node, {'Con': 0, 'LPS': 0, 'IL_4': 0, 'IRD': 0})
            # 构建悬停信息
            hover_text = f"Gene: {node}<br>"
            for condition, value in values.items():
                hover_text += f"{condition}: {value}<br>"
            node_labels.append(node)  # 仅添加基因名称作为标签
            node_hovertext.append(hover_text)  # 添加详细的悬停信息
            node_color.append('blue')  # 设置基因节点颜色
            node_size.append(10)  # 设置基因节点较小的大小
        else:
            node_labels.append(node)
            node_hovertext.append(f"Pathway: {node}")
            node_color.append('orange')  # node color
            node_size.append(30)  # nodesize

    # 创建节点 Trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,  # show gene or pathways
        hovertext=node_hovertext,  # show Z-scores, hovertext
        textposition="bottom center",
        textfont=dict(
            family="Arial",
            size=14,
            color="black"
        ),
        marker=dict(
            color=node_color,
            size=node_size,
            opacity=1,
            line=dict(width=1)
        )
    )
    # 创建 Plotly 图形
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Pathway Network',
                        titlefont_size=16,
                        width=500,  # set width 1200
                        height=500,  # set height 800
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper")],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    return fig


def load_cluster_data(file_path=f'{file_read_path}\Protein_KEGG_enriched_results_clustered_Macrophage.xlsx',
                      cluster_number=0):
    sheet_name = f'Cluster_{cluster_number}'
    # Read the sheet corresponding to the cluster number
    cluster_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Select 'Term' and 'Genes' columns
    select_out_df = cluster_df.loc[:, ['Term', 'Genes']]
    # Convert 'Genes' column to a list using ast.literal_eval
    select_out_df['Genes'] = select_out_df['Genes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    return select_out_df

# test files 
# select_df = load_cluster_data(file_path = f'{file_read_path}\Protein_KEGG_enriched_results_clustered_Macrophage.xlsx',
#                       cluster_number = get_Kmeans(df_mean_multi_clustered, '1/sp|O08529|CAN2_MOUSE'))
#
# # print(select_df['Genes'].apply(type))
# df_only = select_df.iloc[:5,:]
# # df
#
# tsttt = df_only.copy()
#
# tsttt.loc[len(tsttt)] = ['Metabolites', list(meta_gene[meta_gene['kmeans']==get_Kmeans(df_mean_multi_clustered, '1/sp|O08529|CAN2_MOUSE')].index)]

# tsttt

# plot_network_figure(tsttt.iloc[[5]],gene_value_dict)


import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import threading
from dash import dash_table

# Initialize Dash app
app = Dash(__name__)

# Determine valid molecular names present in both df_lfq and df_mean_multi_clustered
valid_proteins = df_lfq.index.intersection(df_mean_multi_clustered.index)
valid_metabolites = df_meta.index.intersection(df_mean_multi_clustered.index)

# Define the layout with tabs
app.layout = html.Div([
    html.H1("Protein and Metabolome Dysregulation on Macrophages"),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Protein Scatter-Box Plot', value='tab-1'),
        dcc.Tab(label='Metabolite Scatter-Box Plot', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])


# Callback to render the content of each tab
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            # First Row: Scatter Plot and DataTable side by side
            html.Div([
                # Left side: Scatter Plot
                html.Div([
                    html.H2("Protein Scatter Plot with Boxplots"),
                    html.Label("Select protein:"),
                    dcc.Dropdown(
                        id='protein-dropdown',
                        options=[{'label': protein, 'value': protein} for protein in df_lfq.index],
                        value=df_lfq.index[0] if len(df_lfq.index) > 0 else None,
                        style={'width': '80%'}
                    ),
                    dcc.Graph(id='scatter-box-plot-protein')
                ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                # Right side: DataTable and Message
                html.Div([
                    html.H3("Co-regulated molecules"),

                    # Message Div
                    html.Div(
                        id='protein-message',
                        children="",  # Initially empty
                        style={'color': 'red', 'marginBottom': '10px'}
                    ),

                    # DataTable
                    dash_table.DataTable(
                        id='protein-data-table',
                        columns=[],  # Columns will be populated by callback
                        data=[],  # Data will be populated by callback
                        page_size=15,  # Display 15 rows per page
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '5px'
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        sort_action='native',  # Enable sorting
                        filter_action='native'  # Enable filtering
                    )
                ], style={'width': '45%', 'display': 'inline-block', 'paddingLeft': '2%'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'}),

            # Second Row: Heatmap spanning full width
            html.Div([
                dcc.Graph(
                    id='protein-heatmap',
                    config={'displayModeBar': False},  # Optional: Hide the mode bar
                    style={'marginTop': '40px'}  # Add top margin for spacing
                )
            ], style={'width': '80%', 'paddingTop': '20px'}),

            # Third Row: Network plot spanning full width
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='protein-network-plot-1',
                        config={'displayModeBar': False},
                        style={'marginTop': '40px'}
                    )
                ], style={'width': '45%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Graph(
                        id='protein-network-plot-2',
                        config={'displayModeBar': False},
                        style={'marginTop': '40px'}
                    )
                ], style={'width': '45%', 'display': 'inline-block', 'paddingLeft': '5%'})
            ], style={'width': '80%', 'paddingTop': '20px'})
        ])
    elif tab == 'tab-2':
        return html.Div([
            # First Row: Scatter Plot and DataTable side by side
            html.Div([
                # Left side: Scatter Plot
                html.Div([
                    html.H2("Metabolite Scatter Plot with Boxplots"),
                    html.Label("Select metabolite:"),
                    dcc.Dropdown(
                        id='metabolite-dropdown',
                        options=[{'label': metabolite, 'value': metabolite} for metabolite in df_meta.index],
                        value=df_meta.index[0] if len(df_meta.index) > 0 else None,
                        style={'width': '80%'}
                    ),
                    dcc.Graph(id='scatter-box-plot-metabolite')
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                # Right side: DataTable and Message
                html.Div([
                    html.H3("Co-regulated molecules"),

                    # Message Div
                    html.Div(
                        id='metabolite-message',
                        children="",  # Initially empty
                        style={'color': 'red', 'marginBottom': '10px'}
                    ),

                    # DataTable
                    dash_table.DataTable(
                        id='metabolite-data-table',
                        columns=[],  # Columns will be populated by callback
                        data=[],  # Data will be populated by callback
                        page_size=15,  # Display 15 rows per page
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '5px'
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        sort_action='native',  # Enable sorting
                        filter_action='native'  # Enable filtering
                    )
                ], style={'width': '38%', 'display': 'inline-block', 'paddingLeft': '2%'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'}),

            # Second Row: Heatmap spanning full width
            html.Div([
                dcc.Graph(
                    id='metabolite-heatmap',
                    config={'displayModeBar': False},  # Optional: Hide the mode bar
                    style={'marginTop': '40px'}  # Add top margin for spacing
                )
            ], style={'width': '100%', 'paddingTop': '20px'}),

            # Third Row: Network Plots spanning full width
            html.Div([
                # First Network Plot
                html.Div([
                    dcc.Graph(
                        id='metabolite-network-plot-1',
                        config={'displayModeBar': False},
                        style={'marginTop': '40px'}
                    )
                ], style={'width': '45%', 'display': 'inline-block'}),

                # Second Network Plot
                html.Div([
                    dcc.Graph(
                        id='metabolite-network-plot-2',
                        config={'displayModeBar': False},
                        style={'marginTop': '40px'}
                    )
                ], style={'width': '40%', 'display': 'inline-block', 'paddingLeft': '5%'})
            ], style={'width': '100%', 'paddingTop': '20px'})
        ])


# Callback to update the protein scatter-box plot
@app.callback(
    Output('scatter-box-plot-protein', 'figure'),
    [Input('protein-dropdown', 'value')]
)
def update_scatterbox_pro_plot(protein_name):
    if protein_name is not None:
        return plot_interactive_scatter_box_protein(protein_name=protein_name, dfdf=df_lfq)
    else:
        # Return an empty figure or a placeholder
        return {}


# Callback to update the metabolite scatter-box plot
@app.callback(
    Output('scatter-box-plot-metabolite', 'figure'),
    [Input('metabolite-dropdown', 'value')]
)
def update_scatterbox_meta_plot(selected_metabolite):
    if selected_metabolite is not None:
        return plot_interactive_scatter_box_metabolite(metabolite_name=selected_metabolite, dfdf=df_meta)
    else:
        # Return an empty figure or a placeholder
        return {}


# Callback to update the protein DataTable and Message
@app.callback(
    [Output('protein-data-table', 'columns'),
     Output('protein-data-table', 'data'),
     Output('protein-message', 'children')],
    [Input('protein-dropdown', 'value')]
)
def update_protein_data_table(protein_name):
    if protein_name is not None:
        return process_data_for_table(protein_name)
    else:
        return [], [], ""


# Callback to update the metabolite DataTable and Message
@app.callback(
    [Output('metabolite-data-table', 'columns'),
     Output('metabolite-data-table', 'data'),
     Output('metabolite-message', 'children')],
    [Input('metabolite-dropdown', 'value')]
)
def update_metabolite_data_table(metabolite_name):
    if metabolite_name is not None:
        return process_data_for_table(metabolite_name)
    else:
        return [], [], ""

# Callback to update the Protein Heatmap
@app.callback(
    Output('protein-heatmap', 'figure'),
    [Input('protein-dropdown', 'value')]
)
def update_protein_heatmap(protein_name):
    if protein_name is not None:
        heatmap_fig = plot_corr_heatmap_selected_molecules(protein_name)
        return heatmap_fig
    else:
        # Return an empty figure or a placeholder
        return {}


# Callback to update the Protein Network Plot
@app.callback(
    [Output('protein-network-plot-1', 'figure'),
     Output('protein-network-plot-2', 'figure')],
    [Input('protein-dropdown', 'value')]
)

def update_protein_network_plot(protein_name):
    if protein_name is not None:
        file_path = f'{file_read_path}\Protein_KEGG_enriched_results_clustered_Macrophage.xlsx'
        select_df_for_pro_network = load_cluster_data(file_path, cluster_number = get_Kmeans(df_mean_multi_clustered, protein_name)).iloc[:5,:]
        mol_network = select_df_for_pro_network.copy()
        mol_network.loc[len(mol_network)] = ['Metabolites', list(meta_gene[meta_gene['kmeans']==get_Kmeans(df_mean_multi_clustered, protein_name)].index)]
        select_df_for_meta_network = mol_network.iloc[[5]]
        network_fig1 = plot_network_figure(select_df_for_pro_network,gene_value_dict)
        network_fig2 = plot_network_figure(select_df_for_meta_network,gene_value_dict)
        return network_fig1,network_fig2
    else:
        # Return an empty figure or a placeholder
        return {}

# Callback to update the Metabolite Heatmap
@app.callback(
    Output('metabolite-heatmap', 'figure'),
    [Input('metabolite-dropdown', 'value')]
)
def update_metabolite_heatmap(metabolite_name):
    if metabolite_name is not None:
        heatmap_fig = plot_corr_heatmap_selected_molecules(metabolite_name)
        return heatmap_fig
    else:
        # Return an empty figure or a placeholder
        return {}

# Callback to update the Metabolite Network Plots
@app.callback(
    [Output('metabolite-network-plot-1', 'figure'),
     Output('metabolite-network-plot-2', 'figure')],
    [Input('metabolite-dropdown', 'value')]
)

def update_metabolite_network_plot(metabolite_name):
    if metabolite_name is not None:
        file_path = f'{file_read_path}\Protein_KEGG_enriched_results_clustered_Macrophage.xlsx'
        select_df_for_pro_network = load_cluster_data(file_path, cluster_number = get_Kmeans(df_mean_multi_clustered, metabolite_name)).iloc[:5,:]
        mol_network = select_df_for_pro_network.copy()
        mol_network.loc[len(mol_network)] = ['Metabolites', list(meta_gene[meta_gene['kmeans']==get_Kmeans(df_mean_multi_clustered, metabolite_name)].index)]
        select_df_for_meta_network = mol_network.iloc[[5]]
        network_fig1 = plot_network_figure(select_df_for_pro_network,gene_value_dict)
        network_fig2 = plot_network_figure(select_df_for_meta_network,gene_value_dict)
        return network_fig1,network_fig2
    else:
        # Return an empty figure or a placeholder
        return {}

# Expose the server variable for deployments
server = app.server
# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=False, port=8053)

