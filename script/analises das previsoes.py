import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import seaborn as sns
import subprocess
import tempfile
import argparse
import skill_metrics as sm
from matplotlib import rcParams
import matplotlib.lines as mlines
from datetime import timedelta

def pivot (nome_salvar, tipo_df = 'normal'):
    path = r"C:\Users\ferna\OneDrive\Documentos\mestrado\Modelos\Base de dados novas\tabelas csv"
    df = pd.read_csv(f"{path}/{tipo_df}_teste.csv")
    
    if tipo_df == 'normal':
        x = pd.pivot_table(df, index = ['Modelo', 'Horizonte'])
    else:
        x = pd.pivot_table(df, index = ['Modelo', 'Horizonte'], columns=['Condição'])
        
    x.to_csv(path + f"/{nome_salvar}.csv")
    return print("Salvo em:" + path + f"/{nome_salvar}.csv")

path_save = r"C:\Users\ferna\OneDrive\Documentos\mestrado\Modelos\Analises"

#Planilha normal
df = pd.read_csv(r"C:\Users\ferna\OneDrive\Documentos\mestrado\Modelos\Base de dados novas\tabelas csv\df_teste.csv", index_col = 0)
df = df.drop(df.loc[df['P1'].isna() == True].index)
df = df.drop(df.loc[df['Observado'] <= 0].index)
df.index = pd.to_datetime(df.index)
df['Hour'] = df.index.hour
df= df.rename(columns = {'P1':'P_1',
                         'P2':'P_2',
                         'P3':'P_3',})

#Planilha de Erro
df_erro = pd.read_csv(r"C:\Users\ferna\OneDrive\Documentos\mestrado\Modelos\Base de dados novas\tabelas csv\normal_teste.csv", index_col = 0)
df_erro['Modelo'].loc[0] = 'P_1'
df_erro['Modelo'].loc[9] = 'P_2'
df_erro['Modelo'].loc[18] = 'P_3'

# Tabelas pivotadas
for tabela in ['zenith']:
    pivot(f"{tabela}_pivot", tabela)

#Configuração de gráficos
markers_dict = {  
                'P': {'labelColor': 'k', 'Symbol': '+', 'Size': 12, 'FaceColor': 'r', 'EdgeColor': 'r', 'style':':'},
                'EDLSTM P': {'labelColor': 'k', 'Symbol': 'o', 'Size': 12, 'FaceColor': 'g', 'EdgeColor': 'g', 'style':'-'},
                'LSTM P': {'labelColor': 'k', 'Symbol': 'o', 'Size': 12, 'FaceColor': 'w', 'EdgeColor': 'g', 'style':':'},
                'EDLSTM PAZ': {'labelColor': 'k', 'Symbol': 's', 'Size': 12, 'FaceColor': 'b', 'EdgeColor': 'b', 'style':'-'},
                'LSTM PAZ': {'labelColor': 'k', 'Symbol': 's', 'Size': 12, 'FaceColor': 'w', 'EdgeColor': 'b', 'style':':'},
                'EDLSTM ITVUAZD': {'labelColor': 'k', 'Symbol': 'v', 'Size': 12, 'FaceColor': "m", 'EdgeColor': "m", 'style':'-'},
                'LSTM ITVUAZD': {'labelColor': 'k', 'Symbol': 'v', 'Size': 12, 'FaceColor': 'w', 'EdgeColor': "m", 'style':':'},
                'EDLSTM ITVUAZ': {'labelColor': 'k', 'Symbol': 'v', 'Size': 12, 'FaceColor': "cyan", 'EdgeColor': "cyan", 'style':'-'},
                'LSTM ITVUAZ': {'labelColor': 'k', 'Symbol': 'v', 'Size': 12, 'FaceColor': 'w', 'EdgeColor': "cyan", 'style':':'},
                'Observado': {'labelColor': 'k','Symbol': 'h', 'Size': 12,'FaceColor': 'black', 'EdgeColor': 'black', 'style':'-'}, 
               }
COLS_COR = {
    'grid': '#8a8a8a',
    'tick_labels': '#000000',
    'title': '#000000'
}
COLS_STD = {
    'grid': '#8a8a8a',
    'tick_labels': '#000000',
    'ticks': '#8a8a8a',
    'title': '#000000'
}
STYLES_RMS = {
    'color': '#8a8a8a',
    'linestyle': '--'
}
STYLES_RMS = {
    'color': '#8a8a8a',
    'linestyle': '--'
}

#==============
#   Lineplot
#==============
"""   
for alvo in ['Hour', 'Month']:
    for d in ['1','2','3']:
        fig, axes = plt.subplots(figsize=(16, 8))
        
        modelo = list(filter(lambda a: d in a, df.columns))
        modelo.append('Observado')
        
        for m in modelo:
            x = df[[m,alvo]].dropna()
            
            list_info=[]
            for h in df[alvo].unique():
                dict_info = {
                    'Alvo':h,
                    'Media':x[m].loc[x[alvo] == h].mean(),
                    }
                list_info.append(dict_info)
                
            df_info = pd.DataFrame(list_info)
            df_info = df_info.sort_values('Alvo')    
    
            marker = markers_dict.get(m.replace(f'_{d}','').replace('_',' '))
            line = axes.plot(df_info['Alvo'], df_info['Media'], color=marker['EdgeColor'], 
                              linestyle=marker['style'], label=m.replace(f'_{d}','').replace('_',' '), 
                              marker=marker['Symbol'], markersize=marker['Size'], markerfacecolor=marker['FaceColor'])
            
        axes.legend(loc='best', fontsize = 12)
        if alvo == 'Hour':
            axes.set_xlabel("Hora", fontsize = 12)
        else:
            axes.set_xlabel("Mês")
        axes.set_ylabel("Média da Potência Prevista", fontsize = 12)
        plt.show()
        fig.savefig(rf"{path_save}\{d}\lineplot_medias_{alvo}.png", dpi = 600, bbox_inches='tight', facecolor='w')
"""
#==============================
#           Boxplot
#==============================
for d in ['1','2','3']:
    fig, axes = plt.subplots(figsize=(16, 8))
    
    modelo = list(filter(lambda a: d in a, df.columns))
    modelo = list(filter(lambda a: "EDLSTM" in a, modelo))
    modelo.append('Observado')
    modelo = sorted(modelo)
    
    data = []
    pos_list = []
    colors = []
    edges = []
    pos = 1
    
    for h in sorted(df['Hour'].unique()):
        if pos != 1:
            pos+=1
        for m in modelo:
            x = df[m].loc[(df['Hour'] == h)]
            x = x.dropna()
            data.append(x)
            pos_list.append(pos)
            pos += 1
            
            colors.append(markers_dict[f"{m.replace(f'_{d}', '').replace('_',' ')}"]['FaceColor'])
            edges.append(markers_dict[f"{m.replace(f'_{d}', '').replace('_',' ')}"]['EdgeColor'])  
   
    flierprops = dict(marker='o', markerfacecolor='black', markersize = 4)
    
    bplot = axes.boxplot(data,
                         patch_artist=True,
                         positions=pos_list,
                         widths=0.6,
                         flierprops=flierprops)
    
    for patch, color, edge in zip(bplot['boxes'], colors, edges):
        patch.set_facecolor(color)
        patch.set_edgecolor(edge)
        
  
    for patch, median, color, edge in zip(bplot['boxes'], bplot['medians'], colors, edges):
        patch.set_facecolor(color)
        patch.set_edgecolor(edge)
        if color == 'black':
            median.set_color('white')
        else:
            median.set_color('black')
            
    for i in range(6, max(pos_list), 6):
        axes.axvline(x=i, color='grey', linestyle='--', linewidth=0.7)
        
    axes.set_xticks([3 + 6*i for i in range(len(sorted(df['Hour'].unique())))])
    axes.set_xticklabels(sorted(df['Hour'].unique()))
    axes.set_xlabel("Hora (UTC-3)")
    axes.set_xlim(6, max(pos_list)+1)
    axes.set_ylabel("Potência (kWh/kWp)")

    # Configuração da legenda
    handles = []
    labels = []
    
    for m in modelo:
        linestyle = '-' if markers_dict[m.replace(f'_{d}', '').replace('_',' ')]['FaceColor'] != 'w' else ':'
        handle, = axes.plot([0,0], color=markers_dict[m.replace(f'_{d}', '').replace('_',' ')]['EdgeColor'], linestyle=linestyle)
        handles.append(handle)
        labels.append(m.replace(f'_{d}', '').replace('_', ' '))

    axes.legend(handles, labels,loc="best")
    
    for handle in handles:
        handle.set_visible(False)
    
    fig.savefig(rf"{path_save}\{d}\boxplot.png", dpi = 100, bbox_inches='tight', facecolor='w')

#==============================
#   Lineplot RMSE/Bias/SS
#==============================
        
for alvo in ['Hour']: #'Hour', 'Month'
    for erro in ['pRMSE', 'Skill Score']: #'pRMSE', 'pMBE', 'Skill Score'
        for d in ['1','2','3']:
            fig, axes = plt.subplots(figsize=(16, 8))
            
            modelo = list(filter(lambda a: d in a, df.columns))
            if erro == "Skill Score":
                remove_predict = [f"P_{d}"]
                modelo = [item for item in modelo if item not in remove_predict]

            for m in modelo:
                if m != f'P_{d}':
                    x = df[[m,alvo,'Observado',f'P_{d}']].loc[df['zenith']<70].dropna()
                else:
                    x = df[[m,alvo,'Observado']].loc[df['zenith']<70].dropna()
                
                
                list_info=[]
                for h in df[alvo].unique():
                    y = x.loc[x[alvo]==h]
                    if len(y) != 0:
                        dict_info = {
                            'Alvo':h,
                            'pRMSE':mean_squared_error(y[m], y['Observado'], squared=False)/np.mean(y['Observado']),
                            'pMBE':np.mean(y[m] - y['Observado']) / np.mean(y['Observado']),
                            'Skill Score': 1-(mean_squared_error(y[m], y['Observado'], squared=False)/mean_squared_error(y[f"P_{d}"], y['Observado'], squared=False))
                            }
                        list_info.append(dict_info)
                    
                df_info = pd.DataFrame(list_info)
                df_info = df_info.sort_values('Alvo')    
        
                marker = markers_dict.get(m.replace(f'_{d}','').replace('_',' '))
                line = axes.plot(df_info['Alvo'], df_info[erro], color=marker['EdgeColor'], 
                                  linestyle=marker['style'], label=m.replace(f'_{d}','').replace('_',' '), 
                                  marker=marker['Symbol'], markersize=marker['Size'], markerfacecolor=marker['FaceColor'])
            
            #axes.plot(df_info['Alvo'],[0] * len(df_info[erro]),
            #          linestyle = '--',
            #          color = 'black')
            
            axes.legend(loc='best', fontsize = 12)
            axes.set_xticks(np.linspace(start=df_info['Alvo'].iloc[0], stop=df_info['Alvo'].iloc[-1], num=len(df_info['Alvo'])))
            if alvo == 'Hour':
                axes.set_xlabel("Hora", fontsize = 12)
            else:
                axes.set_xlabel("Mês")
            axes.set_ylabel(f"{erro}", fontsize = 12)
            plt.show()
            fig.savefig(rf"{path_save}\{d}\lineplot_{erro}_{alvo}.png", dpi = 100, bbox_inches='tight', facecolor='w')
         
            
         
#==============================
#   Lineplot RMSE/Bias/SS diferente
#==============================
        
for alvo in ['Hour', 'Month']: #'Hour', 'Month'

    for erro in ['pRMSE', 'pMBE', 'Skill Score']: #'pRMSE', 'pMBE', 'Skill Score'
        fig, axes = plt.subplots(3, figsize=(16, 8*3))   
        l=0
        idd = ['(a)','(b)','(c)']
        
        for d in ['1','2','3']:   
            modelo = list(filter(lambda a: d in a, df.columns))
            
            if erro == "Skill Score":
                remove_predict = [f"P_{d}"]
                modelo = [item for item in modelo if item not in remove_predict]

            for m in modelo:
                if m != f'P_{d}':
                    x = df[[m,alvo,'Observado',f'P_{d}']].loc[df['zenith']<70].dropna()
                else:
                    x = df[[m,alvo,'Observado']].loc[df['zenith']<70].dropna()
                
                list_info=[]
                
                for h in df[alvo].unique():
                    y = x.loc[x[alvo]==h]
                    
                    if len(y) != 0:
                        dict_info = {
                            'Alvo':h,
                            'pRMSE':mean_squared_error(y[m], y['Observado'], squared=False)/np.mean(y['Observado'])*100,
                            'pMBE':np.mean(y[m] - y['Observado']) / np.mean(y['Observado'])*100,
                            'Skill Score': 1-(mean_squared_error(y[m], y['Observado'], squared=False)/mean_squared_error(y[f"P_{d}"], y['Observado'], squared=False))*100
                            }
                        list_info.append(dict_info)
                    
                df_info = pd.DataFrame(list_info)
                df_info = df_info.sort_values('Alvo')    
        
                marker = markers_dict.get(m.replace(f'_{d}','').replace('_',' '))
                line = axes[l].plot(df_info['Alvo'], df_info[erro], color=marker['EdgeColor'], 
                                  linestyle=marker['style'], label=m.replace(f'_{d}','').replace('_',' '), 
                                  marker=marker['Symbol'], markersize=marker['Size'], markerfacecolor=marker['FaceColor'])
                #axs[1].annotate('(b)', xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='center', fontsize = 12)
                #axes.plot(df_info['Alvo'],[0] * len(df_info[erro]),
                #          linestyle = '--',
                #          color = 'black')    
                
            axes[l].annotate(idd[l], xy=(0.5, -0.15), xycoords='axes fraction', ha='center', va='center', fontsize = 12)

            if l == 0:    
                axes[l].legend(loc='best', fontsize = 12)
                
            axes[l].set_xticks(np.linspace(start=df_info['Alvo'].iloc[0], stop=df_info['Alvo'].iloc[-1], num=len(df_info['Alvo'])))
            
            if alvo == 'Hour':
                axes[l].set_xlabel("Hora (UTC-3)", fontsize = 12)
                
            else:
                axes[l].set_xlabel("Mês")
                
            axes[l].set_ylabel(f"{erro} (%)", fontsize = 12)
            l+=1
            
        plt.show()
        fig.savefig(rf"{path_save}\lineplot_{erro}_{alvo}.png", dpi = 100, bbox_inches='tight', facecolor='w')
    
"""
#===================================
#   Taylor por hora de previsão
#===================================

for d in ['1','2','3']:
    fig, axs = plt.subplots(1,2,figsize=(14,8))
    plt.subplots_adjust(hspace=0.5)     
    c=0
    for m in list(filter(lambda a: d in a, df_erro['Modelo'].unique())):
        x = df_erro.loc[(df_erro['Modelo']==m)&(df_erro['Horizonte']==int(d))]
        if c == 0:
            sm.taylor_diagram(axs[0],
                              np.asarray([1, x['Razão std'].values[0]]),
                              np.asarray([0, x['RMSE'].values[0]]),
                              np.asarray([1, x['R2'].values[0]]), 
                              markercolors = {
                                  "face": markers_dict[m.replace(f'_{d}','').replace('_',' ')]["FaceColor"],
                                  "edge": markers_dict[m.replace(f'_{d}','').replace('_',' ')]["EdgeColor"]},
                              markersymbol = markers_dict[m.replace(f'_{d}','').replace('_',' ')]["Symbol"],
                              markersize = markers_dict[m.replace(f'_{d}','').replace('_',' ')]["Size"],
                              #markerLabel = ['obs'],
                              #markerLegend = 'on', 
                              styleOBS = ':', colOBS ="#000000", markerobs = 'o',
                              showlabelsRMS = 'on',
                              colRMS = STYLES_RMS['color'],
                              tickRMSangle = 115,
                              styleRMS = STYLES_RMS['linestyle'],
                              titleRMS = 'off', titleOBS = 'Ref',
                                  colscor = COLS_COR,
                                  colsstd = COLS_STD,
                              colframe='#DDDDDD',
                              labelweight='normal',
                              titlecorshape='linear',
                              axismax = 1.75,
                              tickRMS = [0.25,0.75,1.25]
                              #rincstd = [0,1,2,3,4,5],
                              #tickstd = [0,1,2,3,4,5]
                              )
        
        else:
            sm.taylor_diagram(axs[0],
                              np.asarray([1, x['Razão std'].values[0]]),
                              np.asarray([0, x['RMSE'].values[0]]),
                              np.asarray([1, x['R2'].values[0]]), 
                              markercolors = {
                                  "face": markers_dict[m.replace(f'_{d}','').replace('_',' ')]["FaceColor"],
                                  "edge": markers_dict[m.replace(f'_{d}','').replace('_',' ')]["EdgeColor"]},
                              markersymbol = markers_dict[m.replace(f'_{d}','').replace('_',' ')]["Symbol"],
                              markersize = markers_dict[m.replace(f'_{d}','').replace('_',' ')]["Size"],
                              #markerLabel = ['obs'],
                              #markerLegend = 'on', 
                              styleOBS = ':', colOBS ="#000000", markerobs = 'o',
                              showlabelsRMS = 'on',
                              colRMS = STYLES_RMS['color'],
                              tickRMSangle = 115,
                              styleRMS = STYLES_RMS['linestyle'],
                              titleRMS = 'off', titleOBS = 'Ref',
                                  colscor = COLS_COR,
                                  colsstd = COLS_STD,
                              colframe='#DDDDDD',
                              labelweight='normal',
                              titlecorshape='linear',
                              overlay = 'on')
            
        c+=1

        #RMSE X MBE            

        axs[1].scatter(x['MBE'], x['RMSE'], 
                    edgecolors = markers_dict[m.replace(f'_{d}','').replace('_',' ')]["EdgeColor"],
                    facecolors = markers_dict[m.replace(f'_{d}','').replace('_',' ')]["FaceColor"],
                    marker = markers_dict[m.replace(f'_{d}','').replace('_',' ')]['Symbol'],
                    s = 100)
        #axs[2].annotate(d, (x['Bias'], x['RMSE']), xytext=(0, 5), textcoords='offset points')
        
    axs[1].grid(alpha=.9)
    axs[1].set_xlabel('MBE')
    axs[1].set_ylabel('RMSE')
    lim = round(max([max(df_erro['MBE']), -min(df_erro['MBE'])]),2)
    axs[1].set_xlim(-lim-0.05, lim+0.05)
    #axs[2].set_title('Gráfico de Linhas com Múltiplas Séries')
    
    # create legend in the last subplot
    ax = axs[1]
    #ax.axis('off')
    
    # build legend handles    
    legend_handles = []
    legend_handles.append(mlines.Line2D([], [],
                          color=STYLES_RMS['color'],
                          linestyle=STYLES_RMS['linestyle'],
                          label="RMSE"))
    
    for marker_label, marker_desc in markers_dict.items():
        modelo = list(filter(lambda a: d in a, df_erro['Modelo'].unique()))
        
        if marker_label != "Observado":
            marker = mlines.Line2D([], [], 
                                   marker=marker_desc["Symbol"],
                                   markersize=marker_desc["Size"],
                                   markerfacecolor=marker_desc["FaceColor"],
                                   markeredgecolor=marker_desc["EdgeColor"],
                                   linestyle='None',
                                   label=marker_label)
            legend_handles.append(marker)
            #del marker_label, marker_desc, marker
    
    # create legend and free memory
    ax.legend(handles=legend_handles, loc="best")    
    plt.show()
    fig.savefig(rf"{path_save}\{d}\Taylor_Target_MBERMSE.png", dpi = 600, bbox_inches='tight', facecolor='w')
"""
#===================================
#   Taylor por hora de previsão
#           normalizado (p)
#===================================

for d in ['1','2','3']:
    fig, axs = plt.subplots(1,2,figsize=(14,8))
    plt.subplots_adjust(hspace=0.5)     
    c=0
    for m in list(filter(lambda a: d in a, df_erro['Modelo'].unique())):
        x = df_erro.loc[(df_erro['Modelo']==m)&(df_erro['Horizonte']==int(d))]
        if c == 0:
            sm.taylor_diagram(axs[0],
                              np.asarray([1, x['Razão std'].values[0]]),
                              np.asarray([0, x['pRMSE'].values[0]]),
                              np.asarray([1, x['R2'].values[0]]), 
                              markercolors = {
                                  "face": markers_dict[m.replace(f'_{d}','').replace('_',' ')]["FaceColor"],
                                  "edge": markers_dict[m.replace(f'_{d}','').replace('_',' ')]["EdgeColor"]},
                              markersymbol = markers_dict[m.replace(f'_{d}','').replace('_',' ')]["Symbol"],
                              markersize = markers_dict[m.replace(f'_{d}','').replace('_',' ')]["Size"],
                              #markerLabel = ['obs'],
                              #markerLegend = 'on', 
                              styleOBS = ':', colOBS ="#000000", markerobs = 'o',
                              showlabelsRMS = 'on',
                              colRMS = STYLES_RMS['color'],
                              tickRMSangle = 115,
                              styleRMS = STYLES_RMS['linestyle'],
                              titleRMS = 'off', titleOBS = 'Ref',
                                  colscor = COLS_COR,
                                  colsstd = COLS_STD,
                              colframe='#DDDDDD',
                              labelweight='normal',
                              titlecorshape='linear',
                              axismax = 1.75,
                              tickRMS = [0.25,0.75,1.25]
                              #rincstd = [0,1,2,3,4,5],
                              #tickstd = [0,1,2,3,4,5]
                              )
        
        else:
            sm.taylor_diagram(axs[0],
                              np.asarray([1, x['Razão std'].values[0]]),
                              np.asarray([0, x['pRMSE'].values[0]]),
                              np.asarray([1, x['R2'].values[0]]), 
                              markercolors = {
                                  "face": markers_dict[m.replace(f'_{d}','').replace('_',' ')]["FaceColor"],
                                  "edge": markers_dict[m.replace(f'_{d}','').replace('_',' ')]["EdgeColor"]},
                              markersymbol = markers_dict[m.replace(f'_{d}','').replace('_',' ')]["Symbol"],
                              markersize = markers_dict[m.replace(f'_{d}','').replace('_',' ')]["Size"],
                              #markerLabel = ['obs'],
                              #markerLegend = 'on', 
                              styleOBS = ':', colOBS ="#000000", markerobs = 'o',
                              showlabelsRMS = 'on',
                              colRMS = STYLES_RMS['color'],
                              tickRMSangle = 115,
                              styleRMS = STYLES_RMS['linestyle'],
                              titleRMS = 'off', titleOBS = 'Ref',
                                  colscor = COLS_COR,
                                  colsstd = COLS_STD,
                              colframe='#DDDDDD',
                              labelweight='normal',
                              titlecorshape='linear',
                              overlay = 'on')
            
        c+=1

        #RMSE X MBE            

        axs[1].scatter(x['pMBE'], x['pRMSE'], 
                    edgecolors = markers_dict[m.replace(f'_{d}','').replace('_',' ')]["EdgeColor"],
                    facecolors = markers_dict[m.replace(f'_{d}','').replace('_',' ')]["FaceColor"],
                    marker = markers_dict[m.replace(f'_{d}','').replace('_',' ')]['Symbol'],
                    s = 100)
        #axs[2].annotate(d, (x['Bias'], x['RMSE']), xytext=(0, 5), textcoords='offset points')
        
    axs[1].grid(alpha=.9)
    axs[1].set_xlabel('pMBE')
    axs[1].set_ylabel('pRMSE')
    lim = round(max([max(df_erro['pMBE']), -min(df_erro['pMBE'])]),2)
    axs[1].set_xlim(-lim-0.05, lim+0.05)
    #axs[2].set_title('Gráfico de Linhas com Múltiplas Séries')
    
    # create legend in the last subplot
    ax = axs[1]
    #ax.axis('off')
    
    # build legend handles    
    legend_handles = []
    legend_handles.append(mlines.Line2D([], [],
                          color=STYLES_RMS['color'],
                          linestyle=STYLES_RMS['linestyle'],
                          label="RMSE"))
    
    for marker_label, marker_desc in markers_dict.items():
        modelo = list(filter(lambda a: d in a, df_erro['Modelo'].unique()))
        
        if marker_label != "Observado":
            marker = mlines.Line2D([], [], 
                                   marker=marker_desc["Symbol"],
                                   markersize=marker_desc["Size"],
                                   markerfacecolor=marker_desc["FaceColor"],
                                   markeredgecolor=marker_desc["EdgeColor"],
                                   linestyle='None',
                                   label=marker_label)
            legend_handles.append(marker)
            #del marker_label, marker_desc, marker
    
    # create legend and free memory
    ax.legend(handles=legend_handles, loc="best")    
    
    axs[0].annotate('(a)', xy=(0.5, -0.2), xycoords='axes fraction', ha='center', va='center', fontsize = 12)
    axs[1].annotate('(b)', xy=(0.5, -0.1), xycoords='axes fraction', ha='center', va='center', fontsize = 12)
  # Adicionando 'a' abaixo da primeira imagem
    #axs[1].annotate('b', loc='center', pad=20)
    plt.tight_layout()
    plt.show()    
    fig.savefig(rf"{path_save}\{d}\Taylor_Target_pMBEpRMSE.png", dpi = 100, bbox_inches='tight', facecolor='w')

"""    
#========================
#   Scatter cond céu
#========================
for alvo in ['Condição de céu']:
    fig, axs = plt.subplots(9,3,figsize=(8*9,16*3))
    plt.subplots_adjust(hspace=0.3) 
    
    for d in range(1, 4):
        l = 0
        for m in sorted(list(filter(lambda a: str(d) in a, df.columns))):
            x = df[['Observado', m, 'Condição de céu']]
            x = x.dropna()
            
            # Create regression lines and scatter plot
            reg_line_obs = sns.regplot(
                data=x, x="Observado", y="Observado", line_kws=dict(color="black"), scatter=False, ax=axs[l, d-1]
            )
            reg_line_lstm_exogena = sns.regplot(
                data=x, x="Observado", y=m, line_kws=dict(color="r"), scatter=False, ax=axs[l, d-1]
            )
            scatter_plot = sns.scatterplot(
                data=x, x="Observado", y=m, hue=alvo, ax=axs[l, d-1]
            )
            # Set axis labels
            axs[l, d-1].set_ylabel("Potência Prevista Normalizada", fontsize = 12)
            axs[l, d-1].set_xlabel("Potência Medida Normalizada", fontsize = 12)
            axs[l, d-1].set_title(f"{m}", fontsize = 12)
            
            # Increment counter
            l += 1
        
#================
#   Histograma
#================
for d in range(1, 4):
    modelos = sorted(list(filter(lambda a: str(d) in a, df.columns)))
    edlstm = sorted(list(filter(lambda a: 'EDLSTM' in a, modelos)))
    lstm = sorted([item for item in modelos if item not in edlstm] )
    edlstm.append(f"P_{d}")
    
    for models in [lstm, edlstm]:
        fig, ax = plt.subplots(3,2,figsize = (12,10))
        #plt.subplots_adjust(wspace=0.2)
        l,c = 0,0
        for m in models:
            x = df[['Observado', m]]
            x = x.dropna()
            x['Diferenca'] = x[m] - x['Observado']
            
            sns.histplot(data = x['Diferenca'], kde = True, ax =ax[l,c])
            m=m.replace(f"_{d}", "").replace("_"," ")
            ax[l,c].set_xlabel(f"Diferença entre {m} e Observações", fontsize = 12)
            ax[l,c].set_ylabel("Frequência", fontsize = 12)
            lim_x = max([max(x['Diferenca']), -min(x['Diferenca'])])
            ax[l,c].set_xlim(-lim_x, lim_x)
            
            c += 1
            if c == 2:
                c = 0
                l += 1
        ax[2,1].axis('off')
        plt.tight_layout()
        plt.show()
        #fig.savefig(rf"{path_save}\{d}\histplot.png", dpi = 600, bbox_inches='tight', facecolor='w')
        
# Opções:
#   'Condição de céu'  
#   'Tipo de mes'  
#   'Faixa zenith'
for condicao in ['Condição de céu', 'Tipo de mes', 'Faixa zenith']:
    
    for cond in df[condicao].unique():
        
        for d in range(1, 4):
            modelos = sorted(list(filter(lambda a: str(d) in a, df.columns)))
            edlstm = sorted(list(filter(lambda a: 'EDLSTM' in a, modelos)))
            lstm = sorted([item for item in modelos if item not in edlstm] )
            edlstm.append(f"P_{d}")
            
            for models in [lstm, edlstm]:
                fig, ax = plt.subplots(3,2,figsize = (12,10))
                #plt.subplots_adjust(wspace=0.2)
                l,c = 0,0
                for m in models:
                    x = df[['Observado', m]].loc[df[condicao] == cond]
                    x = x.dropna()
                    x['Diferenca'] = x[m] - x['Observado']
                    
                    sns.histplot(data = x['Diferenca'], kde = True, ax =ax[l,c])
                    m=m.replace(f"_{d}", "").replace("_"," ")
                    ax[l,c].set_xlabel(f"Diferença entre {m} e Observações", fontsize = 12)
                    ax[l,c].set_ylabel("Frequência", fontsize = 12)
                    lim_x = max([max(x['Diferenca']), -min(x['Diferenca'])])
                    ax[l,c].set_xlim(-lim_x, lim_x)
                    
                    c += 1
                    if c == 2:
                        c = 0
                        l += 1
                ax[2,1].axis('off')
                plt.tight_layout()
                plt.show()           
                #fig.savefig(rf"{path_save}\{d}\histplot_{cond}.png", dpi = 600, bbox_inches='tight', facecolor='w')
"""

#===========================
#   Histograma + scatter
#===========================
for d in range(1, 4):
    modelos = sorted(list(filter(lambda a: str(d) in a, df.columns)))
    edlstm = sorted(list(filter(lambda a: 'EDLSTM' in a, modelos)))
    lstm = sorted([item for item in modelos if item not in edlstm] )
    #edlstm.append(f"P_{d}")
    
    for models in [edlstm,lstm]:
        fig, ax = plt.subplots(len(models),2,figsize = (12,16))
        #plt.subplots_adjust(wspace=0.9)
        l = 0
        if models[0][0] == 'L':
            tipo = 'LSTM'
        else:
            tipo = "EDLSTM"
            
        for m in models:
            x = df[['Observado', m]]
            x = x.dropna()
            x['Diferenca'] = x[m] - x['Observado']
            
            sns.histplot(data = x['Diferenca'], 
                         kde = True, 
                         color = markers_dict[m.replace(f'_{d}', '').replace('_',' ')]['EdgeColor'],
                         ax =ax[l,0])
            
            sns.regplot(data=x,
                        x="Observado",
                        y="Observado",
                        line_kws=dict(color="black"), scatter=False, ax=ax[l, 1])
            
            sns.regplot(data=x, 
                        x="Observado", 
                        y=m, 
                        line_kws=dict(color="brown"), scatter=False, ax=ax[l, 1])
            if m[:4] == 'LSTM':
                sns.scatterplot(data = x,
                                x = 'Observado',
                                y = m, 
                                edgecolor=markers_dict[m.replace(f'_{d}', '').replace('_',' ')]['EdgeColor'],
                                facecolor=markers_dict[m.replace(f'_{d}', '').replace('_',' ')]['FaceColor'],
                                ax =ax[l,1])
            else: 
                sns.scatterplot(data = x,
                                x = 'Observado',
                                y = m, 
                                color=markers_dict[m.replace(f'_{d}', '').replace('_',' ')]['EdgeColor'],
                                ax =ax[l,1])
            
            m=m.replace(f"_{d}", "").replace("_"," ")
            #lim_x = max([max(x['Diferenca']), -min(x['Diferenca'])])
            
            ax[l,0].set_xlabel(f"Desvio entre {m} e Observações", fontsize = 12)
            ax[l,0].set_ylabel("Frequência", fontsize = 12)
            ax[l,1].set_xlabel("Observações (kWh/kWp)", fontsize = 12)
            ax[l,1].set_ylabel(f"{m} (kWh/kWp)", fontsize = 12)
            #ax[l,0].set_xlim(-lim_x, lim_x)
            ax[l,0].set_xlim(-0.3, 0.3)
            ax[l,0].set_ylim(0,400)
            
            #ax[l,1].set_ylabel(m, fontsize = 12)
            l += 1

        plt.tight_layout()
        plt.show()
        fig.savefig(rf"{path_save}\{d}\histplot_scatter{tipo}.png", dpi = 100, bbox_inches='tight', facecolor='w')
        
        
"""
# Opções:
#   'Condição de céu'  
#   'Tipo de mes'  
#   'Faixa zenith'
for condicao in ['Condição de céu', 'Tipo de mes', 'Faixa zenith']:
    
    for cond in df[condicao].unique():
        
        for d in range(1, 4):
            modelos = sorted(list(filter(lambda a: str(d) in a, df.columns)))
            edlstm = sorted(list(filter(lambda a: 'EDLSTM' in a, modelos)))
            lstm = sorted([item for item in modelos if item not in edlstm] )
            #edlstm.append(f"P_{d}")
            
            for models in [edlstm]:
                fig, ax = plt.subplots(len(edlstm),2,figsize = (12,16))
                #plt.subplots_adjust(wspace=0.2)
                l = 0
                for m in models:
                    x = df[['Observado', m]].loc[df[condicao] == cond]
                    x = x.dropna()
                    x['Diferenca'] = x[m] - x['Observado']
                    
                    sns.histplot(data = x['Diferenca'], 
                                 kde = True, 
                                 color = markers_dict[m.replace(f'_{d}', '').replace('_',' ')]['EdgeColor'],
                                 ax =ax[l,0])
                    
                    sns.regplot(data=x,
                                x="Observado",
                                y="Observado",
                                line_kws=dict(color="black"), scatter=False, ax=ax[l, 1])
                    
                    sns.regplot(data=x, 
                                x="Observado", 
                                y=m, 
                                line_kws=dict(color="brown"), scatter=False, ax=ax[l, 1])
                    if m[:4] == 'LSTM':
                        sns.scatterplot(data = x,
                                        x = 'Observado',
                                        y = m, 
                                        edgecolor=markers_dict[m.replace(f'_{d}', '').replace('_',' ')]['EdgeColor'],
                                        facecolor=markers_dict[m.replace(f'_{d}', '').replace('_',' ')]['FaceColor'],
                                        ax =ax[l,1])
                    else: 
                        sns.scatterplot(data = x,
                                        x = 'Observado',
                                        y = m, 
                                        color=markers_dict[m.replace(f'_{d}', '').replace('_',' ')]['EdgeColor'],
                                        ax =ax[l,1])
                    
                    m=m.replace(f"_{d}", "").replace("_"," ")
                    #lim_x = max([max(x['Diferenca']), -min(x['Diferenca'])])
                    
                    ax[l,0].set_xlabel(f"Desvio entre {m} e Observações", fontsize = 12)
                    ax[l,0].set_ylabel("Frequência", fontsize = 12)
                    ax[l,1].set_xlabel("Observações (kWh/kWp)", fontsize = 12)
                    ax[l,1].set_ylabel(f"{m} (kWh/kWp)", fontsize = 12)
                    #ax[l,0].set_xlim(-lim_x, lim_x)
                    ax[l,0].set_xlim(-0.3, 0.3)
                    
                    #ax[l,1].set_ylabel(m, fontsize = 12)
                    l += 1
        
                plt.tight_layout()
                plt.show()
                fig.savefig(rf"{path_save}\{d}\histplot_scatter_{cond}_edlstm.png", dpi = 600, bbox_inches='tight', facecolor='w')
"""        
        

# , 'Tipo de mes', 'Faixa zenith'
for condicao in ['Condição de céu', 'Tipo de mes', 'Faixa zenith']:
    
    for cond in df[condicao].unique():
        c = 0
        fig, ax = plt.subplots(len(edlstm),3,figsize = (12,16))
        for d in range(1, 4):
            modelos = sorted(list(filter(lambda a: str(d) in a, df.columns)))
            edlstm = sorted(list(filter(lambda a: 'EDLSTM' in a, modelos)))
            lstm = sorted([item for item in modelos if item not in edlstm] )
            #edlstm.append(f"P_{d}")
            
            for models in [edlstm]:
                
                #plt.subplots_adjust(wspace=0.2)
                l = 0
                for m in models:
                    x = df[['Observado', m]].loc[df[condicao] == cond]
                    x = x.dropna()
                    x['Diferenca'] = x[m] - x['Observado']
                    
                    
                    sns.regplot(data=x,
                                x="Observado",
                                y="Observado",
                                line_kws=dict(color="black"), scatter=False, ax=ax[l, c])
                    
                    sns.regplot(data=x, 
                                x="Observado", 
                                y=m, 
                                line_kws=dict(color="brown"), scatter=False, ax=ax[l, c])
                    if m[:4] == 'LSTM':
                        sns.scatterplot(data = x,
                                        x = 'Observado',
                                        y = m, 
                                        edgecolor=markers_dict[m.replace(f'_{d}', '').replace('_',' ')]['EdgeColor'],
                                        facecolor=markers_dict[m.replace(f'_{d}', '').replace('_',' ')]['FaceColor'],
                                        ax =ax[l,c])
                    else: 
                        sns.scatterplot(data = x,
                                        x = 'Observado',
                                        y = m, 
                                        color=markers_dict[m.replace(f'_{d}', '').replace('_',' ')]['EdgeColor'],
                                        ax =ax[l,c])
                    
                    m=m.replace(f"_{d}", "").replace("_"," ")
                    #lim_x = max([max(x['Diferenca']), -min(x['Diferenca'])])
                    
 
                    ax[l,c].set_xlabel("Observações (kWh/kWp)", fontsize = 12)
                    ax[l,c].set_ylabel(f"{m} (kWh/kWp)", fontsize = 12)
                    #ax[l,0].set_xlim(-lim_x, lim_x)
                    ax[l,c].set_xlim(0.05, 0.42)
                    ax[l,c].set_ylim(0.05, 0.42)
                    #ax[l,1].set_ylabel(m, fontsize = 12)
                    l += 1
            c+=1
        plt.tight_layout()
        plt.show()
        fig.savefig(rf"{path_save}\scatter_{cond}_edlstm.png", dpi = 100, bbox_inches='tight', facecolor='w')
        


#'Condição de céu' , 'Tipo de mes', 'Faixa zenith'
for condicao in ['Tipo de mes', 'Faixa zenith']:
    
    for cond in df[condicao].unique():
        c = 0
        fig, ax = plt.subplots(len(edlstm),3,figsize = (12,16))
        for d in range(1, 4):
            modelos = sorted(list(filter(lambda a: str(d) in a, df.columns)))
            edlstm = sorted(list(filter(lambda a: 'EDLSTM' in a, modelos)))
            lstm = sorted([item for item in modelos if item not in edlstm] )
            #edlstm.append(f"P_{d}")
            
            for models in [edlstm]:
                
                #plt.subplots_adjust(wspace=0.2)
                l = 0
                for m in models:
                    x = df[['Observado', m]].loc[df[condicao] == cond]
                    x = x.dropna()
                    x['Diferenca'] = x[m] - x['Observado']
                    
                    
                    sns.histplot(data = x['Diferenca'], 
                                 kde = True, 
                                 color = markers_dict[m.replace(f'_{d}', '').replace('_',' ')]['EdgeColor'],
                                 ax =ax[l,c])
                    
                    m=m.replace(f"_{d}", "").replace("_"," ")
 
                    ax[l,c].set_xlabel(f"Desvio entre {m} e Observações", fontsize = 12)
                    ax[l,c].set_ylabel("Frequência", fontsize = 12)
                    ax[l,c].set_xlim(-0.3, 0.3)
                    ax[l,c].set_ylim(0, 250)
                    '''
                    if cond == 'Claro':
                        ax[l,c].set_ylim(0, 200)
                    elif cond == 'Nublado':
                        ax[l,c].set_ylim(0, 175)
                    else:
                        ax[l,c].set_ylim(0, 175)
                    '''
                    l += 1
            c+=1
        plt.tight_layout()
        plt.show()
        fig.savefig(rf"{path_save}\histograma_{cond}_edlstm.png", dpi = 300, bbox_inches='tight', facecolor='w')













